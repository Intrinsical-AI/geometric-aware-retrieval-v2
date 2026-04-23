#!/usr/bin/env python3
"""Research-only reproducible BEIR benchmark harness.

This script is intentionally scoped to benchmark work, not the public package
surface. It supports:

1. One-off runs against a local BEIR-style dataset or a downloaded BEIR corpus.
2. Deterministic subsampling for FiQA-first comparisons.
3. Stable per-run artifacts plus an aggregate summary derived only from
   ``config.json`` + ``beir_results.json`` pairs found under the results tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from geoIR.core.runner import ExperimentRunner

BEIR_DATASET_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
BENCHMARK_SCHEMA_VERSION = 2
DEFAULT_EXPERIMENT_NAME = "beir_euclidean_vs_geo"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SPLIT = "dev"
RESULT_ROLE_BASELINE = "baseline"
RESULT_ROLE_CANDIDATE = "candidate"
RESULT_STATUS_OK = "ok"
RESULT_METHOD_BASELINE = "Dense cosine baseline"
RESULT_METHOD_SOFT_LOCAL = "Soft graph local"
RESULT_METHOD_SOFT_PPR = "Soft graph + PPR"
RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.01
SUMMARY_REQUIRED_RESULT_KEYS = {
    "method",
    "role",
    "path_kind",
    "status",
    "encode_ms",
    "graph_build_ms",
    "rerank_ms",
    "build_time_ms",
    "eval_time_ms",
    "time_ms",
    "peak_rss_mb",
    "peak_vram_mb",
    "device_name",
    "batch_size",
    "ndcg@10",
    "recall@10",
    "degree_mean",
    "degree_std",
    "entropy",
    "effective_degree",
    "gamma",
    "neighbor_purity@k",
    "edge_overlap_with_dense@k",
}


@dataclass
class DatasetBundle:
    """Normalized local BEIR dataset plus deterministic selections."""

    dataset: str
    dataset_path: Path
    split: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    doc_ids: list[str]
    query_ids: list[str]

    @property
    def doc_texts(self) -> list[str]:
        return [
            " ".join(
                part.strip()
                for part in (
                    self.corpus[doc_id].get("title", ""),
                    self.corpus[doc_id].get("text", ""),
                )
                if part.strip()
            )
            for doc_id in self.doc_ids
        ]

    @property
    def query_texts(self) -> list[str]:
        return [self.queries[query_id] for query_id in self.query_ids]


@dataclass
class PhaseTelemetry:
    """Wall-clock and peak-memory telemetry for one benchmark phase."""

    elapsed_ms: float
    peak_rss_mb: float
    peak_vram_mb: float


@dataclass
class EncodeArtifacts:
    """Embeddings plus encode-phase telemetry reused by both result rows."""

    doc_emb: torch.Tensor
    query_emb: torch.Tensor
    encode_ms: float
    peak_rss_mb: float
    peak_vram_mb: float
    device_name: str
    batch_size: int


class ResourceMonitor:
    """Sample process RSS and CUDA peak VRAM while a phase runs."""

    def __init__(self, device: str, sample_interval: float = RESOURCE_SAMPLE_INTERVAL_SECONDS):
        if psutil is None:  # pragma: no cover
            raise ImportError(
                "psutil is required for benchmark telemetry. "
                "Install the project with the dev extra."
            )
        self.device = device
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self._peak_rss_bytes = int(self.process.memory_info().rss)
        self._peak_vram_bytes = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._cuda_enabled = device.startswith("cuda") and torch.cuda.is_available()

    def _sample_once(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(self.process.memory_info().rss))
        if self._cuda_enabled:
            dev = torch.device(self.device)
            self._peak_vram_bytes = max(
                self._peak_vram_bytes,
                int(torch.cuda.memory_allocated(dev)),
                int(torch.cuda.max_memory_allocated(dev)),
            )

    def _run(self) -> None:
        while self._running:
            self._sample_once()
            time.sleep(self.sample_interval)

    def start(self) -> None:
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        self._running = False
        if self._thread is not None:
            self._thread.join()
        self._sample_once()
        peak_rss_mb = self._peak_rss_bytes / (1024 * 1024)
        if self._cuda_enabled:
            peak_vram_mb = self._peak_vram_bytes / (1024 * 1024)
        else:
            peak_vram_mb = float("nan")
        return peak_rss_mb, peak_vram_mb


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the research-only CLI parser."""
    parser = argparse.ArgumentParser(
        description="Reproducible BEIR benchmark harness for hard vs soft graph retrieval."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fiqa",
        help="BEIR dataset name (e.g. 'fiqa') or an existing local dataset path.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Explicit local BEIR dataset directory. Takes precedence when it exists.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory used for downloaded BEIR datasets.",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--allow-download",
        dest="allow_download",
        action="store_true",
        help="Download the BEIR dataset when no local dataset directory is available.",
    )
    download_group.add_argument(
        "--no-download",
        dest="allow_download",
        action="store_false",
        help="Fail fast instead of downloading when the local dataset is missing.",
    )
    parser.set_defaults(allow_download=False)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to keep after deterministic subsampling.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to keep after deterministic subsampling.",
    )
    parser.add_argument("--k", type=int, default=20, help="Target graph degree.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used for document and query encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding device for the research encoder.",
    )
    parser.add_argument(
        "--rerank",
        type=str,
        default="none",
        choices=["none", "ppr"],
        help=(
            "Soft-graph evaluation mode. 'none' keeps scoring local to k candidates; "
            "'ppr' expands to ppr-topk."
        ),
    )
    parser.add_argument(
        "--ppr-topk",
        type=int,
        default=100,
        help="Candidate set size for PPR reranking when rerank='ppr'.",
    )
    parser.add_argument(
        "--ppr-alpha",
        type=float,
        default=0.2,
        help="Teleport probability for PPR evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser


def seed_all(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str) -> str:
    """Convert a label into a filesystem-safe slug."""
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    return sanitized or "value"


def find_dataset_root(path: Path) -> Path | None:
    """Find the directory containing BEIR corpus/query/qrels files."""
    candidate = path.expanduser()
    if not candidate.exists():
        return None

    if candidate.is_file():
        candidate = candidate.parent

    lineage = [candidate, *candidate.parents]
    for ancestor in lineage:
        if (
            ancestor.is_dir()
            and (ancestor / "corpus.jsonl").exists()
            and (ancestor / "queries.jsonl").exists()
            and (ancestor / "qrels").is_dir()
        ):
            return ancestor.resolve()

    for corpus_path in sorted(candidate.rglob("corpus.jsonl")):
        dataset_root = corpus_path.parent
        if (dataset_root / "queries.jsonl").exists() and (dataset_root / "qrels").is_dir():
            return dataset_root.resolve()
    return None


def download_dataset_archive(dataset: str, download_dir: Path, logger=None) -> Path:
    """Download and unpack a BEIR dataset archive."""
    dataset_name = Path(dataset).name
    target_dir = download_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / f"{dataset_name}.zip"
    archive_url = f"{BEIR_DATASET_BASE_URL}/{dataset_name}.zip"

    if logger is not None:
        logger.info(f"⬇️  Downloading dataset '{dataset_name}' from {archive_url}")

    try:
        urllib.request.urlretrieve(archive_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset '{dataset_name}' from {archive_url}"
        ) from exc

    dataset_root = find_dataset_root(target_dir / dataset_name) or find_dataset_root(target_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            "Downloaded archive for "
            f"'{dataset_name}' but could not find corpus.jsonl/queries.jsonl/qrels."
        )

    if logger is not None:
        logger.info(f"📦 Dataset extracted to {dataset_root}")
    return dataset_root


def resolve_dataset_path(
    dataset: str,
    dataset_dir: Path | None,
    download_dir: Path,
    allow_download: bool,
    *,
    download_dataset_fn: Callable[[str, Path, Any], Path] = download_dataset_archive,
    logger=None,
) -> Path:
    """Resolve a local dataset first, then download if allowed."""
    candidates: list[Path] = []
    if dataset_dir is not None:
        candidates.append(dataset_dir)

    dataset_as_path = Path(dataset).expanduser()
    if dataset_as_path.exists():
        candidates.append(dataset_as_path)

    candidates.append(download_dir.expanduser() / Path(dataset).name)

    checked_locations: list[str] = []
    for candidate in candidates:
        checked_locations.append(str(candidate))
        dataset_root = find_dataset_root(candidate)
        if dataset_root is not None:
            if logger is not None:
                logger.info(f"📦 Using local dataset from {dataset_root}")
            return dataset_root

    if allow_download:
        return download_dataset_fn(dataset, download_dir, logger=logger)

    locations = ", ".join(checked_locations)
    raise FileNotFoundError(
        f"Dataset '{dataset}' was not found locally. Checked: {locations}. "
        "Pass --dataset-dir or rerun with --allow-download."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path} at line {line_number}") from exc
    return records


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load qrels from a BEIR-style TSV file."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if row_number == 1 and row[0].lower().startswith("query"):
                continue
            if len(row) < 3:
                raise ValueError(f"Invalid qrels row in {path} at line {row_number}: {row}")
            query_id, corpus_id, score = row[0], row[1], row[2]
            qrels.setdefault(str(query_id), {})[str(corpus_id)] = int(score)
    return qrels


def load_local_beir_dataset(dataset_path: Path, split: str = DEFAULT_SPLIT) -> tuple[
    dict[str, dict[str, str]],
    dict[str, str],
    dict[str, dict[str, int]],
]:
    """Load a BEIR dataset from local files without relying on external loaders."""
    dataset_root = find_dataset_root(dataset_path)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Could not find a BEIR dataset root under {dataset_path}. "
            "Expected corpus.jsonl, queries.jsonl and qrels/<split>.tsv."
        )

    corpus_records = read_jsonl(dataset_root / "corpus.jsonl")
    query_records = read_jsonl(dataset_root / "queries.jsonl")
    qrels_path = dataset_root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels split at {qrels_path}")

    corpus: dict[str, dict[str, str]] = {}
    for record in corpus_records:
        doc_id = str(record.get("_id") or record.get("id") or "")
        if not doc_id:
            raise ValueError(f"Corpus record missing _id/id: {record}")
        corpus[doc_id] = {
            "title": str(record.get("title", "")),
            "text": str(record.get("text", "")),
        }

    queries: dict[str, str] = {}
    for record in query_records:
        query_id = str(record.get("_id") or record.get("id") or "")
        if not query_id:
            raise ValueError(f"Query record missing _id/id: {record}")
        queries[query_id] = str(record.get("text", ""))

    qrels = load_qrels(qrels_path)
    return corpus, queries, qrels


def deterministic_sample(ids: list[str], limit: int | None, seed: int) -> list[str]:
    """Sample deterministically while keeping output order stable."""
    ordered_ids = sorted(ids)
    if limit is None or limit >= len(ordered_ids):
        return ordered_ids

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(ordered_ids)), limit))
    return [ordered_ids[index] for index in selected_indices]


def subsample_dataset(
    dataset: str,
    dataset_path: Path,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    max_docs: int | None,
    max_queries: int | None,
    seed: int,
    split: str = DEFAULT_SPLIT,
) -> DatasetBundle:
    """Apply deterministic corpus/query subsampling and drop empty qrels."""
    doc_ids = deterministic_sample(list(corpus.keys()), max_docs, seed)
    kept_docs = {doc_id: corpus[doc_id] for doc_id in doc_ids}
    doc_id_set = set(doc_ids)

    query_ids = deterministic_sample(list(queries.keys()), max_queries, seed + 1)
    kept_queries: dict[str, str] = {}
    kept_qrels: dict[str, dict[str, int]] = {}
    kept_query_ids: list[str] = []

    for query_id in query_ids:
        if query_id not in qrels:
            continue
        filtered_docs = {
            doc_id: int(score)
            for doc_id, score in qrels[query_id].items()
            if doc_id in doc_id_set
        }
        if not filtered_docs:
            continue
        kept_queries[query_id] = queries[query_id]
        kept_qrels[query_id] = filtered_docs
        kept_query_ids.append(query_id)

    return DatasetBundle(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        corpus=kept_docs,
        queries=kept_queries,
        qrels=kept_qrels,
        doc_ids=doc_ids,
        query_ids=kept_query_ids,
    )


def default_model_factory(model_name: str, device: str):
    """Lazily construct the research embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def default_encode_texts(
    model: Any,
    texts: list[str],
    *,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode texts into a CPU tensor using a SentenceTransformer-like model."""
    if hasattr(model, "to"):
        model.to(device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings.cpu()


def resolve_device_name(device: str) -> str:
    """Return a stable device label for persisted benchmark rows."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")
        return str(torch.cuda.get_device_name(torch.device(device)))
    if device == "mps":
        return "mps"
    return "cpu"


def measure_phase(
    fn: Callable[..., Any],
    *args: Any,
    monitor_device: str,
    **kwargs: Any,
) -> tuple[Any, PhaseTelemetry]:
    """Measure wall-clock time plus peak RSS/VRAM for one callable."""
    monitor = ResourceMonitor(device=monitor_device)
    start = time.perf_counter()
    monitor.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        peak_rss_mb, peak_vram_mb = monitor.stop()
    elapsed_ms = (time.perf_counter() - start) * 1000
    telemetry = PhaseTelemetry(
        elapsed_ms=float(elapsed_ms),
        peak_rss_mb=float(peak_rss_mb),
        peak_vram_mb=float(peak_vram_mb),
    )
    return result, telemetry


def combine_peak_metrics(*values: float) -> float:
    """Return the max of non-NaN values, or NaN if none are available."""
    valid = [float(value) for value in values if not np.isnan(value)]
    if not valid:
        return float("nan")
    return float(max(valid))


def dense_topk_neighbors(doc_emb: torch.Tensor, k: int) -> list[list[int]]:
    """Return top-k cosine neighbors per document, excluding self."""
    num_docs = doc_emb.shape[0]
    if num_docs == 0:
        return []
    neighbor_k = min(k, num_docs - 1)
    if neighbor_k <= 0:
        return [[] for _ in range(num_docs)]

    normed = F.normalize(doc_emb.to(dtype=torch.float32), dim=1)
    sims = normed @ normed.T
    sims.fill_diagonal_(float("-inf"))
    indices = torch.topk(sims, k=neighbor_k, dim=1).indices
    return [[int(idx) for idx in row.tolist()] for row in indices]


def adjacency_topk_neighbors(adjacency: torch.Tensor, k: int) -> list[list[int]]:
    """Return top-k neighbors per document using adjacency weights."""
    if adjacency.is_sparse:
        dense_adj = adjacency.to_dense()
    else:
        dense_adj = adjacency
    dense_adj = dense_adj.to(dtype=torch.float32).clone()
    num_docs = dense_adj.shape[0]
    neighbor_k = min(k, num_docs - 1)
    if neighbor_k <= 0:
        return [[] for _ in range(num_docs)]

    dense_adj.fill_diagonal_(float("-inf"))
    indices = torch.topk(dense_adj, k=neighbor_k, dim=1).indices
    return [[int(idx) for idx in row.tolist()] for row in indices]


def positive_qrel_doc_sets(qrels: dict[str, dict[str, int]]) -> dict[str, set[str]]:
    """Return the positive qrel query set attached to each document."""
    doc_queries: dict[str, set[str]] = {}
    for query_id, rel_docs in qrels.items():
        for doc_id, score in rel_docs.items():
            if int(score) <= 0:
                continue
            doc_queries.setdefault(str(doc_id), set()).add(str(query_id))
    return doc_queries


def neighbor_purity_at_k(
    *,
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    row_neighbors: list[list[int]],
) -> float:
    """Average fraction of neighbors that share at least one positive qrel query."""
    doc_queries = positive_qrel_doc_sets(qrels)
    scores: list[float] = []

    for doc_index, doc_id in enumerate(doc_ids):
        origin_queries = doc_queries.get(doc_id)
        neighbors = row_neighbors[doc_index]
        if not origin_queries or not neighbors:
            continue

        hits = 0
        for neighbor_index in neighbors:
            neighbor_queries = doc_queries.get(doc_ids[neighbor_index], set())
            if origin_queries & neighbor_queries:
                hits += 1
        scores.append(hits / len(neighbors))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def edge_overlap_with_dense_at_k(
    *,
    dense_neighbors: list[list[int]],
    row_neighbors: list[list[int]],
) -> float:
    """Mean overlap between row neighbors and the dense cosine reference."""
    overlaps: list[float] = []
    for row_neighbor_ids, dense_neighbor_ids in zip(row_neighbors, dense_neighbors, strict=False):
        if not dense_neighbor_ids:
            continue
        overlaps.append(
            len(set(row_neighbor_ids) & set(dense_neighbor_ids)) / len(dense_neighbor_ids)
        )

    if not overlaps:
        return float("nan")
    return float(np.mean(overlaps))


def default_dense_eval_fn(*args, **kwargs):
    from geoIR.eval.metrics import evaluate_retrieval

    return evaluate_retrieval(*args, **kwargs)


def default_soft_local_eval_fn(*args, **kwargs):
    from geoIR.eval.metrics import evaluate_retrieval_soft_local

    return evaluate_retrieval_soft_local(*args, **kwargs)


def default_soft_ppr_eval_fn(*args, **kwargs):
    from geoIR.eval.metrics import evaluate_retrieval_ppr

    return evaluate_retrieval_ppr(*args, **kwargs)


def default_graph_metric_fn(*args, **kwargs):
    from geoIR.eval.metrics import graph_distribution_metrics

    return graph_distribution_metrics(*args, **kwargs)


def default_hard_graph_fn(*args, **kwargs):
    from geoIR.geo.graph import hard_knn_graph_faiss

    return hard_knn_graph_faiss(*args, **kwargs)


def default_soft_graph_fn(*args, **kwargs):
    from geoIR.geo.differentiable import soft_knn_graph

    return soft_knn_graph(*args, **kwargs)


def tensor_cache_paths(
    cache_root: Path,
    dataset: str,
    model_name: str,
    doc_ids: list[str],
    query_ids: list[str],
    seed: int,
) -> tuple[Path, Path]:
    """Return deterministic cache paths derived from the selected ids."""
    digest = hashlib.sha1()
    digest.update(dataset.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    digest.update(str(seed).encode("utf-8"))
    for doc_id in doc_ids:
        digest.update(b"\0doc:")
        digest.update(doc_id.encode("utf-8"))
    for query_id in query_ids:
        digest.update(b"\0qry:")
        digest.update(query_id.encode("utf-8"))
    cache_key = digest.hexdigest()[:12]

    dataset_slug = slugify(dataset)
    model_slug = slugify(model_name)
    target_dir = cache_root.expanduser().resolve() / dataset_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    return (
        target_dir / f"docs_{model_slug}_{cache_key}.pt",
        target_dir / f"queries_{model_slug}_{cache_key}.pt",
    )


def format_results_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize result rows into a stable column order."""
    df = pd.DataFrame(results)
    return df[
        [
            "method",
            "role",
            "path_kind",
            "status",
            "encode_ms",
            "graph_build_ms",
            "rerank_ms",
            "build_time_ms",
            "eval_time_ms",
            "time_ms",
            "peak_rss_mb",
            "peak_vram_mb",
            "device_name",
            "batch_size",
            "ndcg@10",
            "recall@10",
            "degree_mean",
            "degree_std",
            "entropy",
            "effective_degree",
            "gamma",
            "neighbor_purity@k",
            "edge_overlap_with_dense@k",
        ]
    ]


def build_result_row(
    *,
    method: str,
    role: str,
    path_kind: str,
    status: str,
    encode_ms: float,
    graph_build_ms: float,
    rerank_ms: float,
    peak_rss_mb: float,
    peak_vram_mb: float,
    device_name: str,
    batch_size: int,
    ndcg_at_10: float | None,
    recall_at_10: float | None,
    degree_mean: float | None,
    degree_std: float | None,
    entropy: float | None,
    effective_degree: float | None,
    gamma: float | None,
    neighbor_purity_at_k_value: float | None,
    edge_overlap_with_dense_at_k_value: float | None,
) -> dict[str, Any]:
    """Create one normalized benchmark result row."""
    return {
        "method": method,
        "role": role,
        "path_kind": path_kind,
        "status": status,
        "encode_ms": float(encode_ms),
        "graph_build_ms": float(graph_build_ms),
        "rerank_ms": float(rerank_ms),
        "build_time_ms": float(graph_build_ms),
        "eval_time_ms": float(rerank_ms),
        "time_ms": float(encode_ms + graph_build_ms + rerank_ms),
        "peak_rss_mb": np.nan if np.isnan(peak_rss_mb) else float(peak_rss_mb),
        "peak_vram_mb": np.nan if np.isnan(peak_vram_mb) else float(peak_vram_mb),
        "device_name": str(device_name),
        "batch_size": int(batch_size),
        "ndcg@10": np.nan if ndcg_at_10 is None else float(ndcg_at_10),
        "recall@10": np.nan if recall_at_10 is None else float(recall_at_10),
        "degree_mean": np.nan if degree_mean is None else float(degree_mean),
        "degree_std": np.nan if degree_std is None else float(degree_std),
        "entropy": np.nan if entropy is None else float(entropy),
        "effective_degree": np.nan if effective_degree is None else float(effective_degree),
        "gamma": np.nan if gamma is None else float(gamma),
        "neighbor_purity@k": (
            np.nan if neighbor_purity_at_k_value is None else float(neighbor_purity_at_k_value)
        ),
        "edge_overlap_with_dense@k": (
            np.nan
            if edge_overlap_with_dense_at_k_value is None
            else float(edge_overlap_with_dense_at_k_value)
        ),
    }


def safe_degree_std(adjacency: torch.Tensor) -> float:
    """Return a stable degree std even for tiny fixtures."""
    degrees = adjacency.sum(dim=-1)
    if degrees.numel() <= 1:
        return 0.0
    return float(degrees.std(unbiased=False))


def load_summary_record(run_dir: Path) -> dict[str, Any] | None:
    """Load one valid run record from config+results artifacts."""
    config_path = run_dir / "config.json"
    results_path = run_dir / "beir_results.json"
    if not config_path.exists() or not results_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(rows, list):
        return None
    if config.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
        return None

    typed_rows = [row for row in rows if isinstance(row, dict)]
    baseline = next((row for row in typed_rows if row.get("role") == RESULT_ROLE_BASELINE), None)
    candidate = next((row for row in typed_rows if row.get("role") == RESULT_ROLE_CANDIDATE), None)
    if baseline is None or candidate is None:
        return None
    if (
        not SUMMARY_REQUIRED_RESULT_KEYS.issubset(baseline)
        or not SUMMARY_REQUIRED_RESULT_KEYS.issubset(candidate)
    ):
        return None
    if baseline.get("status") != RESULT_STATUS_OK or candidate.get("status") != RESULT_STATUS_OK:
        return None

    rerank = str(config.get("rerank", "none"))
    ppr_topk = config.get("ppr_topk")
    rerank_label = "none" if rerank == "none" else f"ppr@{ppr_topk}"
    record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": config.get("dataset"),
        "model_name": config.get("model_name", DEFAULT_MODEL_NAME),
        "max_docs": config.get("max_docs"),
        "max_queries": config.get("max_queries"),
        "k": config.get("k"),
        "rerank": rerank,
        "rerank_label": rerank_label,
        "hard_graph_ranking_status": config.get("hard_graph_ranking_status", "unknown"),
        "ppr_topk": ppr_topk,
        "ppr_alpha": config.get("ppr_alpha"),
        "seed": config.get("seed"),
        "baseline_method": baseline["method"],
        "candidate_method": candidate["method"],
        "candidate_path_kind": candidate["path_kind"],
        "baseline_encode_ms": baseline["encode_ms"],
        "candidate_encode_ms": candidate["encode_ms"],
        "baseline_graph_build_ms": baseline["graph_build_ms"],
        "candidate_graph_build_ms": candidate["graph_build_ms"],
        "baseline_rerank_ms": baseline["rerank_ms"],
        "candidate_rerank_ms": candidate["rerank_ms"],
        "baseline_time_ms": baseline["time_ms"],
        "candidate_time_ms": candidate["time_ms"],
        "baseline_peak_rss_mb": baseline["peak_rss_mb"],
        "candidate_peak_rss_mb": candidate["peak_rss_mb"],
        "baseline_peak_vram_mb": baseline["peak_vram_mb"],
        "candidate_peak_vram_mb": candidate["peak_vram_mb"],
        "baseline_device_name": baseline["device_name"],
        "candidate_device_name": candidate["device_name"],
        "baseline_batch_size": baseline["batch_size"],
        "candidate_batch_size": candidate["batch_size"],
        "baseline_ndcg_at_10": baseline["ndcg@10"],
        "candidate_ndcg_at_10": candidate["ndcg@10"],
        "delta_ndcg_at_10": candidate["ndcg@10"] - baseline["ndcg@10"],
        "baseline_recall_at_10": baseline["recall@10"],
        "candidate_recall_at_10": candidate["recall@10"],
        "delta_recall_at_10": candidate["recall@10"] - baseline["recall@10"],
        "baseline_degree_mean": baseline["degree_mean"],
        "candidate_degree_mean": candidate["degree_mean"],
        "baseline_degree_std": baseline["degree_std"],
        "candidate_degree_std": candidate["degree_std"],
        "baseline_entropy": baseline["entropy"],
        "candidate_entropy": candidate["entropy"],
        "baseline_effective_degree": baseline["effective_degree"],
        "candidate_effective_degree": candidate["effective_degree"],
        "baseline_neighbor_purity_at_k": baseline["neighbor_purity@k"],
        "candidate_neighbor_purity_at_k": candidate["neighbor_purity@k"],
        "baseline_edge_overlap_with_dense_at_k": baseline["edge_overlap_with_dense@k"],
        "candidate_edge_overlap_with_dense_at_k": candidate["edge_overlap_with_dense@k"],
        "candidate_gamma": candidate["gamma"],
    }
    return record


def build_summary_df(experiment_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build a summary dataframe from valid run artifacts only."""
    records: list[dict[str, Any]] = []
    invalid_runs: list[str] = []
    if not experiment_dir.exists():
        return pd.DataFrame(), invalid_runs

    for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        record = load_summary_record(run_dir)
        if record is None:
            invalid_runs.append(run_dir.name)
            continue
        records.append(record)

    if not records:
        return pd.DataFrame(), invalid_runs

    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["dataset", "max_docs", "rerank", "ppr_topk", "run_id"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    return df, invalid_runs


def latest_summary_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest run per configuration tuple."""
    if summary_df.empty:
        return summary_df
    latest_df = summary_df.sort_values("run_id").drop_duplicates(
        subset=[
            "dataset",
            "max_docs",
            "max_queries",
            "k",
            "rerank",
            "ppr_topk",
            "ppr_alpha",
            "seed",
        ],
        keep="last",
    )
    return latest_df.sort_values(
        by=["dataset", "max_docs", "rerank", "ppr_topk", "run_id"]
    ).reset_index(drop=True)


def format_metric(value: float | int | None, digits: int = 4) -> str:
    """Format floats consistently for markdown output."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown_table(summary_df: pd.DataFrame) -> str:
    """Render a compact markdown table from the latest run per config."""
    if summary_df.empty:
        return "No valid run artifacts found.\n"

    headers = [
        "Dataset",
        "#Docs",
        "Rerank",
        "Baseline",
        "Candidate",
        "Baseline nDCG@10",
        "Candidate nDCG@10",
        "Δ nDCG@10",
        "Baseline Recall@10",
        "Candidate Recall@10",
        "Δ Recall@10",
        "Baseline ms",
        "Candidate ms",
        "Cand encode ms",
        "Cand build ms",
        "Cand rerank ms",
        "Cand RSS MB",
        "Cand VRAM MB",
        "Cand purity@k",
        "Cand overlap@k",
        "γ",
        "Run",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    f"{int(row.max_docs):,}" if row.max_docs is not None else "all",
                    str(row.rerank_label),
                    str(row.baseline_method),
                    str(row.candidate_method),
                    format_metric(row.baseline_ndcg_at_10),
                    format_metric(row.candidate_ndcg_at_10),
                    format_metric(row.delta_ndcg_at_10),
                    format_metric(row.baseline_recall_at_10),
                    format_metric(row.candidate_recall_at_10),
                    format_metric(row.delta_recall_at_10),
                    format_metric(row.baseline_time_ms, digits=1),
                    format_metric(row.candidate_time_ms, digits=1),
                    format_metric(row.candidate_encode_ms, digits=1),
                    format_metric(row.candidate_graph_build_ms, digits=1),
                    format_metric(row.candidate_rerank_ms, digits=1),
                    format_metric(row.candidate_peak_rss_mb, digits=1),
                    format_metric(row.candidate_peak_vram_mb, digits=1),
                    format_metric(row.candidate_neighbor_purity_at_k),
                    format_metric(row.candidate_edge_overlap_with_dense_at_k),
                    format_metric(row.candidate_gamma, digits=3),
                    row.run_id,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_decision_gate(summary_df: pd.DataFrame) -> str:
    """Render the phase decision gate from the latest FiQA runs."""
    if summary_df.empty:
        return "- No valid runs yet; decision gate is pending.\n"

    fiqa = summary_df[summary_df["dataset"] == "fiqa"]
    if fiqa.empty:
        return "- No FiQA-backed runs found yet; decision gate is pending.\n"

    lines: list[str] = []

    row_1k = fiqa[
        (fiqa["max_docs"] == 1000)
        & (fiqa["rerank"] == "none")
        & (fiqa["candidate_path_kind"] == "soft_local")
    ]
    if row_1k.empty:
        lines.append("- Soft-kNN 1k gate: pending; missing FiQA 1k run with `rerank=none`.")
    else:
        item = row_1k.iloc[-1]
        passes = item["delta_recall_at_10"] >= -0.02 and item["delta_ndcg_at_10"] >= 0.01
        verdict = "keep soft-kNN alive" if passes else "do not promote soft-kNN yet"
        lines.append(
            "- Soft-kNN 1k gate: "
            f"{verdict} (`Δ nDCG@10={format_metric(item['delta_ndcg_at_10'])}`, "
            f"`Δ Recall@10={format_metric(item['delta_recall_at_10'])}`)."
        )

    row_5k = fiqa[
        (fiqa["max_docs"] == 5000)
        & (fiqa["rerank"] == "none")
        & (fiqa["candidate_path_kind"] == "soft_local")
    ]
    if row_5k.empty:
        lines.append("- Soft-kNN 5k gate: pending; missing FiQA 5k run with `rerank=none`.")
    else:
        item = row_5k.iloc[-1]
        if item["delta_ndcg_at_10"] < -0.03:
            lines.append(
                "- Soft-kNN 5k gate: trigger geometric redesign "
                f"(`Δ nDCG@10={format_metric(item['delta_ndcg_at_10'])}`)."
            )
        else:
            lines.append(
                "- Soft-kNN 5k gate: keep current geometry investigation open "
                f"(`Δ nDCG@10={format_metric(item['delta_ndcg_at_10'])}`)."
            )

    ppr_rows = fiqa[
        (fiqa["candidate_path_kind"] == "soft_ppr")
        & (fiqa["max_docs"].isin([1000, 5000]))
    ]
    if ppr_rows.empty or len(set(ppr_rows["max_docs"])) < 2:
        lines.append("- PPR gate: pending; missing FiQA PPR runs for both 1k and 5k.")
    elif (ppr_rows["delta_ndcg_at_10"] < 0).all():
        lines.append(
            "- PPR gate: freeze PPR for the next cycle; every FiQA PPR run loses "
            "against the dense baseline."
        )
    else:
        lines.append(
            "- PPR gate: keep open; at least one FiQA PPR run does not lose "
            "against the dense baseline."
        )

    return "\n".join(lines) + "\n"


def render_summary_markdown(summary_df: pd.DataFrame, invalid_runs: list[str]) -> str:
    """Render the generated benchmark summary."""
    latest_df = latest_summary_view(summary_df)
    lines = [
        "# BEIR Benchmark Summary",
        "",
        "This file is generated only from `config.json` + `beir_results.json` artifacts.",
        (
            f"It only accepts benchmark schema v{BENCHMARK_SCHEMA_VERSION}; "
            "legacy result rows are skipped."
        ),
        (
            "Historical manual notes elsewhere in the repo are non-authoritative "
            "and intentionally excluded here."
        ),
        (
            "That includes the previous `msmarco-passage` claims, which are "
            "ignored until a backed run exists."
        ),
        (
            "Hard-graph ranking is currently explicit as absent; the supported "
            "comparison is dense cosine baseline vs one candidate path."
        ),
        "",
        f"Valid runs discovered: `{len(summary_df)}`",
        "",
        "## Latest Backed Runs",
        "",
        render_markdown_table(latest_df).rstrip(),
        "",
        "## Decision Gate",
        "",
        render_decision_gate(latest_df).rstrip(),
        "",
    ]
    if invalid_runs:
        lines.extend(
            [
                "## Skipped Runs",
                "",
                (
                    "These directories were ignored because they were missing a "
                    "valid `config.json` + `beir_results.json` pair:"
                ),
                "",
                *[f"- `{run_id}`" for run_id in invalid_runs],
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_summary_artifacts(experiment_dir: Path) -> tuple[Path, Path]:
    """Regenerate summary artifacts from backed run directories."""
    experiment_root = experiment_dir.expanduser().resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)
    summary_df, invalid_runs = build_summary_df(experiment_root)

    summary_csv = experiment_root / "summary.csv"
    summary_md = experiment_root / "SUMMARY.md"

    if summary_df.empty:
        summary_csv.write_text("run_id,dataset\n", encoding="utf-8")
    else:
        summary_df.to_csv(summary_csv, index=False)

    summary_md.write_text(render_summary_markdown(summary_df, invalid_runs), encoding="utf-8")
    return summary_csv, summary_md


class BeirExperimentRunner(ExperimentRunner):
    """Single-run benchmark harness with injectable dependencies for tests."""

    def __init__(
        self,
        *,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        base_output_dir: str | Path = "research/results",
        cache_root: str | Path = "embeddings",
        model_name: str = DEFAULT_MODEL_NAME,
        model_factory: Callable[[str, str], Any] = default_model_factory,
        encode_fn: Callable[..., torch.Tensor] = default_encode_texts,
        dense_eval_fn: Callable[..., tuple[float, float]] = default_dense_eval_fn,
        soft_local_eval_fn: Callable[..., tuple[float, float]] = default_soft_local_eval_fn,
        soft_ppr_eval_fn: Callable[..., tuple[float, float]] = default_soft_ppr_eval_fn,
        hard_graph_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]] = default_hard_graph_fn,
        soft_graph_fn: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, dict[str, float]]
        ] = default_soft_graph_fn,
        graph_metric_fn: Callable[..., dict[str, float]] = default_graph_metric_fn,
        download_dataset_fn: Callable[[str, Path, Any], Path] = download_dataset_archive,
    ):
        super().__init__(experiment_name=experiment_name, base_output_dir=str(base_output_dir))
        self.args: argparse.Namespace | None = None
        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.model_factory = model_factory
        self.encode_fn = encode_fn
        self.dense_eval_fn = dense_eval_fn
        self.soft_local_eval_fn = soft_local_eval_fn
        self.soft_ppr_eval_fn = soft_ppr_eval_fn
        self.hard_graph_fn = hard_graph_fn
        self.soft_graph_fn = soft_graph_fn
        self.graph_metric_fn = graph_metric_fn
        self.download_dataset_fn = download_dataset_fn

    @property
    def experiment_root(self) -> Path:
        return self.base_output_dir / self.experiment_name

    def _config_snapshot(self) -> dict[str, Any]:
        if self.args is None:
            raise RuntimeError("Arguments not parsed")
        return {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "dataset": self.args.dataset,
            "dataset_dir": (
                str(self.args.dataset_dir) if self.args.dataset_dir is not None else None
            ),
            "download_dir": str(self.args.download_dir),
            "allow_download": bool(self.args.allow_download),
            "max_docs": self.args.max_docs,
            "max_queries": self.args.max_queries,
            "k": self.args.k,
            "device": self.args.device,
            "rerank": self.args.rerank,
            "ppr_topk": self.args.ppr_topk,
            "ppr_alpha": self.args.ppr_alpha,
            "batch_size": self.args.batch_size,
            "seed": self.args.seed,
            "split": DEFAULT_SPLIT,
            "model_name": self.model_name,
            "hard_graph_ranking_status": "absent",
        }

    def _load_dataset_bundle(self) -> DatasetBundle:
        if self.args is None:
            raise RuntimeError("Arguments not parsed")

        dataset_path = resolve_dataset_path(
            self.args.dataset,
            self.args.dataset_dir,
            self.args.download_dir,
            self.args.allow_download,
            download_dataset_fn=self.download_dataset_fn,
            logger=self.logger,
        )
        corpus, queries, qrels = load_local_beir_dataset(dataset_path, split=DEFAULT_SPLIT)
        self.logger.info(f"Original sizes – Docs: {len(corpus):,}, Queries: {len(queries):,}")

        bundle = subsample_dataset(
            self.args.dataset,
            dataset_path,
            corpus,
            queries,
            qrels,
            max_docs=self.args.max_docs,
            max_queries=self.args.max_queries,
            seed=self.args.seed,
            split=DEFAULT_SPLIT,
        )
        self.logger.info(
            f"Subset sizes – Docs: {len(bundle.doc_ids):,}, "
            f"Queries: {len(bundle.query_ids):,}"
        )
        if not bundle.qrels:
            raise ValueError(
                "No relevant query-document pairs remain after deterministic subsampling."
            )
        return bundle

    def _load_embeddings(self, bundle: DatasetBundle) -> EncodeArtifacts:
        doc_emb_path, query_emb_path = tensor_cache_paths(
            self.cache_root,
            bundle.dataset,
            self.model_name,
            bundle.doc_ids,
            bundle.query_ids,
            self.args.seed,
        )

        encode_ms = 0.0
        peak_rss_mb = float("nan")
        peak_vram_mb = float("nan")
        model = None
        if doc_emb_path.exists():
            self.logger.info("🔁 Loading cached document embeddings…")
            doc_emb = torch.load(doc_emb_path)
        else:
            model = self.model_factory(self.model_name, self.args.device)
            self.logger.info(f"🔄 Encoding {len(bundle.doc_texts)} documents…")
            doc_emb, phase = measure_phase(
                self.encode_fn,
                model,
                bundle.doc_texts,
                monitor_device=self.args.device,
                device=self.args.device,
                batch_size=self.args.batch_size,
            )
            torch.save(doc_emb, doc_emb_path)
            encode_ms += phase.elapsed_ms
            peak_rss_mb = combine_peak_metrics(peak_rss_mb, phase.peak_rss_mb)
            peak_vram_mb = combine_peak_metrics(peak_vram_mb, phase.peak_vram_mb)

        if query_emb_path.exists():
            self.logger.info("🔁 Loading cached query embeddings…")
            query_emb = torch.load(query_emb_path)
        else:
            model = model or self.model_factory(self.model_name, self.args.device)
            self.logger.info(f"🔄 Encoding {len(bundle.query_texts)} queries…")
            query_emb, phase = measure_phase(
                self.encode_fn,
                model,
                bundle.query_texts,
                monitor_device=self.args.device,
                device=self.args.device,
                batch_size=self.args.batch_size,
            )
            torch.save(query_emb, query_emb_path)
            encode_ms += phase.elapsed_ms
            peak_rss_mb = combine_peak_metrics(peak_rss_mb, phase.peak_rss_mb)
            peak_vram_mb = combine_peak_metrics(peak_vram_mb, phase.peak_vram_mb)

        return EncodeArtifacts(
            doc_emb=doc_emb.cpu(),
            query_emb=query_emb.cpu(),
            encode_ms=float(encode_ms),
            peak_rss_mb=float(peak_rss_mb),
            peak_vram_mb=float(peak_vram_mb),
            device_name=resolve_device_name(self.args.device),
            batch_size=int(self.args.batch_size),
        )

    def _run_dense_baseline(
        self,
        doc_emb: torch.Tensor,
        query_emb: torch.Tensor,
        bundle: DatasetBundle,
        *,
        encode_artifacts: EncodeArtifacts,
        dense_neighbors: list[list[int]],
    ) -> dict[str, Any]:
        self.logger.info("📐 Evaluating dense cosine baseline…")
        (ndcg, recall), rerank = measure_phase(
            self.dense_eval_fn,
            query_emb,
            doc_emb,
            bundle.qrels,
            bundle.doc_ids,
            monitor_device=self.args.device,
        )
        return build_result_row(
            method=RESULT_METHOD_BASELINE,
            role=RESULT_ROLE_BASELINE,
            path_kind="dense_cosine",
            status=RESULT_STATUS_OK,
            encode_ms=encode_artifacts.encode_ms,
            graph_build_ms=0.0,
            rerank_ms=rerank.elapsed_ms,
            peak_rss_mb=combine_peak_metrics(encode_artifacts.peak_rss_mb, rerank.peak_rss_mb),
            peak_vram_mb=combine_peak_metrics(encode_artifacts.peak_vram_mb, rerank.peak_vram_mb),
            device_name=encode_artifacts.device_name,
            batch_size=encode_artifacts.batch_size,
            ndcg_at_10=ndcg,
            recall_at_10=recall,
            degree_mean=None,
            degree_std=None,
            entropy=None,
            effective_degree=None,
            gamma=None,
            neighbor_purity_at_k_value=neighbor_purity_at_k(
                doc_ids=bundle.doc_ids,
                qrels=bundle.qrels,
                row_neighbors=dense_neighbors,
            ),
            edge_overlap_with_dense_at_k_value=edge_overlap_with_dense_at_k(
                dense_neighbors=dense_neighbors,
                row_neighbors=dense_neighbors,
            ),
        )

    def _run_soft_graph(
        self,
        doc_emb: torch.Tensor,
        query_emb: torch.Tensor,
        bundle: DatasetBundle,
        *,
        encode_artifacts: EncodeArtifacts,
        dense_neighbors: list[list[int]],
    ) -> dict[str, Any]:
        self.logger.info("✨ Building soft k-NN τ-fix graph…")
        (_, adjacency, diagnostics), build = measure_phase(
            self.soft_graph_fn,
            doc_emb,
            monitor_device=self.args.device,
            k=self.args.k,
            return_adjacency=True,
            return_diagnostics=True,
        )

        if self.args.rerank == "ppr":
            method = RESULT_METHOD_SOFT_PPR
            path_kind = "soft_ppr"
            eval_fn = self.soft_ppr_eval_fn
            eval_kwargs = {"topk": self.args.ppr_topk, "alpha": self.args.ppr_alpha}
        else:
            method = RESULT_METHOD_SOFT_LOCAL
            path_kind = "soft_local"
            eval_fn = self.soft_local_eval_fn
            eval_kwargs = {"topk": self.args.k}

        (ndcg, recall), rerank = measure_phase(
            eval_fn,
            query_emb,
            doc_emb,
            bundle.qrels,
            bundle.doc_ids,
            adjacency,
            monitor_device=self.args.device,
            **eval_kwargs,
        )
        candidate_neighbors = adjacency_topk_neighbors(adjacency, self.args.k)
        distribution = self.graph_metric_fn(adjacency)
        return build_result_row(
            method=method,
            role=RESULT_ROLE_CANDIDATE,
            path_kind=path_kind,
            status=RESULT_STATUS_OK,
            encode_ms=encode_artifacts.encode_ms,
            graph_build_ms=build.elapsed_ms,
            rerank_ms=rerank.elapsed_ms,
            peak_rss_mb=combine_peak_metrics(
                encode_artifacts.peak_rss_mb,
                build.peak_rss_mb,
                rerank.peak_rss_mb,
            ),
            peak_vram_mb=combine_peak_metrics(
                encode_artifacts.peak_vram_mb,
                build.peak_vram_mb,
                rerank.peak_vram_mb,
            ),
            device_name=encode_artifacts.device_name,
            batch_size=encode_artifacts.batch_size,
            ndcg_at_10=ndcg,
            recall_at_10=recall,
            degree_mean=float(diagnostics["actual_degree"]),
            degree_std=safe_degree_std(adjacency),
            entropy=float(distribution["entropy"]),
            effective_degree=float(distribution["effective_degree"]),
            gamma=float(diagnostics["gamma_used"]),
            neighbor_purity_at_k_value=neighbor_purity_at_k(
                doc_ids=bundle.doc_ids,
                qrels=bundle.qrels,
                row_neighbors=candidate_neighbors,
            ),
            edge_overlap_with_dense_at_k_value=edge_overlap_with_dense_at_k(
                dense_neighbors=dense_neighbors,
                row_neighbors=candidate_neighbors,
            ),
        )

    def _report_results(self, results: list[dict[str, Any]]) -> None:
        df = format_results_df(results)
        self.logger.info("\n📊 RESULTS:")
        self.logger.info(df.to_string(index=False, float_format="{:.4f}".format))
        self.save_results(df.to_dict(orient="records"), "beir_results.json")
        self.save_dataframe(df, "beir_results.csv")
        write_summary_artifacts(self.experiment_root)
        self.logger.info(f"✅ Results saved to {self.run_dir}")

    def run(self) -> None:
        bundle = self._load_dataset_bundle()
        encode_artifacts = self._load_embeddings(bundle)
        dense_neighbors = dense_topk_neighbors(encode_artifacts.doc_emb, self.args.k)
        results = [
            self._run_dense_baseline(
                encode_artifacts.doc_emb,
                encode_artifacts.query_emb,
                bundle,
                encode_artifacts=encode_artifacts,
                dense_neighbors=dense_neighbors,
            ),
            self._run_soft_graph(
                encode_artifacts.doc_emb,
                encode_artifacts.query_emb,
                bundle,
                encode_artifacts=encode_artifacts,
                dense_neighbors=dense_neighbors,
            ),
        ]
        self._report_results(results)

    def start(self, argv: list[str] | None = None) -> None:
        self.args = build_argument_parser().parse_args(argv)
        seed_all(self.args.seed)
        super().start(config=self._config_snapshot())


def main(argv: list[str] | None = None) -> None:
    runner = BeirExperimentRunner()
    runner.start(argv)


if __name__ == "__main__":
    main()
