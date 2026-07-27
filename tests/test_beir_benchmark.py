from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MODULE_PATH = REPO_ROOT / "research" / "beir_euclidean_vs_geo.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("beir_benchmark_module", BENCHMARK_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_beir_fixture(root: Path) -> Path:
    dataset_dir = root / "fiqa-fixture"
    (dataset_dir / "qrels").mkdir(parents=True)

    corpus_rows = [
        {"_id": "d1", "title": "alpha", "text": "red planet"},
        {"_id": "d2", "title": "beta", "text": "blue ocean"},
        {"_id": "d3", "title": "gamma", "text": "green forest"},
    ]
    queries_rows = [
        {"_id": "q1", "text": "planet"},
        {"_id": "q2", "text": "forest"},
    ]
    qrels_rows = [
        "query-id\tcorpus-id\tscore",
        "q1\td1\t1",
        "q1\td2\t1",
        "q2\td3\t1",
    ]

    (dataset_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row) for row in corpus_rows) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "queries.jsonl").write_text(
        "\n".join(json.dumps(row) for row in queries_rows) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "qrels" / "dev.tsv").write_text("\n".join(qrels_rows) + "\n", encoding="utf-8")
    return dataset_dir


class FakeModel:
    def to(self, _device: str) -> "FakeModel":
        return self

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        del batch_size, show_progress_bar, convert_to_tensor
        rows = []
        mapping = {
            "alpha red planet": [1.0, 0.0, 0.0],
            "beta blue ocean": [0.9, 0.1, 0.0],
            "gamma green forest": [0.0, 1.0, 0.0],
            "planet": [1.0, 0.0, 0.0],
            "forest": [0.0, 1.0, 0.0],
        }
        for text in texts:
            rows.append(
                mapping.get(
                    text,
                    [float(len(text)), float(sum(ord(ch) for ch in text) % 17), 1.0],
                )
            )
        return torch.tensor(rows, dtype=torch.float32)


class SilentLogger:
    def info(self, *_args, **_kwargs) -> None:
        pass


def fake_model_factory(_model_name: str, _device: str) -> FakeModel:
    return FakeModel()


def fake_hard_graph_fn(embeddings: torch.Tensor, k: int):
    size = embeddings.shape[0]
    return torch.eye(size, dtype=torch.float32), {
        "degree_mean": float(k),
        "degree_std": 0.0,
        "entropy": 1.0,
        "effective_degree": float(k),
    }


def fake_soft_graph_fn(
    embeddings: torch.Tensor,
    *,
    k: int,
    return_adjacency: bool,
    return_diagnostics: bool,
):
    del return_adjacency, return_diagnostics
    size = embeddings.shape[0]
    adjacency = torch.zeros((size, size), dtype=torch.float32)
    if size == 3:
        adjacency = torch.tensor(
            [
                [0.0, 0.9, 0.1],
                [0.8, 0.0, 0.2],
                [0.9, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )
    weights = adjacency.clone()
    diagnostics = {"actual_degree": float(k), "gamma_used": 0.5}
    return weights, adjacency, diagnostics


def fake_dense_eval_fn(query_emb, doc_emb, qrels, doc_ids, k_eval: int = 10):
    del query_emb, doc_emb, qrels, doc_ids, k_eval
    return 0.70, 0.80


def fake_soft_local_eval_fn(query_emb, doc_emb, qrels, doc_ids, adjacency, topk: int = 100):
    del query_emb, doc_emb, qrels, doc_ids, adjacency
    if topk <= 2:
        return 0.72, 0.79
    return 0.69, 0.78


def fake_soft_ppr_eval_fn(
    query_emb,
    doc_emb,
    qrels,
    doc_ids,
    adjacency,
    topk: int = 100,
    alpha: float = 0.2,
):
    del query_emb, doc_emb, qrels, doc_ids, adjacency, alpha
    if topk <= 100:
        return 0.68, 0.77
    return 0.66, 0.75


def fake_graph_metric_fn(adjacency: torch.Tensor):
    del adjacency
    return {"entropy": 1.23, "effective_degree": 2.34}


def write_v2_run_artifacts(
    root: Path,
    run_id: str,
    *,
    dataset: str,
    max_docs: int,
    rerank: str,
    ppr_topk: int,
) -> None:
    module = load_benchmark_module()
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {
        "benchmark_schema_version": module.BENCHMARK_SCHEMA_VERSION,
        "dataset": dataset,
        "max_docs": max_docs,
        "max_queries": 100,
        "k": 20,
        "rerank": rerank,
        "ppr_topk": ppr_topk,
        "ppr_alpha": 0.2,
        "batch_size": 256,
        "seed": 42,
        "model_name": "fake-model",
        "hard_graph_ranking_status": "absent",
    }
    candidate_method = "Soft graph local" if rerank == "none" else "Soft graph + PPR"
    candidate_path_kind = "soft_local" if rerank == "none" else "soft_ppr"
    results = [
        {
            "method": "Dense cosine baseline",
            "role": "baseline",
            "path_kind": "dense_cosine",
            "status": "ok",
            "encode_ms": 4.0,
            "graph_build_ms": 0.0,
            "rerank_ms": 10.0,
            "build_time_ms": 0.0,
            "eval_time_ms": 10.0,
            "time_ms": 14.0,
            "peak_rss_mb": 123.0,
            "peak_vram_mb": None,
            "device_name": "cpu",
            "batch_size": 256,
            "ndcg@10": 0.70,
            "recall@10": 0.80,
            "degree_mean": None,
            "degree_std": None,
            "entropy": None,
            "effective_degree": None,
            "gamma": None,
            "neighbor_purity@k": 0.66,
            "edge_overlap_with_dense@k": 1.0,
        },
        {
            "method": candidate_method,
            "role": "candidate",
            "path_kind": candidate_path_kind,
            "status": "ok",
            "encode_ms": 4.0,
            "graph_build_ms": 15.0,
            "rerank_ms": 10.0,
            "build_time_ms": 15.0,
            "eval_time_ms": 10.0,
            "time_ms": 29.0,
            "peak_rss_mb": 145.0,
            "peak_vram_mb": None,
            "device_name": "cpu",
            "batch_size": 256,
            "ndcg@10": 0.72 if rerank == "none" else 0.68,
            "recall@10": 0.79 if rerank == "none" else 0.77,
            "degree_mean": 20.0,
            "degree_std": 0.0,
            "entropy": 1.2,
            "effective_degree": 21.0,
            "gamma": 0.5,
            "neighbor_purity@k": 0.66,
            "edge_overlap_with_dense@k": 0.67,
        },
    ]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "beir_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_v1_run_artifacts(root: Path, run_id: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {"dataset": "fiqa", "max_docs": 1000, "rerank": "none"}
    results = [
        {"method": "Hard k-NN", "time_ms": 10.0, "ndcg@10": 0.7, "recall@10": 0.8},
        {"method": "Soft k-NN τ-fix", "time_ms": 25.0, "ndcg@10": 0.72, "recall@10": 0.79},
    ]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "beir_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def test_find_dataset_root_detects_fixture(tmp_path: Path) -> None:
    module = load_benchmark_module()
    dataset_dir = write_beir_fixture(tmp_path)
    nested = dataset_dir / "nested"
    nested.mkdir()

    assert module.find_dataset_root(dataset_dir) == dataset_dir.resolve()
    assert module.find_dataset_root(nested) == dataset_dir.resolve()


def test_load_local_beir_dataset_reads_fixture(tmp_path: Path) -> None:
    module = load_benchmark_module()
    dataset_dir = write_beir_fixture(tmp_path)

    corpus, queries, qrels = module.load_local_beir_dataset(dataset_dir)

    assert list(sorted(corpus)) == ["d1", "d2", "d3"]
    assert queries == {"q1": "planet", "q2": "forest"}
    assert qrels == {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}


def test_deterministic_sample_is_stable() -> None:
    module = load_benchmark_module()
    ids = ["d3", "d1", "d2", "d4"]

    first = module.deterministic_sample(ids, 2, 42)
    second = module.deterministic_sample(ids, 2, 42)
    third = module.deterministic_sample(ids, 2, 7)

    assert first == second
    assert first != third
    assert first == sorted(first)


def test_subsample_dataset_samples_only_queries_with_kept_qrels(tmp_path: Path) -> None:
    module = load_benchmark_module()
    corpus = {
        "d1": {"title": "alpha", "text": "red planet"},
        "d2": {"title": "beta", "text": "blue ocean"},
    }
    queries = {
        "q1": "missing relevant doc",
        "q2": "planet",
    }
    qrels = {
        "q1": {"d3": 1},
        "q2": {"d1": 1},
    }

    bundle = module.subsample_dataset(
        "fiqa",
        tmp_path,
        corpus,
        queries,
        qrels,
        max_docs=None,
        max_queries=10,
        seed=42,
    )

    assert bundle.query_ids == ["q2"]
    assert bundle.qrels == {"q2": {"d1": 1}}


def test_tensor_cache_paths_are_deterministic(tmp_path: Path) -> None:
    module = load_benchmark_module()

    first = module.tensor_cache_paths(tmp_path, "fiqa", "fake-model", ["d1", "d2"], ["q1"], 42)
    second = module.tensor_cache_paths(tmp_path, "fiqa", "fake-model", ["d1", "d2"], ["q1"], 42)
    third = module.tensor_cache_paths(tmp_path, "fiqa", "fake-model", ["d2", "d3"], ["q1"], 42)

    assert first == second
    assert first != third


def test_resolve_dataset_path_prefers_local_dir(tmp_path: Path) -> None:
    module = load_benchmark_module()
    dataset_dir = write_beir_fixture(tmp_path)
    download_calls: list[tuple[str, Path]] = []

    def fake_download(dataset: str, download_dir: Path, logger=None) -> Path:
        del logger
        download_calls.append((dataset, download_dir))
        return download_dir / dataset

    resolved = module.resolve_dataset_path(
        "fiqa",
        dataset_dir,
        tmp_path / "downloads",
        False,
        download_dataset_fn=fake_download,
    )

    assert resolved == dataset_dir.resolve()
    assert download_calls == []


def test_resolve_dataset_path_errors_when_download_disabled(tmp_path: Path) -> None:
    module = load_benchmark_module()

    with pytest.raises(FileNotFoundError, match="allow-download"):
        module.resolve_dataset_path(
            "fiqa",
            tmp_path / "missing-fiqa",
            tmp_path / "downloads",
            False,
        )


def test_beir_runner_creates_run_artifacts_and_summary(tmp_path: Path) -> None:
    module = load_benchmark_module()
    dataset_dir = write_beir_fixture(tmp_path)
    runner = module.BeirExperimentRunner(
        base_output_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        model_name="fake-model",
        model_factory=fake_model_factory,
        dense_eval_fn=fake_dense_eval_fn,
        soft_local_eval_fn=fake_soft_local_eval_fn,
        soft_ppr_eval_fn=fake_soft_ppr_eval_fn,
        hard_graph_fn=fake_hard_graph_fn,
        soft_graph_fn=fake_soft_graph_fn,
        graph_metric_fn=fake_graph_metric_fn,
    )

    runner.start(
        [
            "--dataset",
            "fiqa",
            "--dataset-dir",
            str(dataset_dir),
            "--download-dir",
            str(tmp_path / "downloads"),
            "--no-download",
            "--max-docs",
            "3",
            "--max-queries",
            "2",
            "--k",
            "1",
            "--batch-size",
            "32",
            "--rerank",
            "none",
            "--seed",
            "7",
        ]
    )

    assert runner.run_dir is not None
    for artifact in ("config.json", "beir_results.json", "beir_results.csv", "run.log"):
        assert (runner.run_dir / artifact).exists()

    results_rows = json.loads((runner.run_dir / "beir_results.json").read_text(encoding="utf-8"))
    assert [row["method"] for row in results_rows] == ["Dense cosine baseline", "Soft graph local"]
    assert results_rows[0]["role"] == "baseline"
    assert results_rows[1]["role"] == "candidate"
    assert results_rows[0]["build_time_ms"] == 0.0
    assert results_rows[0]["batch_size"] == 32
    assert results_rows[1]["device_name"] == "cpu"
    assert results_rows[0]["edge_overlap_with_dense@k"] == pytest.approx(1.0)
    assert results_rows[1]["edge_overlap_with_dense@k"] == pytest.approx(2.0 / 3.0)
    assert results_rows[1]["neighbor_purity@k"] == pytest.approx(2.0 / 3.0)
    assert math.isnan(results_rows[0]["peak_vram_mb"])
    assert math.isnan(results_rows[1]["peak_vram_mb"])
    assert results_rows[0]["encode_ms"] >= 0.0
    assert results_rows[1]["graph_build_ms"] >= 0.0
    assert results_rows[1]["rerank_ms"] >= 0.0

    summary_root = tmp_path / "results" / "beir_euclidean_vs_geo"
    assert (summary_root / "summary.csv").exists()
    assert (summary_root / "SUMMARY.md").exists()
    summary_df = pd.read_csv(summary_root / "summary.csv")
    assert "candidate_peak_rss_mb" in summary_df.columns
    assert "candidate_neighbor_purity_at_k" in summary_df.columns
    assert "candidate_edge_overlap_with_dense_at_k" in summary_df.columns
    summary_text = (summary_root / "SUMMARY.md").read_text(encoding="utf-8")
    assert "schema v2" in summary_text
    assert "Cand RSS MB" in summary_text
    assert "Cand overlap@k" in summary_text
    assert "Hard-graph ranking is currently explicit as absent" in summary_text


def test_dense_topk_neighbors_builds_expected_reference() -> None:
    module = load_benchmark_module()
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    neighbors = module.dense_topk_neighbors(embeddings, k=1)

    assert neighbors == [[1], [0], [1]]


def test_neighbor_purity_and_edge_overlap_are_deterministic() -> None:
    module = load_benchmark_module()
    doc_ids = ["d1", "d2", "d3"]
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}
    dense_neighbors = [[1], [0], [1]]
    candidate_neighbors = [[1], [0], [0]]

    purity = module.neighbor_purity_at_k(
        doc_ids=doc_ids,
        qrels=qrels,
        row_neighbors=candidate_neighbors,
    )
    overlap = module.edge_overlap_with_dense_at_k(
        dense_neighbors=dense_neighbors,
        row_neighbors=candidate_neighbors,
    )

    assert purity == pytest.approx(2.0 / 3.0)
    assert overlap == pytest.approx(2.0 / 3.0)


def test_resolve_device_name_supports_cpu_mps_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_benchmark_module()

    assert module.resolve_device_name("cpu") == "cpu"
    assert module.resolve_device_name("mps") == "mps"

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda device: f"fake:{device}")

    assert module.resolve_device_name("cuda:0") == "fake:cuda:0"


def test_rerank_none_uses_soft_local_path_only(tmp_path: Path) -> None:
    module = load_benchmark_module()
    calls: list[str] = []

    def dense_eval(*args, **kwargs):
        del args, kwargs
        calls.append("dense")
        return 0.7, 0.8

    def soft_local_eval(*args, **kwargs):
        del args, kwargs
        calls.append("soft_local")
        return 0.72, 0.79

    def soft_ppr_eval(*args, **kwargs):
        del args, kwargs
        calls.append("soft_ppr")
        return 0.68, 0.77

    runner = module.BeirExperimentRunner(
        base_output_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        model_name="fake-model",
        model_factory=fake_model_factory,
        dense_eval_fn=dense_eval,
        soft_local_eval_fn=soft_local_eval,
        soft_ppr_eval_fn=soft_ppr_eval,
        hard_graph_fn=fake_hard_graph_fn,
        soft_graph_fn=fake_soft_graph_fn,
        graph_metric_fn=fake_graph_metric_fn,
    )
    runner.logger = SilentLogger()
    runner.args = module.build_argument_parser().parse_args(["--rerank", "none"])
    encode_artifacts = module.EncodeArtifacts(
        doc_emb=torch.ones((1, 3)),
        query_emb=torch.ones((1, 3)),
        encode_ms=0.0,
        peak_rss_mb=1.0,
        peak_vram_mb=float("nan"),
        device_name="cpu",
        batch_size=runner.args.batch_size,
    )
    dense_neighbors = module.dense_topk_neighbors(encode_artifacts.doc_emb, runner.args.k)
    bundle = module.DatasetBundle(
        "fiqa",
        Path("."),
        "dev",
        {"d1": {"title": "alpha", "text": "red planet"}},
        {"q1": "planet"},
        {"q1": {"d1": 1}},
        ["d1"],
        ["q1"],
    )

    baseline = runner._run_dense_baseline(
        encode_artifacts.doc_emb,
        encode_artifacts.query_emb,
        bundle,
        encode_artifacts=encode_artifacts,
        dense_neighbors=dense_neighbors,
    )
    candidate = runner._run_soft_graph(
        encode_artifacts.doc_emb,
        encode_artifacts.query_emb,
        bundle,
        encode_artifacts=encode_artifacts,
        dense_neighbors=dense_neighbors,
    )

    assert calls == ["dense", "soft_local"]
    assert baseline["method"] == "Dense cosine baseline"
    assert candidate["method"] == "Soft graph local"
    assert candidate["path_kind"] == "soft_local"


def test_rerank_ppr_uses_ppr_path_only(tmp_path: Path) -> None:
    module = load_benchmark_module()
    calls: list[tuple[str, dict]] = []

    def dense_eval(*args, **kwargs):
        del args, kwargs
        calls.append(("dense", {}))
        return 0.7, 0.8

    def soft_local_eval(*args, **kwargs):
        del args, kwargs
        calls.append(("soft_local", {}))
        return 0.72, 0.79

    def soft_ppr_eval(*args, **kwargs):
        del args
        calls.append(("soft_ppr", kwargs))
        return 0.68, 0.77

    runner = module.BeirExperimentRunner(
        base_output_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        model_name="fake-model",
        model_factory=fake_model_factory,
        dense_eval_fn=dense_eval,
        soft_local_eval_fn=soft_local_eval,
        soft_ppr_eval_fn=soft_ppr_eval,
        hard_graph_fn=fake_hard_graph_fn,
        soft_graph_fn=fake_soft_graph_fn,
        graph_metric_fn=fake_graph_metric_fn,
    )
    runner.logger = SilentLogger()
    runner.args = module.build_argument_parser().parse_args(
        ["--rerank", "ppr", "--ppr-topk", "200"]
    )
    encode_artifacts = module.EncodeArtifacts(
        doc_emb=torch.ones((1, 3)),
        query_emb=torch.ones((1, 3)),
        encode_ms=0.0,
        peak_rss_mb=1.0,
        peak_vram_mb=float("nan"),
        device_name="cpu",
        batch_size=runner.args.batch_size,
    )
    dense_neighbors = module.dense_topk_neighbors(encode_artifacts.doc_emb, runner.args.k)
    bundle = module.DatasetBundle(
        "fiqa",
        Path("."),
        "dev",
        {"d1": {"title": "alpha", "text": "red planet"}},
        {"q1": "planet"},
        {"q1": {"d1": 1}},
        ["d1"],
        ["q1"],
    )

    candidate = runner._run_soft_graph(
        encode_artifacts.doc_emb,
        encode_artifacts.query_emb,
        bundle,
        encode_artifacts=encode_artifacts,
        dense_neighbors=dense_neighbors,
    )

    assert calls == [("soft_ppr", {"topk": 200, "alpha": 0.2})]
    assert candidate["method"] == "Soft graph + PPR"
    assert candidate["path_kind"] == "soft_ppr"


def test_summary_rejects_previous_schema_runs(tmp_path: Path) -> None:
    module = load_benchmark_module()
    experiment_root = tmp_path / "beir_euclidean_vs_geo"
    write_v2_run_artifacts(
        experiment_root,
        "2025-07-20_23-40-44",
        dataset="fiqa",
        max_docs=1000,
        rerank="none",
        ppr_topk=100,
    )
    write_v1_run_artifacts(experiment_root, "2025-07-20_23-40-45")

    with pytest.raises(ValueError, match="Unsupported or invalid benchmark runs"):
        module.write_summary_artifacts(experiment_root)

    assert not (experiment_root / "summary.csv").exists()
    assert not (experiment_root / "SUMMARY.md").exists()


def test_render_decision_gate_for_soft_local_and_ppr() -> None:
    module = load_benchmark_module()
    summary_df = pd.DataFrame(
        [
            {
                "dataset": "fiqa",
                "max_docs": 1000,
                "rerank": "none",
                "candidate_path_kind": "soft_local",
                "delta_ndcg_at_10": 0.02,
                "delta_recall_at_10": 0.0,
            },
            {
                "dataset": "fiqa",
                "max_docs": 5000,
                "rerank": "none",
                "candidate_path_kind": "soft_local",
                "delta_ndcg_at_10": -0.05,
                "delta_recall_at_10": -0.04,
            },
            {
                "dataset": "fiqa",
                "max_docs": 1000,
                "rerank": "ppr",
                "candidate_path_kind": "soft_ppr",
                "delta_ndcg_at_10": -0.01,
                "delta_recall_at_10": 0.0,
            },
            {
                "dataset": "fiqa",
                "max_docs": 5000,
                "rerank": "ppr",
                "candidate_path_kind": "soft_ppr",
                "delta_ndcg_at_10": -0.04,
                "delta_recall_at_10": -0.05,
            },
        ]
    )

    rendered = module.render_decision_gate(summary_df)

    assert "keep soft-kNN alive" in rendered
    assert "trigger geometric redesign" in rendered
    assert "freeze PPR" in rendered
