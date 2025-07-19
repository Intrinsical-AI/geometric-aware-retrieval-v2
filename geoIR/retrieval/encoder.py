"""geoIR – Default Hugging Face encoder backend

This module implements a minimal yet *production‑ready* wrapper around
Transformers models so they can be used as dual‑ or mono‑encoders inside
the geoIR pipeline.  Key design goals:
  • **Zero‑friction** for newcomers (single dependency: `transformers`).
  • **Extensible**: hooks for sentence‑transformers fine‑tuning or custom pooling.
  • **Efficient** on GPU/CPU through batched inference and optional caching.
  • **Transparent**: every heavy operation (dataset load, encoding, graph build)
    can emit progress bars when `verbose=True`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, Union

import warnings

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from geoIR.core.registry import registry
from geoIR.geo.graph import build_knn_graph
from geoIR.retrieval.index import Index
# ------------------------------------------------------------------
# Optional dataset loader (may be absent in minimal install).
# ------------------------------------------------------------------
try:
    from geoIR.data import load as _load_ds  # type: ignore
except Exception:  # pragma: no cover – fallback lightweight loader
    from geoIR.data.fallback import load_text_file
    
    def _load_ds(path: str, *_, max_docs: int | None = None, **__) -> "MinimalDataset":
        """Fallback loader using the dedicated fallback module."""
        return load_text_file(path, max_docs=max_docs)

# -------------------------------------------------------------
# Type Aliases
# -------------------------------------------------------------
_ModeT = Literal["dual", "mono"]
_MetricT = Literal["cosine", "euclidean"]
_DeviceT = Literal["cpu", "cuda", "mps"]


# -------------------------------------------------------------
# Dataset utilities
# -------------------------------------------------------------

def _infer_text_column(column_names: Sequence[str]) -> str:
    """Heuristic to pick a reasonable text column from a HF dataset."""
    PREFERRED = ("text", "content", "document", "passage")
    for cand in PREFERRED:
        if cand in column_names:
            return cand
    # Fallback: first string‑type column or first column overall
    return column_names[0]



# -------------------------------------------------------------
# Encoder
# -------------------------------------------------------------

class Encoder:  # noqa: D101
    def __init__(
        self,
        model_name: str,
        *,
        mode: _ModeT = "dual",
        device: _DeviceT | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.device: _DeviceT = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        # In dual mode we keep separate query/document encoders *unless* the
        # same name is provided, in which case weights are shared.
        self.q_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_tokenizer = self.q_tokenizer  # shared by default

        if mode == "dual":
            self.q_model = AutoModel.from_pretrained(model_name).to(self.device)
            if model_name.endswith("-doc"):
                # Heuristic: user provided two model names separated by comma
                q_name, d_name = model_name.split(",", maxsplit=1)
                self.d_tokenizer = AutoTokenizer.from_pretrained(d_name)
                self.d_model = AutoModel.from_pretrained(d_name).to(self.device)
            else:
                self.d_model = self.q_model  # weight‑tied
        else:  # mono encoder = same tower
            self.q_model = self.d_model = AutoModel.from_pretrained(model_name).to(self.device)

        # Small warm‑up to catch device / dtype errors early
        with torch.no_grad():
            dummy = torch.tensor([[self.q_tokenizer.cls_token_id]], device=self.device)
            _ = self.q_model(dummy)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_batch(self, texts: Sequence[str], *, is_query: bool) -> torch.Tensor:
        tokenizer = self.q_tokenizer if is_query else self.d_tokenizer
        model = self.q_model if is_query else self.d_model
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :]  # CLS pooling
        return torch.nn.functional.normalize(vec, dim=1) if self.normalize else vec

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute embeddings with graceful fallback if GPU memory is tight.

        The method will attempt to process *batch_size* texts at once.  If a
        CUDA out-of-memory error occurs, the batch size is **halved**
        automatically until the operation succeeds or the batch size reaches 1.
        This prevents the host from freezing while still making good use of the
        available hardware.
        """
        parts: List[np.ndarray] = []
        n = len(texts)
        idx = 0
        cur_bs = batch_size
        while idx < n:
            chunk = texts[idx : idx + cur_bs]
            try:
                emb = self._encode_batch(chunk, is_query=is_query).cpu().numpy()
                parts.append(emb)
                idx += cur_bs  # move window only on success
            except RuntimeError as exc:
                # Handle GPU (CUDA Op Out of Memory) by reducing batch size progressively
                msg = str(exc).lower()
                if "out of memory" in msg and self.device == "cuda" and cur_bs > 1:
                    torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
                    if verbose:
                        warnings.warn(
                            f"[encoder] CUDA OOM – retrying with batch_size={cur_bs}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    continue  # retry same chunk with smaller batch
                raise  # propagate other errors
        return np.vstack(parts, dtype=np.float32)

    def build_index(
        self,
        corpus: Sequence[str],
        *,
        k: int = 10,
        metric: _MetricT = "cosine",
        batch_size: int = 128,
        verbose: bool = False,
    ) -> Index:
        """Encode *corpus* and build a geometric-aware k-NN index.

        This is a convenience wrapper used by the examples so users can do::

            import geoIR as gi
            idx = gi.load_encoder("bge-base").build_index(corpus=my_docs, k=20)

        Parameters
        ----------
        corpus : Sequence[str]
            List of raw documents.
        k : int, default 10
            `k` for the k-NN graph.
        metric : {"cosine", "euclidean"}, default "cosine"
            Distance metric for graph construction.
        batch_size : int, default 128
            Encoding batch size.
        verbose : bool, default False
            Print progress information.

        Returns
        -------
        Index
            An :class:`geoIR.retrieval.index.Index` instance ready for search or geo audit.
        """
        if verbose:
            print(f"[encoder] Encoding {len(corpus)} documents…")
        embeddings = self.encode(corpus, batch_size=batch_size, is_query=False, verbose=verbose)

        if verbose:
            print(f"[encoder] Building {metric} k-NN graph (k={k})…")
        graph = build_knn_graph(embeddings, k=k, metric=metric, verbose=verbose)

        return Index(embeddings=embeddings, corpus=list(corpus), graph=graph)

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return f"Encoder(name={self.model_name}, mode={self.mode}, device={self.device})"


# -----------------------------------------------------------------
# Registry hook → enables `gi.load_encoder("hf", ...)`
# -----------------------------------------------------------------

@registry.register("encoder")("default")
def _load_encoder(name: str, *, mode: _ModeT = "dual", **kw: Any) -> Encoder:  # noqa: D401
    """Factory wrapper for geoIR internal registry."""
    return Encoder(name, mode=mode, **kw)
