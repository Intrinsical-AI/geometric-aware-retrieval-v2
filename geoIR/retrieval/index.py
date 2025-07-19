"""Index object wrapping embeddings, k-NN graph and geo audit helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import json
import hashlib
from datetime import datetime
import pickle
from pathlib import Path
import numpy as np

from geoIR.geo.curvature import ricci_ollivier


@dataclass
class AuditResult:  # noqa: D101
    curvature: dict

    def plot(self, layout: str = "spring", node_size: int = 50, **kwargs):
        """Plot the k-NN graph structure with optional geometric annotations.
        
        Parameters
        ----------
        layout : str, default "spring"
            Graph layout algorithm: "spring", "circular", "kamada_kawai".
        node_size : int, default 50
            Size of nodes in the plot.
        **kwargs
            Additional arguments passed to matplotlib/networkx plotting.
            
        Returns
        -------
        matplotlib.figure.Figure or None
            Figure object if matplotlib available, None otherwise.
            
        Notes
        -----
        Requires matplotlib and optionally plotly for interactive plots.
        Install with: pip install matplotlib plotly
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            import warnings
            warnings.warn(
                "Plotting requires matplotlib and networkx. "
                "Install with: pip install matplotlib networkx",
                UserWarning
            )
            return None
            
        if not hasattr(self, 'graph') or self.graph is None:
            import warnings
            warnings.warn("No graph available. Build index first with .build()")
            return None
            
        # Create layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)  # fallback
            
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(self.graph, pos, ax=ax, node_size=node_size, 
                with_labels=False, node_color='lightblue', 
                edge_color='gray', alpha=0.7, **kwargs)
        
        ax.set_title(f"k-NN Graph ({self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges)")
        plt.tight_layout()
        return fig


class _GeoNamespace:
    """Lightweight attribute namespace so users can call `index.geo.audit(...)`."""

    def __init__(self, index: "Index") -> None:  # noqa: D401
        self._index = index

    # expose audit helper
    def audit(self, *args, **kwargs):  # noqa: D401
        return self._index.geo_audit(*args, **kwargs)


class Index:  # noqa: D101
    """Embeddings + graph retrieval index.

    Supports persistence via :py:meth:`save` / :py:meth:`load`.
    """
    def __init__(self, embeddings: np.ndarray, corpus: List[str], graph):  # type: ignore[valid-type]
        """Instantiate an Index."""
        self.embeddings = embeddings
        self.corpus = corpus
        self.graph = graph

    # ------------------------------------------------------------------
    def geo_audit(self, curvature: bool = True, **_kwargs) -> AuditResult:  # noqa: D401
        curv_dict = ricci_ollivier(self.graph) if curvature else {}
        return AuditResult(curvature=curv_dict)

    # ------------------------------------------------------------------
    def search(
        self,
        query_emb: np.ndarray,
        k: int = 10,
        *,
        metric: str = "cosine",
        search_k: int = 1000,
        connect_k: int = 10,
        mix_kappa: float | None = None,
        alpha: float | None = None,
    ) -> list[int]:  # noqa: D401
        """Search using cosine or cosine+κ(x) mix.

        Parameters
        ----------
        query_emb : np.ndarray
            A pre-computed query embedding of shape `(d,)`.
        mix_kappa : float | None, optional
            Curvature weight (legacy, prefer `alpha`).
        alpha : float | None, optional
            Mixing factor [0, 1] for cosine vs curvature (0 = pure cosine, 1 = pure curvature).
            If both `alpha` and `mix_kappa` are provided, `alpha` takes precedence.
        """

        metric = metric.lower()

        # ------------------------------------------------------------------
        # 2. Fast path – cosine / curvature-mix (legacy)
        # ------------------------------------------------------------------
        if metric in {"cosine", "curvature", "mix"}:
            sims = (self.embeddings @ query_emb).flatten()

            # Support both alpha (preferred) and mix_kappa (legacy)
            if alpha is not None:
                if not 0 <= alpha <= 1:
                    raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
                mix_kappa_val = alpha / (1 - alpha) if alpha < 1.0 else float("inf")
            else:
                mix_kappa_val = mix_kappa

            if mix_kappa_val is not None:
                # compute average curvature per node lazily once
                if not hasattr(self, "_avg_curv"):
                    from collections import defaultdict

                    avg = defaultdict(list)
                    curv = ricci_ollivier(self.graph)
                    for (u, v), kappa in curv.items():
                        avg[u].append(kappa)
                        avg[v].append(kappa)
                    self._avg_curv = {n: float(np.mean(vals)) for n, vals in avg.items()}

                curv_vals = np.array([self._avg_curv.get(i, 0.0) for i in range(len(sims))])
                if alpha is not None and alpha < 1.0:
                    sims = (1 - alpha) * sims + alpha * curv_vals
                else:
                    sims = sims + mix_kappa_val * curv_vals
            return np.argsort(sims)[-k:][::-1]

        # ------------------------------------------------------------------
        # 3. Geodesic rerank (hybrid)  – Strategy C
        # ------------------------------------------------------------------
        if metric == "geodesic":
            import networkx as nx

            # (a) initial candidates by cosine similarity
            sims = (self.embeddings @ query_emb).flatten()
            cand_idx = np.argsort(sims)[-search_k:][::-1]

            # (b) induced subgraph + query node
            sub = self.graph.subgraph(int(i) for i in cand_idx).copy()
            q_node = -1  # temporary id not overlapping with dataset ids (they are >=0)
            for i in cand_idx[:connect_k]:
                w = 1.0 - float(self.embeddings[i] @ query_emb)
                sub.add_edge(q_node, int(i), weight=w)

            # (c) Dijkstra distances
            dist = nx.single_source_dijkstra_path_length(sub, q_node)
            dist.pop(q_node, None)
            top = sorted(dist.items(), key=lambda t: t[1])[:k]
            return [idx for idx, _ in top]

        # ------------------------------------------------------------------
        # 4. Fallback – unsupported metric
        # ------------------------------------------------------------------
        raise ValueError(f"Unsupported metric '{metric}'.")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, file: str | Path) -> 'Index':  # noqa: D401
        """Load a serialised index from *file*.
        
        Args:
            file: Path to the saved index file
            
        Returns:
            Loaded Index instance
        """
        path = Path(file).expanduser().resolve()
        with path.open("rb") as f:
            return pickle.load(f)
            
    def save(self, file: str | Path) -> str:  # noqa: D401
        """Serialise the index to *file* using :pymod:`pickle`.

        Args:
            file: Path where to save the index
            
        Returns:
            Absolute path of the written file.
        """
        path = Path(file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    # ------------------------------------------------------------------
    @property
    def geo(self):  # noqa: D401
        """Expose geometry helpers under `index.geo` namespace."""
        return _GeoNamespace(self)

    # ------------------------------------------------------------------
    class _ReportNamespace:
        """Namespace for `index.report.save(...)`."""

        def __init__(self, index: "Index") -> None:  # noqa: D401
            self._index = index

        def save(self, name: str, dir: Path | str | None = None):  # noqa: D401
            dir_path = Path(dir or ".").resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256(self._index.embeddings.tobytes()).hexdigest()[:12]
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            file_path = dir_path / f"{name}_{sha}_{ts}.json"
            payload = {
                "name": name,
                "sha12": sha,
                "timestamp": ts,
                "embeddings_shape": self._index.embeddings.shape,
                "corpus_size": len(self._index.corpus),
            }
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return str(file_path)

    @property
    def report(self):  # noqa: D401
        """Expose report helpers under `index.report`."""
        return self._ReportNamespace(self)
