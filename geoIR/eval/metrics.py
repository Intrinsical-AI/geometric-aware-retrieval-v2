"""Standard retrieval evaluation metrics."""
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pytrec_eval
import torch
from sentence_transformers import util as st_util
from geoIR.geo.graph_rerank import personalized_pagerank


@dataclass
class MetricResult:  # noqa: D101
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}(score={self.score:.3f})"


def evaluate_retrieval(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    qrels: Dict[str, Dict[str, int]],
    doc_ids: List[str],
    k_eval: int = 10,
) -> Tuple[float, float]:
    """Dense cosine similarity ranking (baseline)."""
    # Similaridad coseno
    sims = st_util.cos_sim(query_emb, doc_emb).cpu().numpy()

    run: Dict[str, Dict[str, float]] = {}
    for i, (qid, scores_row) in enumerate(zip(qrels.keys(), sims)):
        # Rank docs desc
        top_k_idx = np.argsort(-scores_row)[: k_eval * 5]  # take more to be safe
        run[qid] = {doc_ids[idx]: float(scores_row[idx]) for idx in top_k_idx}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k_eval}", f"recall_{k_eval}"})
    results = evaluator.evaluate(run)

    ndcg_scores = [v[f"ndcg_cut_{k_eval}"] for v in results.values()]
    recall_scores = [v[f"recall_{k_eval}"] for v in results.values()]
    return float(np.mean(ndcg_scores)), float(np.mean(recall_scores))


def graph_distribution_metrics(A: torch.Tensor) -> Dict[str, float]:
    """Calculates entropy and effective degree for a graph's adjacency matrix."""
    A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -(A_norm * (A_norm + 1e-12).log()).sum(dim=-1).mean()
    eff_degree = 1.0 / (A_norm.pow(2).sum(dim=-1)).mean()
    return {"entropy": entropy.item(), "effective_degree": eff_degree.item()}


def evaluate_retrieval_ppr(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    qrels: Dict[str, Dict[str, int]],
    doc_ids: List[str],
    A_soft: torch.Tensor,
    k_eval: int = 10,
    topk: int = 100,
    alpha: float = 0.2,
) -> Tuple[float, float]:
    """Graph-based PPR reranking over a soft graph."""
    if pytrec_eval is None:
        raise ImportError("Please install pytrec_eval: pip install pytrec_eval")

    A_soft = A_soft.cpu()
    run: Dict[str, Dict[str, float]] = {}
    sims_all = st_util.cos_sim(query_emb, doc_emb).cpu()
    query_ids = list(qrels.keys())

    for i, qid in enumerate(query_ids):
        sims = sims_all[i]
        _, idx = sims.topk(topk)
        sub_adj = A_soft[idx][:, idx]
        row_sum = sub_adj.sum(dim=-1, keepdim=True) + 1e-12
        sub_adj = sub_adj / row_sum
        prior = sims[idx].clamp(min=1e-6)
        prior = prior / prior.sum()
        ppr_scores = personalized_pagerank(sub_adj, prior, alpha=alpha, iters=20)
        cut = k_eval * 5
        top_idx = torch.argsort(ppr_scores, descending=True)[:cut]
        run_docs = [(doc_ids[idx[top_idx[j]].item()], float(ppr_scores[top_idx[j]])) for j in range(len(top_idx))]
        run[qid] = {d: s for d, s in sorted(run_docs, key=lambda x: -x[1])}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k_eval}", f"recall_{k_eval}"})
    results = evaluator.evaluate(run)
    ndcg_scores = [v[f"ndcg_cut_{k_eval}"] for v in results.values()]
    recall_scores = [v[f"recall_{k_eval}"] for v in results.values()]
    return float(np.mean(ndcg_scores)), float(np.mean(recall_scores))
