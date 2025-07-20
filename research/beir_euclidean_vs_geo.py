#!/usr/bin/env python3
"""BEIR Real Dataset Experiment (Refactored)

Validación de Hard k-NN vs Soft-kNN + τ-fix en un subset *dev-small* de los
corpus reales FiQA o MS-MARCO (vía BEIR), usando la nueva arquitectura de
experimentos.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

from geoIR.core.runner import ExperimentRunner
from geoIR.eval.metrics import (
    evaluate_retrieval,
    evaluate_retrieval_ppr,
    graph_distribution_metrics,
)
from geoIR.geo.differentiable import soft_knn_graph
from geoIR.geo.graph import hard_knn_graph_faiss as build_hard_knn_graph
from geoIR.utils import encode_corpus, load_tensor, save_tensor, seed_all


class BeirExperimentRunner(ExperimentRunner):
    """Encapsula la lógica para experimentos con datasets BEIR."""

    def __init__(self):
        # El __init__ solo debe declarar, no ejecutar lógica pesada.
        super().__init__(experiment_name="beir_euclidean_vs_geo", base_output_dir="research/results")
        self.args: argparse.Namespace | None = None

    def _parse_args(self) -> argparse.Namespace:
        """Configura y parsea los argumentos de línea de comandos."""
        parser = argparse.ArgumentParser(
            description="BEIR real dataset experiment with Soft-kNN τ-fix"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="msmarco",
            help="BEIR dataset name or local path.",
        )
        parser.add_argument(
            "--max-docs",
            type=int,
            default=None,
            help="Maximum number of documents (subsample). Default: all.",
        )
        parser.add_argument(
            "--max-queries",
            type=int,
            default=None,
            help="Maximum number of queries (subsample). Default: all.",
        )
        parser.add_argument(
            "--k", type=int, default=20, help="Target k for k-NN graph"
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device for embedding model"
        )
        parser.add_argument(
            "--rerank",
            type=str,
            default="none",
            choices=["none", "ppr"],
            help="Graph-based reranking method",
        )
        parser.add_argument(
            "--ppr-topk",
            type=int,
            default=100,
            help="Top-K docs to rerank with PPR",
        )
        parser.add_argument(
            "--ppr-alpha",
            type=float,
            default=0.2,
            help="Teleport (restart) probability for PPR",
        )
        parser.add_argument("--seed", type=int, default=42)
        return parser.parse_args()

    def _load_and_subsample_data(self) -> tuple | None:
        """Descarga (si es necesario), carga y submuestrea el dataset."""
        # datasets_dir = Path("datasets")
        # datasets_dir.mkdir(exist_ok=True)

        dataset_path = Path("/home/jd/Documentos/IntrinsicalAI/Repositories/geometric-aware-retrieval-v2/datasets/fiqa")
        # if local_path.exists():
        #     dataset_path = local_path
        #     self.logger.info(f"📦 Using local dataset from {dataset_path}")
        # else:
        #     self.logger.info(f"⬇️  Downloading dataset '{self.args.dataset}'...")
        #     try:
        #         dataset_path = beir_util.download_and_unzip(
        #             f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.args.dataset}.zip",
        #             str(datasets_dir),
        #         )
        #         self.logger.info(f"📦 Dataset downloaded/extracted to {dataset_path}")
        #     except Exception as e:
        #         self.logger.error(f"Failed to download dataset '{self.args.dataset}'. Error: {e}")
        #         return None

        corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="dev")
        self.logger.info(
            f"Original sizes – Docs: {len(corpus):,}, Queries: {len(queries):,}"
        )

        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())

        if self.args.max_docs and len(doc_ids) > self.args.max_docs:
            random.shuffle(doc_ids)
            doc_ids = doc_ids[: self.args.max_docs]
            corpus = {did: corpus[did] for did in doc_ids}
        
        if self.args.max_queries and len(query_ids) > self.args.max_queries:
            random.shuffle(query_ids)
            query_ids = query_ids[: self.args.max_queries]
            queries = {qid: queries[qid] for qid in query_ids}
        
        qrels = {
            qid: {doc_id: score for doc_id, score in qrel.items() if doc_id in corpus}
            for qid, qrel in qrels.items()
            if qid in queries
        }
        qrels = {qid: qrel for qid, qrel in qrels.items() if qrel}

        self.logger.info(
            f"Subset sizes – Docs: {len(corpus):,}, Queries: {len(queries):,}"
        )
        return corpus, queries, qrels, doc_ids

    def _get_embeddings(
        self, model, corpus, queries, doc_ids, qrels
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Codifica o carga desde caché los embeddings."""
        dataset_name = Path(self.args.dataset).name
        emb_dir = Path("embeddings") / dataset_name
        emb_dir.mkdir(parents=True, exist_ok=True)
        doc_emb_path = emb_dir / f"docs_{len(corpus)}.pt"
        query_emb_path = emb_dir / f"queries_{len(qrels)}.pt"

        if doc_emb_path.exists():
            self.logger.info("🔁 Loading cached document embeddings…")
            doc_emb = load_tensor(doc_emb_path)
        else:
            doc_texts = [
                corpus[did].get("title", "") + " " + corpus[did].get("text", "") for did in doc_ids
            ]
            self.logger.info(f"🔄 Encoding {len(doc_texts)} documents…")
            doc_emb = encode_corpus(model, doc_texts, device=self.args.device)
            save_tensor(doc_emb_path, doc_emb)

        if query_emb_path.exists():
            self.logger.info("🔁 Loading cached query embeddings…")
            query_emb = load_tensor(query_emb_path)
        else:
            query_texts = [queries[qid] for qid in qrels.keys()]
            self.logger.info(f"🔄 Encoding {len(query_texts)} queries…")
            query_emb = encode_corpus(model, query_texts, device=self.args.device)
            save_tensor(query_emb_path, query_emb)

        return doc_emb.cpu(), query_emb.cpu()

    def run(self):
        """Orquesta la ejecución completa del experimento."""
        data = self._load_and_subsample_data()
        if data is None:
            return
        corpus, queries, qrels, doc_ids = data
        
        if not qrels:
            self.logger.error("No relevant query-document pairs found. Aborting.")
            return

        model = SentenceTransformer("all-MiniLM-L6-v2", device=self.args.device)
        doc_emb, query_emb = self._get_embeddings(model, corpus, queries, doc_ids, qrels)

        results: List[Dict[str, Any]] = []

        self.logger.info("🔨 Building hard k-NN graph…")
        start = time.time()
        A_hard, hard_diag = build_hard_knn_graph(doc_emb.clone(), self.args.k)
        hard_time = time.time() - start
        hard_ndcg, hard_recall = evaluate_retrieval(
            query_emb, doc_emb, qrels, doc_ids
        )
        results.append({"method": "Hard k-NN", "time_s": hard_time, "ndcg@10": hard_ndcg, "recall@10": hard_recall, **hard_diag, "gamma": np.nan})

        self.logger.info("✨ Building soft k-NN τ-fix graph…")
        start = time.time()
        _, A_soft, diag_soft = soft_knn_graph(doc_emb, k=self.args.k, return_adjacency=True, return_diagnostics=True)
        soft_time = time.time() - start

        if self.args.rerank == "ppr":
            soft_ndcg, soft_recall = evaluate_retrieval_ppr(
                query_emb, doc_emb, qrels, doc_ids, A_soft, topk=self.args.ppr_topk
            )
        else:
            soft_ndcg, soft_recall = evaluate_retrieval_ppr(
                query_emb, doc_emb, qrels, doc_ids, A_soft, topk=self.args.k
            )


        dist_metrics = graph_distribution_metrics(A_soft)
        results.append({"method": "Soft k-NN τ-fix", "time_s": soft_time, "ndcg@10": soft_ndcg, "recall@10": soft_recall, "degree_mean": diag_soft["actual_degree"], "degree_std": float(A_soft.sum(dim=-1).std()), "entropy": dist_metrics["entropy"], "effective_degree": dist_metrics["effective_degree"], "gamma": diag_soft["gamma_used"]})

        self._report_results(results)

    def _report_results(self, results: List[Dict[str, Any]]):
        """Genera y guarda el reporte de resultados."""
        df = pd.DataFrame(results)
        df["time_ms"] = df.pop("time_s") * 1000
        df = df[["method", "time_ms", "ndcg@10", "recall@10", "degree_mean", "degree_std", "entropy", "effective_degree", "gamma"]]

        self.logger.info("\n📊 RESULTS:")
        self.logger.info(df.to_string(index=False, float_format="{:.4f}".format))

        self.save_results(df.to_dict(orient="records"), "beir_results.json")
        self.save_dataframe(df, "beir_results.csv")
        self.logger.info(f"✅ Results saved to {self.run_dir}")

    def start(self):
        """Punto de entrada para iniciar el experimento."""
        self.args = self._parse_args()
        seed_all(self.args.seed)
        
        # Llama al método de la clase base para configurar el entorno
        super().start(config=vars(self.args))


if __name__ == "__main__":
    runner = BeirExperimentRunner()
    runner.start()