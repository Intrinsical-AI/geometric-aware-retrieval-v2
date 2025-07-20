#!/usr/bin/env python3
"""
Script para orquestar los experimentos preliminares de geoIR:
- BEIR (FiQA y MS MARCO) con Hard k-NN vs Soft-kNN+τ-fix
- Variaciones de tamaño de corpus, dispositivo y reranking PPR

Detecta automáticamente si hay GPU disponible y solo usa 'cuda' si Torch la reconoce.
"""
import subprocess
import os
import torch

# Configuración de los experimentos
datasets = ["fiqa"]
max_docs_list = [1000, 5000]
max_queries = 100
k = 20
# Detectar dispositivos disponibles
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
rerank_opts = ["none", "ppr"]
ppr_topks = [100, 200]

script = os.path.join("research", "beir_euclidean_vs_geo.py")

for dataset in datasets:
    for max_docs in max_docs_list:
        for rerank in rerank_opts:
            for device in devices:
                base_cmd = [
                    "python3", script,
                    "--dataset", dataset,
                    "--max-docs", str(max_docs),
                    "--max-queries", str(max_queries),
                    "--k", str(k),
                    "--device", device,
                    "--rerank", rerank
                ]
                if rerank == "ppr":
                    for topk in ppr_topks:
                        cmd = base_cmd + ["--ppr-topk", str(topk)]
                        print("Ejecutando:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                else:
                    print("Ejecutando:", " ".join(base_cmd))
                    subprocess.run(base_cmd, check=True)

print("\n✅ Todos los experimentos preliminares han finalizado.")
