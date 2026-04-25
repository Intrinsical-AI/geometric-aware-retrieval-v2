# BEIR Benchmark Spec

Especificación exhaustiva del harness research-only
`research/beir_euclidean_vs_geo.py`.

Este documento define el contrato operativo del benchmark, las métricas que
persisten los artefactos v2, la semántica exacta de latencia y memoria, y las
hipótesis de trabajo vigentes para interpretar resultados en FiQA.

## Propósito y límites

- El benchmark existe para comparar un baseline denso por coseno contra un
  único camino geométrico candidato por run.
- El contrato es research-only. No redefine la API pública del paquete
  `geoIR`.
- Los artefactos respaldados son `config.json`, `beir_results.json`,
  `beir_results.csv`, `SUMMARY.md` y `summary.csv`.
- `SUMMARY.md` y `summary.csv` son derivados. La fuente de verdad primaria por
  run es el par `config.json` + `beir_results.json`.
- Las notas históricas o manuales fuera de esos artefactos no son evidencia
  respaldada.

## Flujos evaluados

- Baseline: `Dense cosine baseline`
  - Ranking por similitud coseno entre query embeddings y doc embeddings.
- Candidate con `rerank=none`: `Soft graph local`
  - Construye el soft graph.
  - Usa el prior de coseno sobre top-k candidatos y lo mezcla con soporte 1-hop
    del subgrafo local.
- Candidate con `rerank=ppr`: `Soft graph + PPR`
  - Construye el soft graph.
  - Usa difusión `Personalized PageRank` sobre el subgrafo de candidatos.

## Esquema v2

Cada fila de `beir_results.json` y `beir_results.csv` debe contener estos
campos:

| Campo | Tipo | Unidad | Aplica a | Semántica |
| --- | --- | --- | --- | --- |
| `method` | `str` | n/a | baseline/candidate | Nombre legible del camino ejecutado |
| `role` | `str` | n/a | baseline/candidate | `baseline` o `candidate` |
| `path_kind` | `str` | n/a | baseline/candidate | `dense_cosine`, `soft_local` o `soft_ppr` |
| `status` | `str` | n/a | baseline/candidate | Estado normalizado del run |
| `encode_ms` | `float` | ms | baseline/candidate | Tiempo total de codificar documentos y queries |
| `graph_build_ms` | `float` | ms | baseline/candidate | Tiempo de construir el soft graph; `0.0` en baseline |
| `rerank_ms` | `float` | ms | baseline/candidate | Tiempo del ranking/evaluación del camino ejecutado |
| `build_time_ms` | `float` | ms | baseline/candidate | Alias de compatibilidad de `graph_build_ms` |
| `eval_time_ms` | `float` | ms | baseline/candidate | Alias de compatibilidad de `rerank_ms` |
| `time_ms` | `float` | ms | baseline/candidate | `encode_ms + graph_build_ms + rerank_ms` |
| `peak_rss_mb` | `float` o `NaN` | MiB | baseline/candidate | Pico RSS del proceso durante la ventana del row |
| `peak_vram_mb` | `float` o `NaN` | MiB | baseline/candidate | Pico VRAM solo en `cuda`; `NaN` en `cpu` o `mps` |
| `device_name` | `str` | n/a | baseline/candidate | `cpu`, `mps` o nombre de GPU CUDA |
| `batch_size` | `int` | n/a | baseline/candidate | Batch size usado en encoding |
| `ndcg@10` | `float` | ratio | baseline/candidate | `nDCG` medio a cutoff 10 |
| `recall@10` | `float` | ratio | baseline/candidate | `Recall` medio a cutoff 10 |
| `degree_mean` | `float` o `NaN` | grado esperado | baseline/candidate | Diagnóstico del grafo |
| `degree_std` | `float` o `NaN` | grado esperado | baseline/candidate | Diagnóstico del grafo |
| `entropy` | `float` o `NaN` | nats | baseline/candidate | Entropía media de distribución de vecinos |
| `effective_degree` | `float` o `NaN` | grado efectivo | baseline/candidate | Diagnóstico del grafo |
| `gamma` | `float` o `NaN` | n/a | baseline/candidate | Temperatura usada en `soft_knn_graph` |
| `neighbor_purity@k` | `float` o `NaN` | ratio | baseline/candidate | Pureza estructural inducida por qrels |
| `edge_overlap_with_dense@k` | `float` o `NaN` | ratio | baseline/candidate | Solapamiento contra vecindad densa por coseno |

`config.json` debe persistir además `benchmark_schema_version`, `batch_size`,
dataset, seed, `k`, flags de rerank, device y metadata del run.

## Ventanas de medición

- `encode_ms`
  - Suma del tiempo de codificar documentos y queries.
  - Se repite en baseline y candidate para que ambas filas describan el coste
    end-to-end del mismo run.
  - Si se reutiliza caché de embeddings y no se codifica nada, el valor puede
    ser `0.0`.
- `graph_build_ms`
  - Solo mide la construcción del soft graph.
  - No incluye cálculos post-hoc de diagnósticos estructurales.
- `rerank_ms`
  - Mide solo el camino de ranking/evaluación:
    - baseline: evaluación densa por coseno
    - candidate: scoring `soft_local` o `PPR`
- `peak_rss_mb`
  - Pico RSS del proceso muestreado durante la ventana del row.
  - El row baseline toma el máximo entre encode y rerank.
  - El row candidate toma el máximo entre encode, graph build y rerank.
- `peak_vram_mb`
  - Pico de `torch.cuda.max_memory_allocated` durante la misma ventana.
  - Solo se considera soportado en `cuda`.

## Definiciones formales

### `nDCG@10`

- Se usa `pytrec_eval` sobre el run TREC construido por el camino ejecutado.
- El score reportado es la media de `ndcg_cut_10` sobre todas las queries del
  subset.

### `Recall@10`

- Se usa `pytrec_eval` sobre el mismo run.
- El score reportado es la media de `recall_10` sobre todas las queries del
  subset.

### `edge_overlap_with_dense@k`

Referencia:

- Para cada documento `i`, se calcula `N_dense(i, k)` como sus top-k vecinos
  por coseno entre documentos, excluyendo self.

Vecindad por fila:

- Baseline: `N_row(i, k) = N_dense(i, k)`.
- Candidate: `N_row(i, k)` son los top-k vecinos por peso de la adyacencia del
  soft graph.

Score por documento:

`overlap_i = |N_row(i, k) ∩ N_dense(i, k)| / min(k, n - 1)`

Score global:

`edge_overlap_with_dense@k = mean_i(overlap_i)`

Interpretación:

- Baseline esperado: `1.0`.
- Candidate: valor en `[0, 1]`.
- Mide cuánto se aparta la estructura geométrica de la vecindad densa local.

### `neighbor_purity@k`

Etiquetas estructurales:

- A cada documento `d` se le asocia el conjunto de queries `Q(d)` para las que
  es relevante con `score > 0` en `qrels`.
- Los documentos sin `Q(d)` quedan fuera del promedio.

Score por documento etiquetado:

- Se toma la vecindad `N_row(i, k)` de la fila evaluada.
- Se cuenta cuántos vecinos `j` cumplen `Q(d_i) ∩ Q(d_j) ≠ ∅`.

`purity_i = hits_i / |N_row(i, k)|`

Score global:

`neighbor_purity@k = mean_i(purity_i)` sobre documentos etiquetados con al
menos un vecino.

Interpretación:

- No es `precision@k`.
- No mide ranking query-condicionado.
- Mide si la vecindad estructural preserva afinidad inducida por qrels.

## Qué explica cada métrica

- `nDCG@10`
  - Explica calidad del orden en la cabeza del ranking.
  - Puede mejorar sin cambiar cobertura.
- `Recall@10`
  - Explica cobertura de relevantes en el top-10.
  - No distingue bien entre permutaciones internas si la cobertura es igual.
- `edge_overlap_with_dense@k`
  - Explica cuánto se deforma la vecindad local frente al baseline denso.
  - No dice por sí sola si la deformación mejora o empeora IR.
- `neighbor_purity@k`
  - Explica si el grafo preserva afinidad estructural inducida por qrels.
  - No reemplaza métricas IR ni justifica causalmente mejoras en ranking.
- `peak_rss_mb` y `peak_vram_mb`
  - Explican coste de memoria por row.
  - No sustituyen profiling detallado por kernel o por operador.

## Hipótesis de trabajo vigentes para FiQA

- `Soft graph local` puede mejorar orden local en subsets pequeños, en especial
  cuando el prior denso ya recupera los candidatos correctos y el grafo solo
  reordena dentro de ese shortlist.
- `PPR` tiende a sobredifundir masa al crecer el subgrafo de candidatos y puede
  perder precisión local aun manteniendo cobertura.
- La degradación al pasar de 1k a 5k docs puede venir de:
  - puentes espurios entre regiones semánticas,
  - suavización excesiva del soft graph,
  - aumento de vecinos plausibles pero no relevantes en difusión.

Estas hipótesis orientan análisis posteriores, pero no cuentan como claims
respaldados sin un run v2 reproducible.

## Reglas de interpretación

- Solo cuentan como backed claims los resultados que aparecen en artefactos v2
  válidos.
- Los runs legacy quedan fuera del agregado hasta rerun real.
- `SUMMARY.md` y `summary.csv` deben poder regenerarse sin edición manual.
- Si un claim aparece en notas históricas pero no en artefactos respaldados,
  debe tratarse como no confiable.

## Runbook corto

Ejecutar un run aislado:

```bash
uv run python research/beir_euclidean_vs_geo.py \
  --dataset fiqa \
  --dataset-dir datasets/fiqa \
  --no-download \
  --max-docs 1000 \
  --max-queries 100 \
  --k 20 \
  --batch-size 256 \
  --device cpu \
  --rerank none
```

Ejecutar la matriz fija de FiQA:

```bash
uv run python run_exps.py --dataset fiqa --no-download --batch-size 256
```

Probes rápidos:

- [PROBES.md](./PROBES.md) para replay del summary y diagnóstico de columnas.
- [TESTING_PLAN.md](./TESTING_PLAN.md) para contratos verificables y backlog de
  cobertura.
- [results/beir_euclidean_vs_geo/README.md](./results/beir_euclidean_vs_geo/README.md)
  para reglas operativas de artefactos.
