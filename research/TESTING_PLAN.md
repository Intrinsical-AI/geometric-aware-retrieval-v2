# BEIR Benchmark Testing Plan

Plan research-only para el harness `research/beir_euclidean_vs_geo.py`.
No define APIs públicas; define contratos verificables y backlog de cobertura.

## Objetivo

- Aislar fallos del benchmark en minutos usando probes efímeros.
- Convertir cada hallazgo semántico o de reproducibilidad en un test de `pytest`.
- Mantener `SUMMARY.md` y `summary.csv` como artefactos derivados, nunca editados a mano.

## Runbook operativo

- Spec del benchmark: ver [BEIR_BENCHMARK_SPEC.md](BEIR_BENCHMARK_SPEC.md).
- Probes de terminal: ver [PROBES.md](PROBES.md).
- Suite automatizada actual:
  - [tests/test_beir_benchmark.py](../tests/test_beir_benchmark.py)
  - [tests/test_run_exps.py](../tests/test_run_exps.py)

## Capas de cobertura

1. Unit tests del harness.
   - `find_dataset_root`
   - `resolve_dataset_path`
   - parseo de `corpus.jsonl`, `queries.jsonl`, `qrels`
   - `deterministic_sample`
   - claves de caché
   - carga de summary records

2. Contract tests de artefactos.
   - cada run válido genera `config.json`, `beir_results.json`, `beir_results.csv`, `run.log`
   - `summary.csv` y `SUMMARY.md` se regeneran desde artefactos respaldados
   - runs legacy o incompletos se excluyen

3. Integration tests offline.
   - fixture BEIR mínimo local
   - encoder falso
   - grafo falso
   - evaluación falsa
   - rerun determinista con misma seed

4. Semantic tests del benchmark.
   - baseline usa ranking denso
   - `rerank=none` usa scoring local sin PPR
   - `rerank=ppr` usa PPR
   - los nombres de método reflejan el camino ejecutado

5. Matrix runner tests.
   - `run_exps.py` construye exactamente la matriz `FiQA x {1000, 5000} x {none, ppr@100, ppr@200} x cpu`
   - sin GPU
   - sin datasets extra
   - con flags de descarga coherentes

6. Golden tests ligeros del agregado.
   - estructura mínima de `SUMMARY.md`
   - columnas mínimas de `summary.csv`
   - exclusión explícita de filas no respaldadas

## Cobertura ya implementada

- Resolución robusta del dataset root, incluyendo paths anidados.
- Preferencia por dataset local sobre descarga.
- Error explícito cuando no hay dataset local y la descarga está deshabilitada.
- Subsampling determinista.
- Cache key estable.
- Run end-to-end offline con artefactos completos.
- Contrato de `rerank=none` y `rerank=ppr`.
- Exclusión de runs legacy del summary.
- Decisión gate para `soft_local` y `soft_ppr`.
- Matriz fija del runner.

## Backlog de cobertura priorizado

1. Verificar estabilidad exacta de `doc_ids` y `query_ids` tras dos reruns completos con misma seed.
2. Cubrir un caso de qrels vacíos después de subsampling y exigir fallo claro.
3. Cubrir corrupción de `config.json` o `beir_results.json` y exclusión del agregado.
4. Añadir golden test de orden estable de filas en `summary.csv`.
5. Añadir test de regresión sobre `hard_graph_ranking_status` para evitar claims implícitos de hard-graph ranking.
6. Añadir smoke test opcional que valide un rerun v2 real sobre un fixture algo más grande.

## Criterio de aceptación de la siguiente fase

- Un ingeniero puede localizar un fallo de matriz en menos de 10 minutos con `PROBES.md`.
- Cada bug semántico detectado tiene un test de regresión asociado.
- Ningún summary agrega filas fuera del esquema v2 respaldado.
- Las etiquetas de método, métricas y tiempos siguen alineadas.
