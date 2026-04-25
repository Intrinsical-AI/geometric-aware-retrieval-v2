# BEIR Benchmark Results

Este directorio contiene los artefactos de runs del benchmark research-only
`beir_euclidean_vs_geo`.

La fuente de verdad agregada es:

- `SUMMARY.md`: resumen generado automáticamente a partir de `config.json` + `beir_results.json`.
- `summary.csv`: tabla plana generada automáticamente con los mismos runs válidos.
- `../../BEIR_BENCHMARK_SPEC.md`: contrato exhaustivo del benchmark, métricas, telemetría e hipótesis.

Regla operativa:

- No mantener tablas manuales en este `README.md`.
- No sacar conclusiones desde notas históricas si no están respaldadas por un par `config.json` + `beir_results.json`.
- El agregado acepta solo el esquema de benchmark actual; los runs legacy quedan fuera hasta rerun real.
- Los claims manuales previos sobre `msmarco-passage` quedan no confiables hasta que exista un rerun respaldado por artefactos.

Para regenerar el resumen agregado sin editar archivos a mano, vuelve a ejecutar
una corrida con `research/beir_euclidean_vs_geo.py` o la matriz fija con
`run_exps.py`; cada run refresca `SUMMARY.md` y `summary.csv`.

Para aislamiento rápido y probes efímeros desde terminal, ver
`research/PROBES.md`.
