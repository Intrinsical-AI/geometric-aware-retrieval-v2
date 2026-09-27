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
- El agregado acepta solo el esquema de benchmark actual; los runs de otro esquema hacen fallar la generación. El archivo hermano `.v1_archive` conserva las observaciones anteriores sin mezclarlas con v2.
- Los claims manuales previos sobre `msmarco-passage` quedan no confiables hasta que exista un rerun respaldado por artefactos.

Para regenerar el resumen de los artefactos existentes sin ejecutar nuevos
experimentos, usa `make summary`. No descarga modelos ni datasets. Las corridas
explícitas con `research/beir_euclidean_vs_geo.py` o `run_exps.py` también
actualizan el agregado, pero no son necesarias para verificar lo ya registrado.

Los seis runs v2 conservados pierden nDCG@10 frente al baseline denso; las
decisiones negativas quedan en `SUMMARY.md`. El directorio hermano
`.failed_sigill` conserva la configuración de un intento sin resultados completos
y no entra en el agregado de runs terminados.

Para aislamiento rápido y probes efímeros desde terminal, ver
`research/PROBES.md`.
