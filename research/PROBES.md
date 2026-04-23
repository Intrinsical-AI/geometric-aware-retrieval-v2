# BEIR Benchmark Probes

Runbook research-only para aislar el benchmark paso a paso desde terminal sin
añadir tooling nuevo. Todos los probes usan snippets efímeros y deben
ejecutarse desde la raíz del repo.

Contrato exhaustivo del benchmark:

- [BEIR_BENCHMARK_SPEC.md](/home/z3r0/.local/share/dev-home/Proyectos/geometric-aware-retrieval-v2/research/BEIR_BENCHMARK_SPEC.md)

## Probe 1: censo de matriz

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

root = Path("research/results/beir_euclidean_vs_geo")
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    files = {name: (run_dir / name).exists() for name in ("config.json", "beir_results.json", "beir_results.csv", "run.log")}
    print(run_dir.name, files)
PY
```

## Probe 2: replay del summary

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

summary_df, invalid = module.build_summary_df(Path("research/results/beir_euclidean_vs_geo"))
print(summary_df.to_string(index=False))
print("invalid_runs =", invalid)
print(module.render_summary_markdown(summary_df, invalid))
PY
```

## Probe 3: ensamblado de matriz

```sh
.venv/bin/python - <<'PY'
import run_exps

args = run_exps.build_argument_parser().parse_args([])
for command in run_exps.build_commands(args):
    print(command)
PY
```

## Probe 4: trazado de una corrida aislada con fixture falso

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    fixture = root / "fiqa-fixture"
    (fixture / "qrels").mkdir(parents=True)
    (fixture / "corpus.jsonl").write_text('{"_id":"d1","title":"alpha","text":"red planet"}\n', encoding="utf-8")
    (fixture / "queries.jsonl").write_text('{"_id":"q1","text":"planet"}\n', encoding="utf-8")
    (fixture / "qrels" / "dev.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

    class FakeModel:
        def to(self, _device): return self
        def encode(self, texts, **_kwargs): return torch.ones((len(texts), 3), dtype=torch.float32)

    runner = module.BeirExperimentRunner(
        base_output_dir=root / "results",
        cache_root=root / "cache",
        model_name="fake-model",
        model_factory=lambda *_args: FakeModel(),
        dense_eval_fn=lambda *args, **kwargs: (0.7, 0.8),
        soft_local_eval_fn=lambda *args, **kwargs: (0.72, 0.79),
        soft_ppr_eval_fn=lambda *args, **kwargs: (0.68, 0.77),
        hard_graph_fn=lambda emb, k: (torch.eye(len(emb)), {"degree_mean": float(k), "degree_std": 0.0, "entropy": 1.0, "effective_degree": float(k)}),
        soft_graph_fn=lambda emb, **kwargs: (torch.eye(len(emb)), torch.eye(len(emb)), {"actual_degree": 1.0, "gamma_used": 0.5}),
        graph_metric_fn=lambda _adj: {"entropy": 1.23, "effective_degree": 2.34},
    )
    runner.start(["--dataset", "fiqa", "--dataset-dir", str(fixture), "--no-download", "--max-docs", "1", "--max-queries", "1", "--k", "1", "--rerank", "none"])
    print("run_dir =", runner.run_dir)
    print((runner.run_dir / "beir_results.json").read_text(encoding="utf-8"))
PY
```

## Probe 5: baseline denso vs tiempo medido

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Logger:
    def info(self, *_args, **_kwargs): pass

calls = []
runner = module.BeirExperimentRunner(
    dense_eval_fn=lambda *args, **kwargs: calls.append(("dense_eval", kwargs)) or (0.1, 0.2),
    hard_graph_fn=lambda *args, **kwargs: calls.append(("hard_graph", kwargs)) or (torch.eye(1), {"degree_mean": 1.0, "degree_std": 0.0, "entropy": 0.0, "effective_degree": 1.0}),
)
runner.logger = Logger()
runner.args = module.build_argument_parser().parse_args([])
bundle = module.DatasetBundle("fiqa", Path("."), "dev", {"d1": {"title": "a", "text": "b"}}, {"q1": "x"}, {"q1": {"d1": 1}}, ["d1"], ["q1"])
print(runner._run_dense_baseline(torch.ones((1, 3)), torch.ones((1, 3)), bundle))
print(calls)
PY
```

## Probe 6: semántica de `rerank=none`

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Logger:
    def info(self, *_args, **_kwargs): pass

calls = []
runner = module.BeirExperimentRunner(
    soft_local_eval_fn=lambda *args, **kwargs: calls.append(("soft_local", kwargs)) or (0.3, 0.4),
    soft_ppr_eval_fn=lambda *args, **kwargs: calls.append(("soft_ppr", kwargs)) or (0.5, 0.6),
    soft_graph_fn=lambda emb, **kwargs: (torch.eye(len(emb)), torch.eye(len(emb)), {"actual_degree": 1.0, "gamma_used": 0.5}),
    graph_metric_fn=lambda _adj: {"entropy": 0.0, "effective_degree": 1.0},
)
runner.logger = Logger()
runner.args = module.build_argument_parser().parse_args(["--rerank", "none"])
bundle = module.DatasetBundle("fiqa", Path("."), "dev", {"d1": {"title": "a", "text": "b"}}, {"q1": "x"}, {"q1": {"d1": 1}}, ["d1"], ["q1"])
print(runner._run_soft_graph(torch.ones((1, 3)), torch.ones((1, 3)), bundle))
print(calls)
PY
```

## Probe 7: estabilidad de selección y caché

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

ids = ["d3", "d1", "d2", "d4"]
print(module.deterministic_sample(ids, 2, 42))
print(module.deterministic_sample(ids, 2, 42))
print(module.tensor_cache_paths(Path("/tmp/probe-cache"), "fiqa", "fake-model", ["d1", "d2"], ["q1"], 42))
print(module.tensor_cache_paths(Path("/tmp/probe-cache"), "fiqa", "fake-model", ["d1", "d2"], ["q1"], 42))
PY
```

## Probe 8: degradación al escalar

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

summary_df, _ = module.build_summary_df(Path("research/results/beir_euclidean_vs_geo"))
cols = ["dataset", "max_docs", "rerank", "candidate_path_kind", "delta_ndcg_at_10", "delta_recall_at_10", "candidate_gamma", "candidate_degree_mean", "candidate_entropy"]
print(summary_df[cols].to_string(index=False))
PY
```

## Probe 9: telemetría y estructura vs baseline denso

```sh
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path("research/beir_euclidean_vs_geo.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

summary_df, _ = module.build_summary_df(Path("research/results/beir_euclidean_vs_geo"))
cols = [
    "dataset",
    "max_docs",
    "rerank_label",
    "candidate_encode_ms",
    "candidate_graph_build_ms",
    "candidate_rerank_ms",
    "candidate_peak_rss_mb",
    "candidate_peak_vram_mb",
    "candidate_neighbor_purity_at_k",
    "candidate_edge_overlap_with_dense_at_k",
    "delta_ndcg_at_10",
    "delta_recall_at_10",
]
print(summary_df[cols].to_string(index=False))
PY
```
