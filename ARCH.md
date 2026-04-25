# geoIR Architecture Draft

Status: draft  
Last updated: 2026-04-23  
Basis: static codebase exploration and file-level evidence from the current repository state.

## Purpose

This document defines an initial architecture map and a conservative support boundary for the repository.

It is intentionally stricter than the README. Until the team validates the public contract, any flow not explicitly marked as stable in this document should be treated as experimental.

## Current Repository Shape

The repository currently combines three concerns:

1. A Python package published as `geoIR`, centered on retrieval, geometry, evaluation, and training code.
2. A CLI exposed via `geoIR = "geoIR.cli:app"` in `pyproject.toml`.
3. Research and experiment scripts under `research/` plus orchestration in `run_exps.py`.

Main directories:

- `geoIR/core`: config, registry, experiment runner.
- `geoIR/retrieval`: encoder, index, high-level retriever.
- `geoIR/geo`: graph construction, curvature, differentiable geometric pipeline.
- `geoIR/training`: trainer and loss integration.
- `geoIR/eval`: retrieval metrics, LLM judges, RARE/SUD prototypes.
- `geoIR/data`: lightweight file loaders and synthetic data helpers.
- `research/`: experiment scripts and result artifacts.

## Architectural Boundary

### Public supported surface

No flow is considered stable by default in this draft.

Promotion to stable should require all of the following:

- a documented contract,
- a reproducible acceptance check,
- declared dependencies in `pyproject.toml`,
- at least one regression or smoke test,
- CI coverage consistent with that contract.

### Experimental / prototype surface

Until validated, the following should be treated as experimental or unsupported:

- package import behavior and dependency bootstrapping,
- CLI subcommands,
- quick experiment helpers,
- training workflows,
- LLM-as-a-judge utilities,
- research scripts and batch orchestration.

### Proposed CLI v0 Scope

The CLI will be intentionally smaller than the package surface.

Decision text for the current draft:

- Keep `geoIR encode` as a supported utility command for local embedding inspection and debugging.
- Keep `geoIR audit` as the primary CLI command for geometric interpretability over local corpora.
- Treat `geoIR audit --plot` as experimental until `AuditResult` carries graph data and the plotting path is documented and tested.
- Remove `geoIR report-save` from the public CLI surface.
- Keep `geoIR search` only as a deprecated and unsupported compatibility stub until the next clean CLI cut.
- Remove `geoIR eval` from the public CLI surface.

This shrink is intentional. The CLI is being optimized for geometric inspection and local debugging, not for general retrieval serving, benchmark orchestration, or LLM-judge workflows.

## Initial Support Matrix

This matrix covers the flows that should be validated with the team first.

| Flow | Intended audience | Initial status | Why it is classified this way | Candidate acceptance check |
| --- | --- | --- | --- | --- |
| `import geoIR` | library consumers | Experimental | The package now lazy-registers the retrieval backend so plain `import geoIR` no longer pulls the optional HuggingFace stack at import time. The flow remains experimental until the team decides whether no-extra package import is part of the supported contract and adds an explicit smoke test. | In a clean environment without the `hf` extra, verify `python -c "import geoIR"` succeeds. |
| `geoIR search` | CLI users | Deprecated / unsupported | The command remains temporarily visible for compatibility, but now fails fast by design. The underlying defects still exist: `geoIR.data` exposes no `load` entry point, and `Index.search` expects `query_emb: np.ndarray` while the old CLI passed a raw `str`. The command is pending removal in a future clean CLI cut. | Keep a smoke test that verifies the deprecated command exits with a clear unsupported message and code `2`. |
| `geoIR eval` | CLI users | Removed from CLI v0 | The historical command was removed from the public CLI surface. Its previous implementation called `_SUD(obj["gt_docs"], obj["new_docs"], reference=obj["reference"])`, which did not match `SUD(query, gt_docs, new_docs, *, judges=None, policy="mean")`. | Keep it absent from CLI help until or unless a future CLI surface reintroduces a supported evaluation workflow. |
| `quick_experiment` | library/demo users | Experimental | The public docstring in `geoIR/__init__.py` suggests `beir/fiqa` style datasets, but the implementation passes `dataset` to `load_corpus(str(corpus_path))`, which treats it as a local plain-text file path. The triplet construction (`negatives = corpus[1:] + corpus[:1]`) is a synthetic rotation, not a real triplet-mining step. | Decide whether this helper is local-demo only or a real benchmark entry point; then test against a tiny fixture and align the docstring with the actual behavior. |
| `training` (classic) | model developers | Unsupported | `_train_classic` calls `self.encoder.q_model.fit(...)`, but `encoder.q_model` is built via `AutoModel.from_pretrained(model_name)` in `geoIR/retrieval/encoder.py`. `AutoModel` does not expose a `.fit(train_objectives=...)` API; only `sentence_transformers.SentenceTransformer` does. The classic path cannot execute as written. | Either wrap the encoder so its `q_model` is a `SentenceTransformer`, or rewrite `_train_classic` against the HF Trainer / a plain PyTorch loop. Add one regression test covering the chosen contract. |
| `training` (geometric) | model developers | Unsupported | Two compounding issues in `geoIR/training/trainer.py::_train_geometric`. (a) `encoder.encode` returns a detached `np.ndarray`, but the trainer calls `n_vecs.unsqueeze(1)` and `total_loss.backward()`, which require `torch.Tensor` objects with grad. (b) `Encoder._encode_batch` is decorated with `@torch.inference_mode()`, which disables autograd even if the return type were changed — so gradients cannot flow through the encoder at all. The geometric mode cannot train as written. | Introduce a gradient-aware encoding path (e.g. a separate `encode_grad` method or a flag that bypasses `inference_mode`) and add a regression test that asserts a non-zero gradient on encoder parameters after one step. |
| `research/beir_euclidean_vs_geo.py` | researchers | Experimental / research-only | The harness now resolves local BEIR-style datasets from `--dataset`, `--dataset-dir`, or `--download-dir`, and can download when `--allow-download` is set. It remains research-only until its cost profile, dependencies, and artifact contract are accepted as part of the maintained workflow. | Keep fixture-backed tests for local datasets, run one live FiQA smoke with download enabled, and document the supported recipe. |

## Evidence Snapshot

The initial support matrix is based on the following repository evidence:

- `pyproject.toml`: project metadata, optional extras (`hf`, `viz`, `dev`), CLI entry point.
- `geoIR/__init__.py`: lazy retrieval backend registration on first `load_encoder(...)` call, plus `quick_experiment` implementation (local corpus + synthetic triplets).
- `geoIR/cli.py`: current CLI contracts, including `search` deprecation/fail-fast and removal of `eval` / `report-save`.
- `geoIR/data/`: no `__init__.py`; exposes `loader.py`, `loaders.py`, `fallback.py`, none of which define a `load` symbol.
- `geoIR/retrieval/encoder.py`: eager `transformers` import, `@torch.inference_mode()` on `_encode_batch`, `np.ndarray` return type, `q_model` built as `AutoModel`.
- `geoIR/retrieval/index.py`: `Index.search(query_emb: np.ndarray, ...)` signature.
- `geoIR/eval/sud.py`: `SUD(query, gt_docs, new_docs, *, judges, policy)` signature.
- `geoIR/training/trainer.py`: classic path depends on `SentenceTransformer.fit`; geometric path depends on gradient-carrying tensors.
- `research/beir_euclidean_vs_geo.py`: parameterized BEIR dataset resolution and schema-v2 artifact generation.
- `.github/workflows/ci.yml`, `Makefile`, `README.md`: tooling and documented workflows (not re-audited in this pass).

## Module Map

### `geoIR/core`

Role:
- typed configuration,
- lightweight internal registry,
- experiment runner abstraction.

Notes:
- `core/config.py` is the closest thing to a canonical experiment schema.
- `core/runner.py` is reusable, but currently used mostly from research scripts.

### `geoIR/retrieval`

Role:
- encoder loading,
- corpus encoding,
- graph-backed index construction,
- retrieval entry points.

Notes:
- This is the most important public-surface candidate.
- It currently mixes convenience helpers, packaging concerns, and backend assumptions.

### `geoIR/geo`

Role:
- k-NN graph construction,
- curvature computation,
- differentiable geometric operations,
- graph reranking.

Notes:
- This is the main algorithmic hotspot.
- `geo/differentiable.py` and `geo/graph.py` contain the highest concentration of complexity and should be treated as core research logic.

### `geoIR/training`

Role:
- model fine-tuning orchestration for classic and geometric modes.

Notes:
- This layer is intended to bridge encoder outputs, losses, and geometric regularization.
- It currently appears under-integrated with the actual encoder implementation.

### `geoIR/eval`

Role:
- baseline retrieval metrics,
- graph-based reranking evaluation,
- LLM judge helpers,
- prototype metrics such as RARE and SUD.

Notes:
- This package mixes benchmark-style evaluation with judge-driven experimental metrics.
- Dependency boundaries are currently unclear.

### `geoIR/data`

Role:
- small local text loaders,
- fallback dataset shape,
- synthetic dataloaders.

Notes:
- This package does not currently provide a single canonical data access interface.
- Public callers should not assume BEIR or HF datasets are abstracted cleanly here yet.

### `research/`

Role:
- experiment scripts,
- result snapshots,
- exploratory workflows.

Notes:
- This area should remain outside the stable surface until its prerequisites and portability story are documented.
- The benchmark contract for `research/beir_euclidean_vs_geo.py` is documented in `research/BEIR_BENCHMARK_SPEC.md`.

## Cross-Cutting Concerns

### Dependency boundary

Current dependency usage suggests three practical tiers:

1. Core scientific runtime: `numpy`, `torch`, `networkx`, `faiss-cpu`.
2. HF and embedding stack: `transformers`, `sentence-transformers`, `datasets`, `beir`.
3. Optional research/judge integrations: `pytrec_eval`, `openai`, `GraphRicciCurvature`, plotting stack.

These tiers are not yet enforced cleanly in imports.

### Tooling boundary

The repository should converge on one workflow:

- dependency management via `uv`,
- one CI path,
- one local validation path,
- one documented install story,
- one pre-commit policy.

Current state does not yet meet that bar.

## Risks

- Public docs may promise flows that are not actually executable.
- Import-time dependency coupling can break basic library usage.
- Training and evaluation paths may appear available while still being prototype-grade.
- Research scripts can be mistaken for supported product workflows.

## Questions To Validate With The Team

1. Do we want `import geoIR` without the `hf` extra to be part of the supported contract? This PR makes that possible via lazy retrieval backend registration, but it is not yet declared stable.
2. Does the team accept the proposed CLI v0 scope: keep `encode` and `audit`, keep `audit --plot` experimental, keep `search` deprecated/unsupported until the next clean cut, and remove `eval` plus `report-save` from the public CLI surface?
3. Is `quick_experiment` a demo helper or a maintained benchmark entry point? The current docstring and implementation disagree.
4. Is training in scope for the near-term public API, or only for internal research? Both classic and geometric paths fail to execute today; neither can be promoted without non-trivial work.
5. Should `research/` remain intentionally non-portable, or should it become reproducible from a clean checkout?
6. Should `geoIR/data/` be a proper package with a documented loader surface (and a stable `load` entry point), or should callers use the submodules directly?

## Recommended Exploration Order

1. `retrieval` and CLI public contracts.
2. `training` integration contract.
3. `eval` dependency boundaries and import behavior.
4. `research` portability and tooling.

## Testing Strategy Draft

Once the team validates the support matrix, the minimum regression matrix should cover only the approved stable flows.

Suggested starting point:

- import smoke test for `import geoIR`,
- CLI smoke test for the first supported subcommand,
- `quick_experiment` fixture-based smoke test if it remains public,
- one classic training regression if classic mode is supported,
- one geometric training regression if geometric mode is supported,
- explicit exclusion of research scripts from mandatory CI until promoted.

## Non-Goals For This Draft

This document does not:

- redefine APIs,
- fix implementation issues,
- classify every module as production-ready,
- promote any flow to stable without team confirmation.
