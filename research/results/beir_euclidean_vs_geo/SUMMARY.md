# BEIR Benchmark Summary

This file is generated only from `config.json` + `beir_results.json` artifacts.
It accepts only benchmark schema v2; any other or invalid run fails summary generation.
Historical manual notes elsewhere in the repo are non-authoritative and intentionally excluded here.
That includes the previous `msmarco-passage` claims, which are ignored until a backed run exists.
Hard-graph ranking is currently explicit as absent; the supported comparison is dense cosine baseline vs one candidate path.

Valid runs discovered: `6`

## Latest Backed Runs

| Dataset | #Docs | Rerank | Baseline | Candidate | Baseline nDCG@10 | Candidate nDCG@10 | Δ nDCG@10 | Baseline Recall@10 | Candidate Recall@10 | Δ Recall@10 | Baseline ms | Candidate ms | Cand encode ms | Cand build ms | Cand rerank ms | Cand RSS MB | Cand VRAM MB | Cand purity@k | Cand overlap@k | γ | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fiqa | 1,000 | none | Dense cosine baseline | Soft graph local | 0.6081 | 0.4512 | -0.1569 | 0.7143 | 0.5714 | -0.1429 | 147885.9 | 118559.5 | 117843.3 | 593.0 | 123.3 | 1834.6 | n/a | 0.0045 | 0.9311 | 0.555 | 2026-09-12_17-09-22 |
| fiqa | 1,000 | ppr@100 | Dense cosine baseline | Soft graph + PPR | 0.6081 | 0.5722 | -0.0359 | 0.7143 | 0.7143 | 0.0000 | 200207.6 | 1459.2 | 0.0 | 1308.0 | 151.2 | 605.8 | n/a | 0.0045 | 0.9311 | 0.555 | 2026-09-12_17-16-00 |
| fiqa | 1,000 | ppr@200 | Dense cosine baseline | Soft graph + PPR | 0.6081 | 0.5483 | -0.0598 | 0.7143 | 0.6667 | -0.0476 | 15228.4 | 250.3 | 0.0 | 175.8 | 74.6 | 615.0 | n/a | 0.0045 | 0.9311 | 0.555 | 2026-09-12_17-20-03 |
| fiqa | 5,000 | none | Dense cosine baseline | Soft graph local | 0.5489 | 0.4035 | -0.1454 | 0.6869 | 0.5960 | -0.0909 | 549680.3 | 521747.8 | 518289.5 | 2830.5 | 627.8 | 1872.2 | n/a | 0.0050 | 0.9310 | 0.558 | 2026-09-12_17-20-24 |
| fiqa | 5,000 | ppr@100 | Dense cosine baseline | Soft graph + PPR | 0.5489 | 0.5072 | -0.0416 | 0.6869 | 0.6364 | -0.0505 | 204105.3 | 3599.1 | 0.0 | 3305.4 | 293.7 | 1484.7 | n/a | 0.0050 | 0.9310 | 0.558 | 2026-09-12_17-30-45 |
| fiqa | 5,000 | ppr@200 | Dense cosine baseline | Soft graph + PPR | 0.5489 | 0.4824 | -0.0664 | 0.6869 | 0.6364 | -0.0505 | 159631.7 | 6470.4 | 0.0 | 5785.3 | 685.2 | 1509.5 | n/a | 0.0050 | 0.9310 | 0.558 | 2026-09-12_17-38-05 |

## Decision Gate

- Soft-kNN 1k gate: do not promote soft-kNN yet (`Δ nDCG@10=-0.1569`, `Δ Recall@10=-0.1429`).
- Soft-kNN 5k gate: trigger geometric redesign (`Δ nDCG@10=-0.1454`).
- PPR gate: freeze PPR for the next cycle; every FiQA PPR run loses against the dense baseline.

