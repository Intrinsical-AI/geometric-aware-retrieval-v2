# **Resumen** de los resultados

| Dataset         | #Docs | Rerank   | nDCG\@10                  | Recall\@10      | Time (ms) Hard → Soft    | γ     |
| --------------- | ----- | -------- | ------------------------- | --------------- | ------------------------ | ----- |
| fiqa            | 1 000 | none     | 0.7202 → 0.7413 (+0.0211) | 0.875 → 0.875   | 17.1 → 105.7 (+520 %)    | 0.556 |
| fiqa            | 1 000 | ppr\@100 | 0.7202 → 0.6952 (–0.0250) | 0.875 → 0.875   | 15.3 → 117.9 (+671 %)    | 0.556 |
| fiqa            | 1 000 | ppr\@200 | 0.7202 → 0.6491 (–0.0711) | 0.875 → 0.875   | 22.2 → 113.9 (+413 %)    | 0.556 |
| fiqa            | 5 000 | none     | 0.6197 → 0.5458 (–0.0739) | 0.7708 → 0.6875 | 208.7 → 1 139.8 (+447 %) | 0.554 |
| fiqa            | 5 000 | ppr\@100 | 0.6197 → 0.5625 (–0.0572) | 0.7708 → 0.6875 | 225.6 → 776.3 (+244 %)   | 0.554 |
| fiqa            | 5 000 | ppr\@200 | 0.6197 → 0.5213 (–0.0984) | 0.7708 → 0.6458 | 219.3 → 790.5 (+260 %)   | 0.554 |
| msmarco-passage | 1 000 | none     | 0.7202 → 0.7413 (+0.0211) | 0.875 → 0.875   | 19.6 → 115.1 (+487 %)    | 0.556 |
| msmarco-passage | 1 000 | ppr\@100 | 0.7202 → 0.6952 (–0.0250) | 0.875 → 0.875   | 17.3 → 109.7 (+534 %)    | 0.556 |
| msmarco-passage | 1 000 | ppr\@200 | 0.7202 → 0.6491 (–0.0711) | 0.875 → 0.875   | 16.3 → 98.2 (+502 %)     | 0.556 |

---

## Análisis

1. **Soft-kNN sin reranking (none)**

   * En **subsets pequeños** (1 000 docs), mejora estable de **+2.1 pp** en nDCG\@10 tanto en FiQA como en MS MARCO.
   * El coste temporal sube \~5×–6× (de ≈17 ms a ≈105 ms), aceptable en prototipos.

2. **Soft-kNN + PPR**

   * Introducir PPR sobre el grafo soft **degrada** sistemáticamente nDCG\@10 (–2.5 pp con top-100; –7.1 pp con top-200).
   * Parece que un reranking de difusión penaliza la relevancia local en estos subconjuntos.

3. **Escalado a 5 000 docs**

   * La ganancia en **nDCG\@10 se invierte** (soft < hard, –7.4 pp sin PPR). Aquí el grafo es más grande y la uniformidad de grado (20 fijo) arrastra vecinos poco relevantes.
   * El tiempo con soft-kNN pasa a \~1.1 s (O(1000 ms)), versus \~200 ms en hard, un salto 5–6× similar al caso pequeño, pero en absoluto viable sin optimización GPU/indices aproximados.

4. **γ auto-calibrado estable (\~0.554–0.556)**

   * Da igual el corpus o tamaño: la fórmula de τ-fix converge siempre alrededor de 0.55. Podría valer fijarlo y saltarse el paso de calibración para ahorrar tiempo.

---

## Interpretación y siguientes pasos

* **Soft-kNN puro** muestra un **beneficio real** en escenarios **ligeros** (<1 000 docs). Es donde mejor aplica: prototipos con colecciones reducidas o stages intermedios de reranking.
* **Para colecciones medianas** (≥5 000 docs), el grafo soft “diluye” la relevancia—necesitamos:

  1. **Top-m candidato previo** (FAISS) antes de soft-kNN
  2. **Adaptar τ** según el tamaño (no solo la varianza global σ²).
  3. **Optimizar cálculos de grafo** (GPU, PyKeOps, muestreo más agresivo).
* **PPR** no está funcionando bien en estas configuraciones iniciales; conviene explorar otros re-rankers geométricos (p.ej. Personalized PageRank con prior en el query embedding o Graph-Neural-ReRank).

En resumen, el **turrón gordo** debería centrarse en:

1. Escalar soft-kNN a corpus medianos/grandes con índices aproximados y cálculo en GPU.
2. Refinar τ y/o la forma de ponderar pesos de arista según posición en ranking.
3. Alternativas a PPR para explotar la estructura del grafo.
