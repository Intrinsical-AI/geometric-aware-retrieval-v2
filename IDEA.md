
1. **El espacio latente como variedad maleable**
   Imagina que cada query y cada documento no viven en un espacio plano, sino sobre una **superficie curva** (una variedad riemanniana) que el modelo **aprende a deformar**. Esa deformación es tu función $\phi$.

2. **Métrica inducida y geodésicas**
   Junto a $\phi$ nace una **métrica local** $g(x)$ (del jacobiano de $\phi$) que dicta **cómo medir distancias** en cada punto. Las verdaderas “distancias semánticas” son las **geodésicas** sobre esa superficie.

3. **Pérdida contrastiva geométrica**
   En lugar de usar InfoNCE con cosine, aplicas **InfoNCE-geo**, que compara anclas, positivos y negativos **según su distancia geodésica** $d_g$. Con ello enseñas al encoder a respetar la curvatura relevante para la tarea.

4. **Regularización de curvatura**
   Incorporas un término suave ($\mathcal L_{Ricci}$ u opcionalmente $\mathcal L_{Forman}$) que **penaliza las regiones de curvatura negativa excesiva**, empujando tu espacio a consolidarse en zonas “bien conectadas” y aplanarse donde interese.

5. **Evaluación centrada en el LLM**
   Usas métricas como **RARE** (¿cuánto mejora la generación del LLM con tus docs?) y **SUD** (¿los docs que recuperas pero no etiquetados ayudan realmente?) para cerrar el bucle: **lo que importa es mejorar la respuesta generativa**, no sólo el recall.

---

> **En conjunto:** aprendes **de forma end-to-end** una variedad y su métrica, recuperas documentos por **geodésicas** en ese espacio y validas que **esa geometría realmente potencia** la generación de respuestas. Así transformas la metáfora del “cartógrafo” en un sistema operativo concreto.
