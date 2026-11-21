

| Comentario | Qué sucedía | Corrección | Dónde |
|---|---|---|---|
| E2LSH ≈ fuerza bruta | `suggest_w` devolvía 4 × mediana y la grilla multiplicaba por 0.5, 1, 2. `select_frozen` elegía mayor recall si fallaban metas. | `w_ref` = mediana (×1), grilla 0.25/0.5/0.75/1; r hasta 8 cuando cabe; se exige ≤50% candidatos para declarar configuración calificada; selección de respaldo privilegia menor fracción de candidatos entre opciones de calidad comparable. | `experimento_streaming.py` |
| Recall@10 ≥0.90 | Al re-rankear solo con algunas columnas se perdía precisión; recall a menudo muy sensible a distancias casi empatadas. | Re-ranking usando **todas** las columnas guardadas de cada réplica, no solo L×r; el recall se reporta pero no es objetivo rígido. Se añade concordancia de voto y se mantienen accuracy/F1. | `lp/e2lsh.py`, `experimento_streaming.py` |
| 128 consultas máximas | Se submuestreaban val/test y se reducían soportes por clase. | Default `queries_per_phase=0` (todas). Submuestrear debe ser voluntario. | `preparacion_streaming.py`, `scripts/preparar_cohorte.py` |
| Sin exacto/CSV | Se medía solo la pasada global y no tiempos por n ni CSV. | `latency_ms_median` para top-k exacto con scores precalculados; CSV plano por (método,n,réplica); opción de replay streaming de A1 por n y métrica, tiempo real desde genes y su CSV separado. Cociente exploratorio identificado como **no end-to-end**. | `experimento_streaming.py`, `benchmark_exact_streaming.py`, `scripts/experimentar.py`, `scripts/graficar_escalabilidad.py` |
| Prefijos no estratificados | `base_order` era aleatorio sin cuotas de clase, afectaba accuracy del exacto en n pequeño. | Orden único anidado con presencia inicial de cada clase y déficit proporcional después; conteo por clase por n en JSON. | `splits.py`, `preparacion_streaming.py` |
| 512 columnas, solo 192 usadas | LSH re-rank utilizaba m=128 fijo; E2LSH usaba solo L×r. | LSH re-ranking usa `cols//replicas` (512 con una réplica); E2LSH usa `lp_cols//replicas` (256 por defecto); hash puede seguir usando menos. | `experimento_streaming.py` |

## Criterios de interpretación

El límite de candidatos ≤50% es un **umbral experimental** ajustable en el código, no una garantía de que existirá una solución que lo cumpla. Las configuraciones que no cumplen se etiquetan. Usar todos los features proyectados para re-ranking no implica que haya más candidatos: solo mejora su ordenación si están en el índice. Los primeros n que apenas permiten una muestra por clase deben discutirse como casos con soportes insuficientes.

`latency_ratio_vs_cached_exact` compara dos **etapas de consulta diferentes**: topk sobre scores exactos ya calculados contra búsqueda de ANN sobre sketch. No presentarlo como mejora del cálculo de distancias. El replay `--exact-benchmark-raw` mide exacto desde genes, pero el ANN usado para comparar sigue omitiendo generación de proyecciones; el ratio `amortized_ratio_vs_exact_stream` tampoco es speedup end-to-end. Para una comparación justa, incluir costo de proyección y de construcción del índice en el escenario elegido.

## Alcance de pruebas

Se comprueban las distancias exactas vs A1, proyecciones Gaussian/Cauchy, la política de estratificación, la selección consciente de candidatos, re-ranking p1/p2 con todas las columnas y ambos CSV. Las pruebas de integración usan datos sintéticos del repositorio; **no** se han obtenido mediciones reales de TCGA completo. Ver `REGISTRO_VERIFICACION.md`.
