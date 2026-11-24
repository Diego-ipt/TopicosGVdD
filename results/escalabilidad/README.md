# Resultados de VALIDACIÓN y TEST final sobre la cohorte real (corridas del 22-11-2026)

> Esta carpeta mezcla resultados de dos etapas del proyecto. **No combinarlos en una misma tabla.**
>
> | Archivo | Origen | Protocolo |
> |---|---|---|
> | `expresion.csv`, `mutaciones.csv`, `resumen.json` | rama `main`, commit `bd9b65d` (pipeline antiguo) | cohorte 10.215, primeros 5.000 genes, sin train/val/test por paciente |
> | `validation.{csv,json}`, `configuraciones_congeladas.json`, `pareado_val.csv`, `diagnostico_rerank_val.csv`, `figuras_validacion/` | esta corrida | cohorte 9.227, train/val/test por paciente, fase VALIDACIÓN |
> | `test_final.csv`, `pareado_test.csv`, `figuras_test/` | esta corrida | misma cohorte, fase TEST final (una sola vez, configuraciones congeladas) |
>
> Lo mismo aplica a `results/figures/escalabilidad_*.png` (pipeline antiguo).

## Qué se ejecutó

- **Fases:**
  - VALIDACIÓN: se eligieron y congelaron las configuraciones.
  - **TEST final:** ejecutado **una sola vez** el 22-11-2026, con las 40 configuraciones congeladas (5 métodos × 8 tamaños) y sin reajustar nada.
  - Antes y después del test se verificó (md5) que `configuraciones_congeladas.json` y los archivos de validación no cambiaron.
- **Código:** protocolo del commit `f597aa7`, más los scripts de gráficos (`graficar_compromiso.py`, `graficar_congeladas.py`) y de diagnóstico (`diagnostico_rerank.py`), que no cambian el protocolo.
- **Datos:** TCGA PanCanAtlas A1/A2/A3 (tamaños verificados: 1.882.540.959 / 2.945.129 / 8.463.670 bytes).
  - `dataset_id` = `072bac689034c9fe199b0cabeba249f443089256f995a4c2f47f6aaa192593da` (igual en `metadata.json`, `validation.json` y `configuraciones_congeladas.json`).
- **Cohorte:** 9.227 pacientes, 33 tipos (train 6.459 / val 1.384 / test 1.384). Genes: 20.531 leídos, 20.306 usados.
- **Parámetros:**
  - transform `log2clip` + z-score con estadísticas de train;
  - 512 columnas coseno y 256 columnas p-estables;
  - semilla 2026, 1 réplica, k = 10;
  - 1.384 consultas de validación y 1.384 de test;
  - tamaños de base n = 50, 100, 250, 500, 1000, 2000, 4000, 6459.
- **Política de calidad** (sin modificar):
  - recall@10 ≥ 0,90;
  - pérdida de accuracy ≤ 0,02;
  - pérdida de F1 ≤ 0,03;
  - pérdida de sensibilidad por clase ≤ 0,05;
  - cobertura ≥ 0,90;
  - candidatos ≤ 50 %.
  - **Ninguna configuración aproximada la cumple** (todas con `qualified=false`). Las configuraciones congeladas son de diagnóstico: regla `diagnostic_unqualified_near_best_accuracy_then_candidates`.
- **Entorno:**
  - WSL2, Ubuntu 26.04.1, kernel 6.6.87.2;
  - CPU AMD Ryzen 7 7735HS, 16 hilos visibles; ~7,4 GB de RAM asignados a WSL;
  - Python 3.12.16, numpy 2.5.5, pandas 3.1.1, matplotlib 3.11.2, numba 0.68.1;
  - benchmarks con `OPENBLAS/OMP/MKL/NUMBA_NUM_THREADS=1`.
- **Pruebas automáticas:** `150 passed, 14 skipped` (re-ejecutadas el 22-11-2026 tras agregar los scripts de gráficos).

## Comandos (desde la raíz del repo)

```bash
python scripts/preparar_cohorte.py data/raw --out data/processed --seed 2026 --transform log2clip --cols 512 --lp-cols 256   # 166,6 s, pico 791 MB
python scripts/escalabilidad.py validar --data data/processed --out results/escalabilidad --k 10 --verify-hash
python scripts/benchmark_pareado.py --data data/processed --raw data/raw \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/pareado_val.csv --phase val --queries 0 --max-dense-mib 2048
python scripts/graficar_escalabilidad.py --csv results/escalabilidad/validation.csv \
  --pareado results/escalabilidad/pareado_val.csv --out results/escalabilidad/figuras_validacion
python scripts/graficar_compromiso.py --csv results/escalabilidad/validation.csv \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/figuras_validacion --sizes 1000 6459
python scripts/graficar_congeladas.py --csv results/escalabilidad/validation.csv \
  --selection results/escalabilidad/configuraciones_congeladas.json --out results/escalabilidad/figuras_validacion
python scripts/diagnostico_rerank.py data/processed results/escalabilidad/diagnostico_rerank_val.csv
# TEST final (una sola vez, con las configuraciones congeladas)
python scripts/escalabilidad.py test --data data/processed --out results/escalabilidad --verify-hash   # 40 s
python scripts/benchmark_pareado.py --data data/processed --raw data/raw \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/pareado_test.csv --phase test --queries 0 --max-dense-mib 2048
python scripts/graficar_escalabilidad.py --csv results/escalabilidad/test_final.csv \
  --pareado results/escalabilidad/pareado_test.csv --out results/escalabilidad/figuras_test
```

Logs (no versionados): `logs/preparar.log`, `logs/validar.log`, `logs/pareado_val.log`, `logs/test.log`, `logs/pareado_test.log`. `validation.json` (26 MB) y `test_final.json` (3 MB) tampoco se versionan por tamaño.

## Qué figura usar

| Figura | Sirve para la PPT | Nota |
|---|---|---|
| `compromiso_recall_n6459.png` | sí | todas las configuraciones; el anillo marca la congelada. El panel de latencia NO incluye la proyección de la consulta. |
| `tiempo_pareado_vs_n.png` / `speedup_pareado_vs_n.png` | sí | benchmark de consulta individual: exacto denso vs ANN + proyección; sin E/S ni construcción |
| `accuracy_congelada_vs_n.png`, `recall_congelada_vs_n.png`, `candidatos_congelada_vs_n.png` | sí | solo la configuración congelada de cada método |
| `accuracy_vs_n.png`, `recall_vs_n.png`, `candidatos_vs_n.png`, `concordancia_vs_n.png` (en `figuras_validacion/`) | **no** | `graficar_escalabilidad.py` dibuja toda la grilla unida por líneas; pensado para `test_final.csv` |
| `figuras_test/*.png` | sí | test final: una configuración congelada por método, así que las curvas se leen bien |

## Cifras clave del TEST final (n = 6459)

| Método (configuración congelada) | Accuracy (Δ vs su exacto) | recall@10 | Candidatos | ms/consulta (pareado) | Speedup |
|---|---|---:|---:|---:|---:|
| Exacto coseno | 0,9270 | 1 | 100 % | 25,81 | — |
| LSH L=16, r=6 | 0,9169 (−1,01 pp) | 0,682 | 24,5 % | 5,74 | 4,49× |
| E2LSH p=2 L=32, r=4, w×1,0 | 0,9162 (−1,08 pp) | 0,464 | 13,3 % | 3,81 | 6,77× |
| Exacto L1 | 0,9335 | 1 | 100 % | 43,65 | — |
| E2LSH p=1 L=16, r=2, w×0,75 | 0,8793 (−5,42 pp) | 0,372 | 24,8 % | 9,20 | 4,75× |

- El test confirma la validación: el recall cambia 0,01 o menos en todos los métodos.
- **Ninguna configuración cumple la política de calidad en test** (0 de 40); el recall sigue bajo 0,90.

## Cifras clave de VALIDACIÓN (n = 6459)

| Método (configuración congelada) | Accuracy (Δ vs su exacto) | recall@10 | Candidatos | ms/consulta (pareado) | Speedup |
|---|---|---:|---:|---:|---:|
| Exacto coseno | 0,9335 | 1 | 100 % | 21,43 | — |
| LSH L=16, r=6 | 0,9263 (−0,72 pp = 10 de 1.384 consultas) | 0,683 | 24,5 % | 5,09 | 4,21× |
| E2LSH p=2 L=32, r=4, w×1,0 | 0,9061 (−2,75 pp) | 0,460 | 13,3 % | 3,73 | 5,74× |
| Exacto L1 | 0,9342 | 1 | 100 % | 40,55 | — |
| E2LSH p=1 L=16, r=2, w×0,75 | 0,8902 (−4,41 pp) | 0,374 | 24,8 % | 9,43 | 4,30× |

**Memoria (qué se midió):**
- La base densa del exacto (6.459 × 20.306 float32) ocupa 524,6 MB.
- El índice LSH ocupa 14,5 MB; los de E2LSH, 9,1 MB (p=2) y 7,9 MB (p=1).
- Para consultar, el ANN además necesita la matriz de proyección: 83,2 MB en LSH (512 columnas) y 41,6 MB en E2LSH (256 columnas).
- La comparación justa es ~525 MB contra ~98 MB (LSH), no contra 14,5 MB.

## Diagnóstico de re-ranking (`diagnostico_rerank_val.csv`, solo validación)

Pregunta: ¿el recall se pierde al generar candidatos o al ordenarlos?
- Los candidatos se ordenan en el espacio proyectado (512 o 256 columnas).
- Se compara con ordenar **los mismos candidatos** usando la distancia exacta.

| n | Configuración | Candidatos | recall (orden proyectado, protocolo) | recall (orden exacto, mismos candidatos) |
|---:|---|---:|---:|---:|
| 6459 | LSH 16×6 (congelada) | 24,5 % | 0,683 | 0,856 |
| 6459 | LSH 32×6 | 41,0 % | 0,725 | **0,965** |
| 6459 | LSH 16×4 | 63,2 % | 0,731 | 0,981 |
| 6459 | E2LSH p=2 32×4 (congelada) | 13,3 % | 0,460 | 0,546 |
| 6459 | E2LSH p=1 16×2 w×0,75 (congelada) | 24,8 % | 0,374 | 0,573 |
| 1000 | LSH 16×6 (congelada) | 24,6 % | 0,708 | 0,785 |
| 1000 | LSH 32×6 | 41,1 % | 0,787 | **0,930** |

**Lectura:**
- La mayor parte del recall perdido en LSH se pierde al ordenar en el espacio proyectado, no al generar candidatos.
- Con re-ranking exacto, LSH 32×6 superaría la meta de 0,90 examinando ~41 % de la base, dentro del límite de 50 %.
- Hacerlo exige acceder a los vectores originales de los candidatos (memoria o disco) y aumenta el tiempo por consulta. Es un **cambio de método**: decisión del equipo, a evaluar solo en validación.

## Limitaciones

- Una semilla y una réplica: no hay estimación de variabilidad.
- El test ya se usó una vez. Cualquier cambio posterior de método (por ejemplo, el re-ranking exacto) se decide solo con validación, y un segundo test se informa como tal.
- El benchmark de tiempos excluye E/S y construcción de índices; se corrió con un hilo y en WSL.
- 33 clases desbalanceadas; con n pequeño, algunas clases tienen muy pocos ejemplos en la base.
- El plan original de `main` pedía candidatos ≤ 20 %; el protocolo actual permite ≤ 50 %. Pendiente de acuerdo del equipo.
