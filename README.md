# TCGA — escalabilidad de k-NN/LSH con streaming

Proyecto de Tópicos en Grandes Volúmenes de Datos (2026). Parte de **correcciones-streaming** y aplica el feedback experimental sobre reducción de candidatos, calidad del voto, bases estratificadas, consultas completas, tiempos de exacto y resultados CSV.

## Metodología

La cohorte se construye mediante **A1 ∩ A2 ∩ A3**, con una alícuota tumoral aprobada por paciente, y se particiona en train/val/test sin compartir pacientes. Los flags A3 `Do_not_use` y `AWG_excluded_because_of_pathology` con `True`/`1` causan exclusión; no hay políticas adicionales. La matriz A1 se lee **gen por gen**, aplica `log2(max(x,0)+1)`, calcula media y desviación **solo con train**, imputa ausentes y acumula proyecciones y las distancias exactas de consultas×base en archivos mmap. No existe matriz densa pacientes×genes en el flujo principal.

El orden de la base es **anidado y estratificado**: incorpora un paciente de cada clase de entrenamiento al comienzo (si alcanza n) y agrega el resto por déficits de cuota, manteniendo `B_n ⊂ B_{n+1}`. El entrenamiento completo define los parámetros por gen para todos los n; así el experimento aísla el efecto del tamaño de la base, no cambios en normalización. Validación calibra y selecciona por n; test solo utiliza parámetros congelados.

| Método | Métrica de referencia | Hash | Re-ranking |
| --- | --- | --- | --- |
| kNN exacto coseno | Coseno de Z normalizado por paciente | No aplica | Exacto |
| LSH coseno por bandas | Coseno | `L × r` bits SimHash | **Todas** las columnas disponibles del bloque de réplica (`m=512` por defecto) |
| E2LSH p=2 (Gauss) | L2 de Z normalizado por paciente, mismo orden que coseno | `L × r` hashes p-estables | Todas las columnas Gaussianas por réplica (`256` por defecto) |
| kNN exacto L1 | L1 sobre Z sin normalización por paciente | No aplica | Exacto |
| E2LSH p=1 (Cauchy) | L1 sobre Z | `L × r` hashes p-estables | Mediana en todas las columnas Cauchy por réplica |

**No se confunden** los vecinos L1 con los de coseno. RP/PCA no son métodos principales.

## Instalación

```bash
python -m pip install -e '.[test,plot]'
# Opcional: numba/psutil para rendimiento del exacto L1
python -m pip install -e '.[performance]'
```

## Comandos — desarrollo con el demo sintético

```bash
python scripts/preparar_cohorte.py demo/raw --out demo/processed --max-genes 90
python scripts/escalabilidad.py validar --data demo/processed --out demo/results --k 3 --sizes 5 10 18

# Revisar demo/results/validation.csv y configuraciones_congeladas.json
python scripts/escalabilidad.py test --data demo/processed --out demo/results
python scripts/graficar_escalabilidad.py --csv demo/results/test_final.csv
```

El demo sintético no permite concluir resultados sobre pacientes reales.

## Comandos — datos reales TCGA

```bash
# Por defecto --cols=512, --lp-cols=256, --queries-per-phase=0 (TODAS).
python scripts/preparar_cohorte.py data/raw --out data/processed \
    --seed 2026 --transform log2clip --cols 512 --lp-cols 256
# Ajuste de (L,r,w) solo en validación:
python scripts/escalabilidad.py validar --data data/processed \
    --out results/escalabilidad --k 10 --verify-hash

# Test solo después de fijar y revisar la selección:
python scripts/escalabilidad.py test --data data/processed \
    --out results/escalabilidad --verify-hash

python scripts/graficar_escalabilidad.py --csv results/escalabilidad/test_final.csv
```

El tamaño creciente predeterminado es 50, 100, 250, 500, 1.000, 2.000, 4.000 y el total de train (hasta el máximo disponible); con muchos cánceres, n=50 es un caso de pocas muestras por clase: consultar `base_class_counts` antes de interpretarlo. `--queries-per-phase N` (opción de `preparar_cohorte.py`) limita las consultas de cada fase **solo si se solicita** (útil para desarrollo); con `0` se usan todas.

### Calibración y selección

`w_ref` es la **mediana de la distancia exacta al k-ésimo vecino en validación** (sin multiplicación automática ×4), y se evalúa `w=w_ref×{0.25, 0.5, 0.75, 1.0}`. La grilla E2LSH incluye `(L,r)=(8,2),(16,2),(16,4),(32,4),(16,8),(32,8)` hasta el máximo `--lp-cols // replicas` disponible. En LSH coseno se comprueban `(8,4),(16,4),(16,6),(32,6)`, siempre usando **todas** las columnas disponibles para re-ranking. Las columnas de hash y re-ranking se informan por separado.

**Política vigente preregistrada:** accuracy no más de 0,02 bajo su propio exacto; F1 macro no más de 0,03; cobertura ≥0,90; sensibilidad por clase (≥10 consultas), pérdida máxima 0,05; **recall@k≥0,90**; candidatos≤50 %. El recall se conserva como requisito rígido, aunque pueda no alcanzarse: si no hay configuración que lo satisfaga, se informa. Si falla cualquier objetivo, la configuración congelada alternativa es **solo diagnóstica**, con `qualified=false` y `selection_reason`. Los umbrales no se alteran tras observar test.

La concordancia del voto con exacto y la accuracy real se reportan **por separado**. Se congelan parámetros por **método, réplica y n**; el test no reoptimiza.

## Benchmark exacto desde genes (opcional, puede ser costoso)

Sin `--exact-benchmark-raw` el exacto tiene `latency_ms_median` medida con la lectura de una fila **de distancias precalculadas** y el top-k. No es el tiempo de calcular distancias a partir de genes. Para medir el costo auténtico de búsqueda exacta en streaming para **cada tamaño n y cada métrica**, añadir la carpeta original de A1/A2/A3:

```bash
python scripts/escalabilidad.py validar --data data/processed \
    --out results/escalabilidad --k 10 --sizes 50 250 1000 \
    --exact-benchmark-raw data/raw

# En la ejecución final de test (una sola vez):
python scripts/escalabilidad.py test --data data/processed \
    --out results/escalabilidad --exact-benchmark-raw data/raw
```

**Importante:** esta opción vuelve a leer A1 *para cada combinación de tamaño n y métrica*. Sigue siendo streaming y usa mmap para scores, pero su costo puede ser elevado. Genera `exact_streaming_val.csv` o `exact_streaming_test.csv` con tiempos `accumulate_seconds`, `topk_seconds`, `total_seconds_with_io` y amortización por consulta. `validation.csv` y `test_final.csv` agregan `exact_stream_ms_per_query_amortized` y `amortized_ratio_vs_exact_stream`.

**Precaución:** este cociente es **exploratorio, no una aceleración end-to-end**. La consulta ANN mide búsqueda sobre **sketch ya precalculado**, mientras que la búsqueda exacta de novo incluye todas las operaciones de comparación en los genes. Para una aceleración end-to-end rigurosa haría falta instrumentar también la generación de sketches y amortizar construcción/lectura de datos bajo el mismo escenario. No afirmar que el ratio aislado demuestra mejor tiempo de sistema completo.

## Resultados

- Preparación: `splits.npz`, `sketches.npz`, `gram.npy`, `l1.npy` (mmap), `metadata.json`, auditorías de cohorte. **No borrar ni alterar** los archivos ya usados en validación/test.
- Validación: `validation.json`, **`validation.csv`** (fila por método, n y réplica), `configuraciones_congeladas.json`.
- Test: `test_final.json`, **`test_final.csv`** y figuras opcionales con `graficar_escalabilidad.py`.
- Benchmark exacto desde genes (opcional): `exact_streaming_val.csv` / `exact_streaming_test.csv`.

`process_peak_rss_mib_cumulative` es el RSS máximo **acumulado del proceso**, no memoria aislada de cada algoritmo. `resident_bytes_estimate` no incluye todas las estructuras Python ni los sketches compartidos. La ausencia de candidatos se informa en `coverage` y `fraction_with_less_than_k` (no se hace búsqueda exhaustiva de respaldo). Las consultas y el split son iguales para todos los métodos.

## Pruebas

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q
```

Ver `GUIA_PROTOCOLO_VALIDADO.md`, `CAMBIOS_FEEDBACK.md` y `REGISTRO_VERIFICACION.md`.

## Segunda revisión: CSV, objetivos preregistrados y benchmark pareado

Leer **`CAMBIOS_FEEDBACK_2.md`** antes de interpretar resultados. Se solucionó
el bug de `n` en los CSV y se recuperó la política de calidad original
(accuracy Δ≥−0.02, F1 macro Δ≥−0.03, sensibilidad por clase Δ≥−0.05,
recall@k≥0.90, cobertura≥0.90, candidatos≤50 %). Si no se cumple, se informa;
no se alteran umbrales tras examinar test.

Los CSV se generan junto con cada JSON (`validation.csv`, `test_final.csv`).
Curva de compromiso recall@k vs fracción de candidatos y vs latencia, con todas
las configuraciones de la grilla de validación:

```bash
python scripts/graficar_compromiso.py --csv results/escalabilidad/validation.csv \
  --selection results/escalabilidad/configuraciones_congeladas.json
```

`graficar_escalabilidad.py` está pensado para `test_final.csv` (una configuración por
método). Para curvas vs n a partir de `validation.csv`, usar solo la configuración
congelada de cada método:

```bash
python scripts/graficar_congeladas.py --csv results/escalabilidad/validation.csv \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/figuras_validacion
```

La corrida final con **nueva política** requiere volver a validar (y después
probar test con un conjunto genuinamente no observado, si está disponible).
No reutilizar `configuraciones_congeladas.json` generado por la versión previa.

**Para la comparación temporal pareada por consulta**, utilizar el benchmark
opcional, que construye una matriz densa temporal solamente para medir el
exacto y mide la proyección + búsqueda ANN de la misma consulta:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python scripts/benchmark_pareado.py --data data/processed --raw data/raw \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/pareado_test.csv --phase test --queries 32

python scripts/graficar_escalabilidad.py \
  --csv results/escalabilidad/test_final.csv \
  --pareado results/escalabilidad/pareado_test.csv
```

El exacto usa una base densa precargada, ANN usa pesos de proyección
precargados; ninguno mide E/S de archivos ni preprocesamiento nuevo de A1.
Es una comparación online de consulta, no end-to-end desde datos crudos.
`--queries 0` usa todo el test; `--max-dense-mib` limita la memoria estimada.
El antiguo `--exact-benchmark-raw` permanece para medir el replay streaming,
pero no es comparable directamente con latencia de consulta ANN.
