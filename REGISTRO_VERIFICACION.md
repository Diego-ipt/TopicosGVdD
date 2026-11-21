
## Suite completa

Desde la raíz del proyecto:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=src python -m pytest -q --disable-warnings
```

**Resultado observado:** **150 passed, 14 skipped, 6 warnings, 0 failed**.
Las advertencias son de pruebas numéricas antiguas con entradas ausentes/NaN;
las pruebas omitidas requieren datos externos reales, no incluidos.

## Nuevas pruebas

- `_write_csv`: al copiar `metrics`, no se sobreescribe `n` (tamaño de base)
  con `metrics['n']` (consultas). Regresión con tamaños 50/250/1991 y 428
  consultas.
- Metas rígidas originales: accuracy 0.02, F1 macro 0.03, sensibilidad
  (soporte≥10) 0.05, recall@k 0.90; cobertura 0.90 y candidatos 0.50.
- `final_test_streaming` rechaza configuraciones congeladas con otro protocolo.
- Benchmark online pareado con demo: los exactos densos de coseno y L1
  corresponden al mismo tamaño de base y consultas que LSH/E2LSH, incluyendo
  proyección online. Comprobación de acuerdo con verdad exacta streaming.

## Ejecución de integración en demo (no TCGA real)

Se ejecutó validación y test con `--sizes 5 10 18`, seguido del benchmark
pareado con dos consultas, y la generación de figuras a partir de los nuevos
CSV. Se verificó `n` = 5, 10, 18 (no `n_queries`=6 en las mismas filas),
`pareado_test.csv` con 15 filas y **nueve PNG** (incluidos tiempos y razón
pareada). El exacto denso coincidió con el exacto streaming para las consultas
evaluadas en demo (`dense_exact_agreement_with_streaming_oracle=1`).

Las razones de tiempo del demo no demuestran aceleración sobre TCGA: el demo
solo contiene decenas de muestras y pocas decenas de genes.

## Restricciones

El pipeline real A1/A2/A3 sigue siendo streaming; la matriz densa de genes es
solo temporal y opcional para el benchmark pareado. No se midió TCGA completo.
La comparación temporal pareada parte de vectores de genes estandarizados e
índices preparados en memoria. No mide el sistema desde archivos crudos ni
incluye indexación offline, procesamiento de E/S o carga de datos.

Inspeccionar resultados de test antes de esta revisión implica que el test ya
no es completamente virgen. Es una limitación que se debe informar para no
presentar la evaluación como una validación confirmatoria independiente.
