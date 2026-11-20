# Protocolo auditado: TCGA PanCanAtlas, búsqueda ANN y clasificación

**Estado:** código validado sobre datos sintéticos y tests automáticos; **pendiente validación de la integración con los archivos originales A1/A2/A3**, que no estaban en el ZIP del repositorio. **No es un sistema clínico.**

## 1. Qué es train / val / test

1. **TRAIN (aprox. 70 % de los pacientes):** única base de referencia de vecinos; ajustar sobre train medias, desviaciones, imputación, genes sin variación, PCA e índices de búsqueda. Las etiquetas son para el voto k-NN; los algoritmos ANN no entrenan un diagnóstico de forma supervisada.
2. **VALIDACIÓN (aprox. 15 %):** consultas a la base de TRAIN; comparar distancias y algoritmos, escoger hiperparámetros, entre ellos `w` de E2LSH. Las consultas de validación **nunca** están en el índice y **nunca** se reutilizan como test. Guardar parámetros seleccionados y congelarlos.
3. **TEST (aprox. 15 %):** consultas finales, realizadas una sola vez con las configuraciones congeladas. No volver a ajustar usando resultados del test.

Particiones estratificadas por `type`, una sola alícuota por paciente. Para clases con n<3 se produce error. Por redondeo el porcentaje global puede variar ligeramente. Si se necesita reportar métricas confiables en clases escasas, usar repeticiones estratificadas en evaluación exploratoria y un test externo; **no** presentar Wilson con n=2 como sensibilidad clínica.

Toda medición de Recall@k compara vecinos aproximados contra `ExactKNN` con la **misma** base train, representación y métrica. FNR por clase = 1 - sensibilidad (recall de clasificación por clase), no confundir con `1 - Recall@k` de recuperación.

## 2. Archivos y política de cohorte

- **A1:** `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` (columnas = alícuotas, filas = genes).
- **A2:** `TCGA-CDR-SupplementalTableS1.xlsx` (hoja `TCGA-CDR`, clave `bcr_patient_barcode`, etiqueta `type`).
- **A3:** `merged_sample_quality_annotations.tsv` (clave `aliquot_barcode`, plataforma IlluminaHiSeq RNASeqV2, señales `Do_not_use` y `AWG_excluded_because_of_pathology`).

La selección es **estricta**: no se incorpora un paciente sin etiqueta, muestra tumoral válida (01, 03, 06) y anotación de A3 inequívoca; se prioriza tipo de muestra 01 salvo LAML (03) y SKCM (06). Las filas de IlluminaGA RNASeqV2 no se confunden con las de HiSeq. Una columna A1 que no coincide con A3 queda excluida y se informa en `auditoria_muestras.csv`.

Una coincidencia de alícuota completa es preferente; un barcode de A1 abreviado a 16 caracteres usa anotaciones de la muestra solo si **todas las anotaciones de alícuotas HiSeq coinciden** en la decisión. Se considera vacío/NA en dos flags de A3 equivalente a «no marcado»; **este supuesto debe verificarse con los A3 reales**. Se puede activar `--strict-annotations` para incluir también comentarios a nivel paciente/muestra/alícuota. El filtro estricto podría excluir muestras legítimas si la columna contiene observaciones inocuas: validar la política con el equipo antes de congelarla.

**IMPORTANTE:** A1 es una matriz ya normalizada/corregida por lotes y puede contener negativos. La opción por defecto es `--transform as_is` (NO aplicar un segundo log). `--transform log2clip` solamente existe para análisis de sensibilidad; convierte negativos en 0 antes del log y modifica su significado. No asumir que es preferible. La corrección original por lotes de PanCanAtlas se realizó antes de vuestro split y no equivale a evaluación clínica independiente.

Se filtran genes con varianza cero o menos de 2 valores finitos **en train**; la media y desviación se calculan en train. Valores ausentes de val/test se imputan con la media de train. Con z-score, tras centrar, la imputación es cero.

## 3. Preparar el entorno

Desde la raíz del proyecto descomprimido, con Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate                   # Linux / macOS
# En Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[test,plot]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
pytest -q
```

Fijar número de hilos **antes** de importar NumPy (para comparar tiempos). Para reproducibilidad estricta, congelar versiones con `python -m pip freeze > versions.txt` y anotar SO, CPU y RAM. El método ExactL2 usa distancias directas en float64 y es más costoso, pero es una referencia numéricamente segura.

## 4. Prueba de humo sin datos reales

El generador crea un subconjunto **completamente artificial**: **no** usar sus métricas para reportar desempeño sobre TCGA.

```bash
python scripts/generar_demo.py --out demo/raw
python scripts/preparar_cohorte.py demo/raw --out demo/processed
python scripts/experimentar.py validar --data demo/processed \
  --out demo/results --methods rp simhash simhash_rp lsh --k 3
python scripts/experimentar.py test --data demo/processed --out demo/results
```

Verificar que existan `demo/processed/pacientes.csv`, `auditoria_muestras.csv`, `splits.npz`, `X_gene_major.npy` y `metadata.json`. Tras validación deben existir `demo/results/validation.json` y `configuraciones_congeladas.json`. Tras test debe existir `test_final.json`. En el test aparece solo **una configuración previamente seleccionada de validación por método**.

## 5. Preparar el TCGA real

Depositar A1, A2 y A3 dentro de `data/raw/` (directamente o en las subcarpetas UUID de gdc-client). **No** copiar los datos reales dentro del repositorio versionado. Comprobar los archivos y los MD5 especificados en el README antes de comenzar.

```bash
find data/raw -type f \( -name 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv' \
  -o -name 'TCGA-CDR-SupplementalTableS1.xlsx' \
  -o -name 'merged_sample_quality_annotations.tsv' \)

python scripts/preparar_cohorte.py data/raw --out data/processed --seed 2026
```

**ATENCIÓN:** el A1 oficial pesa cerca de 1.9 GB y se lee dos veces. La matriz `.npy` procesada usa aprox. `n_muestras × n_genes × 4` bytes en disco y la búsqueda exacta requiere además RAM para train/consultas. Para inspeccionar la infraestructura en una prueba pequeña:

```bash
python scripts/preparar_cohorte.py data/raw --out data/dev_first_256genes --max-genes 256
```

`--max-genes` selecciona los **primeros** 256 genes de A1, no genes óptimos. Sirve **solo** para depuración. Para una comparación científica de 1.000/5.000 genes, definir selección reproducible de genes, ajustada únicamente en train, en lugar de usar los primeros genes.

En `metadata.json`, revisar número de pacientes, genes leídos y retenidos, conteos por clase y causas de exclusión. **No asumir que se mantendrán los 10.215 pacientes del pipeline anterior.** Abrir y revisar muestras aleatorias en `auditoria_muestras.csv` contra A3 original.

## 6. Verificar cohortes y ausencia de fuga

```bash
python - <<'PY'
import csv
import numpy as np
from pathlib import Path
r = Path('data/processed')
with (r/'pacientes.csv').open(encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
assert len({v['patient_id'] for v in rows}) == len(rows), 'Pacientes duplicados'
with np.load(r/'splits.npz') as z:
    a,b,c = (set(z[k].tolist()) for k in ('train','val','test'))
    assert not(a&b or a&c or b&c), 'Fuga: pacientes repetidos'
    assert (a|b|c) == set(range(len(rows))), 'Falta asignar pacientes'
    assert set(z['labels']) == {v['type'] for v in rows}
    print('Pacientes:',len(rows),'train/val/test:',len(a),len(b),len(c))
X = np.load(r/'X_gene_major.npy', mmap_mode='r')
assert X.shape[1] == len(rows), 'Matriz y etiquetas no coinciden'
assert np.isfinite(X).all(), 'Hay NaN/Inf después del preprocesamiento'
print('Datos válidos:', X.shape[0], 'genes x', X.shape[1], 'pacientes')
PY
```

Los tests automatizados incluyen una prueba en que se cambia **solo un valor de validación** y se demuestra que medias, desviaciones y perfiles de train no cambian. También se prueba que muestras marcadas `Do_not_use`, mismatches de plataforma y alícuotas duplicadas se manejan de modo determinista.

## 7. Validar métodos y congelar hiperparámetros

Para comparar métodos que aproximan **coseno**:

```bash
python scripts/experimentar.py validar --data data/processed \
  --out results/protocolo_cosine --metric cosine --k 10 \
  --methods rp simhash simhash_rp lsh
```

Opcionalmente añadir `pca e2lsh`. Los métodos se calculan secuencialmente para limitar memoria, pero los resultados dependen del hardware. El `w` de E2LSH se calcula exclusivamente con consultas de **validación** y se guarda como número congelado para la ejecución final.

Estudio separado de magnitud global sin normalización de filas:

```bash
python scripts/experimentar.py validar --data data/processed --metric l2 \
  --out results/protocolo_l2 --k 10 --methods rp e2lsh
python scripts/experimentar.py validar --data data/processed --metric l1 \
  --out results/protocolo_l1 --k 10 --methods e2lsh
```

**IMPORTANTE:** el z-score por gen por defecto no elimina la magnitud global de cada fila, pero sí modifica las unidades; comparar también `--no-zscore` creando **otra cohorte** en una carpeta diferente. Elegir representación/métrica **usando solo validación**: cualquier comparación entre múltiples representaciones debe preceder a test final y registrar cuál se eligió.

Los métodos se restringen para no comparar métricas incompatibles: `cosine` permite RP normalizado, PCA normalizado, SimHash, LSH y E2LSH p=2 en unitarios; `l2` permite RP sin normalizar y E2LSH p=2; `l1` permite E2LSH p=1. ExactKNN corresponde siempre a la métrica que se está aproximando.

`validation.json` contiene Recall@k, exactitud, F1, balanced accuracy, FNR/sensibilidad por clase, IC de Wilson de sensibilidad, fracción de candidatos, consultas con menos de k vecinos, latencia de extremo a extremo (firma/proyección+consulta), construcción y estimación de RAM. Las abstenciones se cuentan como errores y tienen columna `SIN_PREDICCION` en la matriz. Incluye IC bootstrap emparejado de diferencia de exactitud frente al exacto.

`configuraciones_congeladas.json` elige una configuración por método. Marca `qualified=false` cuando **ninguna configuración** de ese método alcanza las metas de calidad del README (Recall@10 >= 0.90, pérdida accuracy <= 0.02, pérdida F1-macro <= 0.03), más una comprobación de pérdida máxima de sensibilidad <= 0.05 en clases de validación con soporte >= 10. La fracción de candidatos <= 20 %, memoria y speedup también deben comprobarse manualmente para alcanzar **todas** las metas del README: no son condiciones automáticas de `qualified`.

## 8. Test final (una vez por experimento elegido)

Revisar los resultados de validación y decidir la métrica/representación y los algoritmos definitivos ANTES de ejecutar:

```bash
python scripts/experimentar.py test --data data/processed \
  --out results/protocolo_cosine \
  --selection results/protocolo_cosine/configuraciones_congeladas.json \
  --verify-hash
```

`--verify-hash` relee el archivo de matriz para comprobar que no cambió desde la preparación; `dataset_id` evita utilizar configuraciones de otro procesamiento/split. **No comparar versiones de `test_final.json` para seguir tuneando.** Usar solo test para informe final, además de intervalos y resultados por clase.

## 9. Cambios de algoritmos y qué se mantiene

- `ExactKNN` L2: **corregido** para usar `scipy.spatial.distance.cdist` en float64 y no prefiltrar vecinos por distancias inexactas (más lento, pero baseline numérico).
- `PCAKNN`: se añadió `resident_bytes()` que cuenta base retenida `_Xn` y distancias; el contrato anterior de `index_bytes()` sigue igual para no romper tests. Se usa `keep_data=False` en el nuevo runner.
- `SimHashRPKNN` (**nuevo**): firma SimHash con re-ranking en RP y mide transformación de la consulta. No necesita guardar toda la base d-dimensional. Hace scan de firmas, no es sublineal.
- `E2LSH`: el algoritmo no se reescribió; **el ajuste de `w` en el experimento antiguo** `escalabilidad.py` ya no usa consultas evaluadas. Para el protocolo nuevo, `w` se aprende exclusivamente en validación y se congela.
- `BandedLSH`: código de búsqueda conservado; el nuevo runner mide consultas sin k candidatos y admite pruebas de bandas. Sin `fallback` se abstiene; activar fallback introduce costo lineal que debe informarse, nunca ocultarse.
- `ProjectedKNN`, SimHash básico, MinHash y sketches: sin cambios de lógica. MinHash/sketches de mutaciones no son competidores del experimento principal sobre RNA continuo.
- `scripts/experimento_completo.py` y `scripts/experimento_escalabilidad.py` son **exploratorios históricos**: **no** aplican A3 ni constituyen el flujo train/val/test final. Usar para resultados definitivos **solo** `preparar_cohorte.py` + `experimentar.py`.

## 10. Alcance y pendientes conocidos

1. Faltan los archivos originales en el ZIP. La política de flags de A3, cruces de alícuotas y el embudo final de pacientes deben verificarse contra originales reales.
2. A1 ya fue corregida por lotes externamente antes de las particiones: existe potencial de dependencia entre muestras en el preprocesamiento de origen. El resultado será validación interna de TCGA, no validación externa clínica.
3. El barrido incluido es pequeño, pensado como base reproducible de experimentación. Para el informe final ejecutar varias semillas (de preferencia repeticiones predefinidas), describir SO/CPU/RAM/hilos, latencia total y separada, sensibilidad por clase e intervalos; no usar test para seleccionar semillas.
4. Los IC Wilson con soportes muy pequeños son amplios; no garantizan sensibilidad clínica. El objetivo es clasificar tipos histológicos previamente caracterizados, no detectar presencia/ausencia de cáncer en personas sanas.
5. `p95` en el runner es por consulta de **una ejecución**, tras 2 consultas de calentamiento; el README original propone 5 repeticiones y 5 semillas. Para afirmar robustez temporal se deben agregar repeticiones/estadística por corrida.
6. `resident_bytes_estimate` no es pico de memoria del proceso: medir RSS externo para resultados de hardware, en especial con BLAS/PCA.

**Fuentes:** GDC PanCanAtlas: https://gdc.cancer.gov/about-data/publications/pancanatlas; anotaciones de calidad de RNA discutidas en https://github.com/umccr/TCGA-data-harmonization/tree/master/expression.
