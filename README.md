# TopicosGVdD

**Búsqueda aproximada de perfiles de expresión génica similares con proyecciones aleatorias, SimHash y LSH para la clasificación del tipo de cáncer (TCGA Pan-Cancer)**

Proyecto semestral del curso **Tópicos en Grandes Volúmenes de Datos (503302)**, Universidad de Concepción, 2026.

|             |                                                   |
| ----------- | ------------------------------------------------- |
| Profesora   | Cecilia Hernández                                |
| Ayudante    | Álvaro Guzmán                                   |
| Integrantes | Antonia Montero · Diego Pérez · Angie Ramírez |

---

## 1. Problema

Cada tumor del TCGA se describe con la expresión de **≈ 20.500 genes**. Dado el perfil de un tumor nuevo, queremos:

1. **Recuperar los k pacientes más parecidos** de una base de datos (búsqueda k-NN por similitud coseno).
2. **Inferir el tipo de cáncer** (33 tipos TCGA) por voto de esos vecinos.

| Magnitud                     | Valor                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| Muestras × genes            | ≈ 11.000 × 20.500 ≈ 2,3 × 10⁸ valores (≈ 0,9 GB en float32; ≈ 1,8 GB en texto) |
| Una consulta exacta          | ≈ n·d ≈ 2,3 × 10⁸ operaciones                                                    |
| Todos los pares de pacientes | ≈ 1,2 × 10¹² operaciones                                                          |
| Memoria por paciente         | 82.000 B (exacto) → 1.024 B (RP, m = 256) → 32 B (SimHash, b = 256)                 |

El cuello de botella no es almacenar la matriz (cabe en RAM), sino **comparar vectores de 20.000 dimensiones** y cómo crece ese costo al aumentar el número de pacientes.

### Pregunta de investigación

> ¿En qué medida las representaciones compactas (proyecciones aleatorias y firmas SimHash) y LSH permiten reducir el costo de buscar perfiles de expresión génica similares, manteniendo alta recuperación de vecinos y preservando la información necesaria para identificar el tipo de cáncer?

El proyecto es **experimental y comparativo**: no asumimos que LSH sea más rápido. Con n ≈ 10.000 es probable que la búsqueda exacta con BLAS sea competitiva; medimos **en qué condiciones** conviene cada método y cuánta calidad se sacrifica. Un resultado negativo bien medido también es un resultado.

---

## 2. Métodos

| Código          | Método                                        | Idea                                                                 | Consulta              | Memoria (n = 11.000) |
| ---------------- | ---------------------------------------------- | -------------------------------------------------------------------- | --------------------- | -------------------- |
| **EX**     | k-NN exacto (línea base)                      | Coseno contra todos los perfiles, con BLAS                           | O(n·d)               | ≈ 0,9 GB            |
| **RP**     | Proyección aleatoria (Johnson–Lindenstrauss) | 20.500 → m dimensiones (m = 64…1.024) y k-NN ahí                  | O(d·m + n·m)        | ≈ 11 MB (m = 256)   |
| **SH**     | SimHash (Charikar, 2002)                       | Firma de b bits por paciente; distancia de Hamming (XOR + popcount)  | O(d·b + n·b/64)     | ≈ 352 KB (b = 256)  |
| **LSH**    | LSH por bandas sobre bits SimHash              | L tablas de r bits; solo se re-ordenan los candidatos que colisionan | O(d·L·r + \|C\|·d) | L·n IDs + tablas    |
| **PCA**    | PCA aleatorizada + k-NN (referencia)           | Reducción dependiente de los datos; responde "¿por qué no PCA?"   | O(d·m + n·m)        | ≈ RP                |
| **VOTO**   | Clasificación por voto de los k vecinos       | Mayoría simple (variante ponderada por similitud)                   | O(k)                  | —                   |
| E2LSH *(opc.)* | LSH p-estable (Datar et al., 2004)             | h(v) = ⌊(a·v + β)/w⌋                                             | O(d·r·L + \|C\|·d) | L·n IDs             |
| MBK *(opc.)*   | Mini-batch k-means sobre proyecciones o firmas | Clustering de una pasada; ARI contra las etiquetas                   | —                    | O(K·m)              |

RP, SH y LSH comparten **una misma matriz aleatoria Ω (d × 1.024)** por semilla, de modo que los métodos quedan anidados y sus diferencias no se deben a aleatoriedad distinta. RP, SH y LSH se implementan a mano; scikit-learn se usa solo para métricas, PCA, MBK y como referencia en los tests. No se evalúan bibliotecas de ANN (FAISS, Annoy).

**Descartados:** MinHash (diseñado para Jaccard entre conjuntos; los perfiles son vectores continuos) y árboles k-d / ball tree (degeneran a fuerza bruta en alta dimensión).

```
Matriz de expresión (n × 20.500)
        │  recorte a 0 → log2(x+1) → filtro de genes → z-score por gen → L2 por muestra
        ▼
 ┌──────────────┬──────────────────┬────────────────────┬───────────────────────────┐
 │ EX           │ RP               │ SH                 │ LSH                       │
 │ coseno en d  │ x·Ω[:, :m]/√m    │ signo(x·Ω[:, :b])  │ bandas sobre signo(x·Ω)   │
 │              │ + k-NN en m      │ + Hamming lineal   │ → candidatos → re-ranking │
 └──────────────┴──────────────────┴────────────────────┴───────────────────────────┘
        ▼
 Top-k vecinos → recall@k vs EX · voto de etiquetas → tipo de cáncer
```

---

## 3. Datos

Fuente: **TCGA PanCanAtlas**, NCI Genomic Data Commons — [https://gdc.cancer.gov/about-data/publications/pancanatlas](https://gdc.cancer.gov/about-data/publications/pancanatlas)

| #  | Archivo                                                       | Uso                                                                                                             |
| -- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| A1 | `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` | Matriz de expresión RNA-Seq normalizada y ajustada por lote (≈ 20.500 genes × ≈ 11.000 muestras, ≈ 1,8 GB) |
| A2 | `TCGA-CDR-SupplementalTableS1.xlsx`                         | Etiqueta de tipo de cáncer por paciente (TCGA Clinical Data Resource)                                          |
| A3 | `merged_sample_quality_annotations.tsv`                     | Filtro de calidad de muestras                                                                                   |

No se usan metilación, miRNA, RPPA, número de copias, mutaciones ni PARADIGM.

**Reglas principales**

- Cruce por **barcode de paciente** (primeros 12 caracteres, `TCGA-XX-XXXX`).
- **Una muestra por paciente:** tipo 01 (tumor primario); 03 en LAML; 06 (o 01) en SKCM. Las muestras normales (11) se excluyen.
- Los estadísticos del preprocesamiento se ajustan **solo con la base** (regla anti-fuga).
- Particiones por paciente y estratificadas por tipo: **Q_test = 500** consultas (reporte), **Q_val = 250** (calibración) y bases anidadas **B_n**, n ∈ {250, 500, 1.000, 2.000, 4.000, 8.000, n_max}, con `split_seed = 2026`.
- Verdad de terreno única: k-NN exacto por coseno con k_max = 50.
- Tipos pequeños (p. ej. CHOL ≈ 36, DLBC ≈ 48 muestras en cBioPortal) se mantienen y se reporta además un escenario sin tipos de menos de 50 muestras.

**Descarga.** Los datos **no se versionan en git**. Se descargan con la [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) (v2.3) a partir del manifiesto *Open Access* de la página PanCanAtlas (`pancan_manifest.txt`, columnas `id`, `filename`, `md5`, `size`). Cada archivo queda en una carpeta con su UUID:

```bash
gdc-client download -m pancan_manifest.txt
```

El manifiesto lista 14 archivos, de los cuales el proyecto solo necesita estos tres (para no bajar las ómicas que no se usan, se puede usar un manifiesto con solo estas filas):

| #  | UUID GDC                                 | MD5                                  | Tamaño         |
| -- | ---------------------------------------- | ------------------------------------ | --------------- |
| A1 | `3586c0da-64d0-4b74-a449-5ff4d9136611` | `02e72c33071307ff6570621480d3c90b` | 1.882.540.959 B |
| A2 | `1b5f413e-a8d1-4d10-92eb-7c4ae739ed81` | `a4591b2dcee39591f59e5e25a6ce75fa` | 2.945.129 B     |
| A3 | `1a7d7be8-675d-4e60-a105-19d4121bdebf` | `05ddd2270fb1fb24fbdc2fe9bf7384e5` | 8.463.670 B     |

**Verificado (22/10/2026)** tras la descarga: los MD5 de A1–A3 coinciden con el manifiesto.

- **A1:** 20.531 genes × 11.069 muestras (genes en filas). La primera columna es `gene_id` con formato `"SYMBOL|EntrezID"` entre comillas; 29 genes no tienen símbolo (`?|EntrezID`). Los encabezados son barcodes de alícuota (`TCGA-OR-A5J1-01A-11R-A29S-07`).
- **A3:** columnas `patient_barcode`, `aliquot_barcode`, `cancer type`, `platform`, `AWG_excluded_because_of_pathology`, `Do_not_use`, entre otras; 10.439 filas con `platform = IlluminaHiSeq_RNASeqV2` (y 646 de `IlluminaGA_RNASeqV2`).

Los archivos van en `data/raw/` y su hash en `data/CHECKSUMS.sha256` (sí versionado). Al resto del grupo se le distribuyen los `.npy` procesados por Drive o pendrive; para equipos con poca conexión existe un subconjunto de desarrollo `X_dev_f32.npy` (≈ 1.250 pacientes, ≈ 103 MB).

---

## 4. Experimentos y métricas

| Exp.               | Pregunta                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| E0                 | ¿Qué tan difícil es el problema? (distribución de cosenos, pureza de vecinos exactos)                           |
| E1a / E1b          | Costo de EX vs n · ¿ayuda filtrar genes (top-G)?                                                                  |
| E2                 | ¿Se cumple JL? Distorsión de distancias vs m, contra la cota teórica                                             |
| E3                 | RP para k-NN (y PCA de referencia)                                                                                  |
| E4                 | SimHash con búsqueda lineal en Hamming: error angular, recall vs b                                                 |
| E5                 | Calibración de LSH: grilla L × r, recall vs fracción de candidatos                                               |
| **E6**       | **Escalabilidad:** tiempo de consulta y construcción vs n; punto de cruce n*                                 |
| **E7**       | **Clasificación:** exactitud, F1-macro, exactitud balanceada, acuerdo de voto, matriz de confusión 33 × 33 |
| E8 / E9 *(opc.)* | E2LSH vs SH · subtipos PAM50 en mama y clustering (ARI)                                                            |

**Métricas:** recall@k frente a EX, candidatos examinados, tiempo por consulta (mediana y p95), tiempo de construcción, memoria del índice (teórica y medida), exactitud, F1-macro, exactitud balanceada y acuerdo de voto. Los tiempos se miden en una misma máquina, con 1 hilo de BLAS, una pasada de calentamiento y 5 pasadas medidas; los barridos usan 5 semillas (10 en las configuraciones finales).

**Criterios de éxito** (se congelan el 08/11 en `experiments/LOG.md`): recall@10 ≥ 0,90; fracción de candidatos ≤ 20 % (LSH); Δexactitud ≤ 2 pp y ΔF1-macro ≤ 3 pp respecto de EX; ≥ 10× menos memoria por paciente; speedup > 1 con n_max.

---

## 5. Estructura del repositorio (planificada)

```
TopicosGVdD/
├── README.md
├── pyproject.toml             # paquete tcgaknn (pip install -e .)
├── environment.yml            # entorno conda
├── configs/                   # un YAML por experimento (base, avance_1000, sweep_*, scaling_n, classification)
├── data/
│   ├── raw/                   # A1–A3 (no versionado)
│   ├── processed/             # .npy, samples.csv, splits/, gt/ (no versionado)
│   ├── CHECKSUMS.sha256       # versionado
│   └── README.md              # origen, URL y fecha de descarga
├── src/tcgaknn/
│   ├── io.py  preprocess.py  splits.py
│   ├── exact.py  projection.py  simhash.py  lsh.py  pca_ref.py  e2lsh.py
│   ├── classify.py  metrics.py  timing.py  runner.py
│   └── c/hamming.c            # opcional (popcount en C)
├── scripts/                   # download_data, preprocess, ground_truth, describe_data,
│                              # run_experiment, plot_*, make_tables
├── experiments/LOG.md         # bitácora de corridas y decisiones
├── results/                   # raw/ (CSV), figures/, tables/
├── report/                    # informe y diapositivas
└── tests/                     # pytest con datos sintéticos
```

Todos los índices exponen la misma interfaz: `fit(X)` y `query(Q, k) -> (ids, scores, stats)`. **Regla de oro:** toda cifra o figura del informe sale de `scripts/` a partir de un CSV en `results/raw/`; los notebooks son solo para explorar.

---

## 6. Instalación y reproducción

> Los comandos describen el flujo previsto; se irán habilitando a medida que se suban los módulos.

```bash
conda env create -f environment.yml && conda activate tcgaknn
pip install -e .
python -c "import numpy as np; assert hasattr(np, 'bitwise_count')"   # requiere NumPy >= 2.0

# Datos
python scripts/download_data.py --verify-only
python scripts/preprocess.py metadata  configs/base.yaml
python scripts/preprocess.py matrix    configs/base.yaml
python scripts/preprocess.py splits    configs/base.yaml
python scripts/preprocess.py transform configs/base.yaml --G all
python scripts/ground_truth.py configs/base.yaml --G all
python scripts/describe_data.py configs/base.yaml

# Tests y experimentos
pytest -q
python scripts/run_experiment.py configs/avance_1000.yaml
python scripts/run_experiment.py configs/scaling_n.yaml
python scripts/run_experiment.py configs/classification.yaml

# Figuras y tablas
python scripts/plot_recall_vs_time.py results/raw/*.csv
python scripts/plot_scaling.py results/raw/scaling_n.csv
python scripts/make_tables.py results/raw/
```

En Windows se ejecutan los mismos comandos `python` en el mismo orden. Para compilar el núcleo opcional en C se recomienda WSL/Linux con gcc.

### Inventario de hardware

Los tiempos que se comparan salen siempre de la **máquina de referencia**.

| Integrante      | CPU | Núcleos / hilos | RAM | SO | Backend BLAS | Disco libre |
| --------------- | --- | ---------------- | --- | -- | ------------ | ----------- |
| Antonia Montero |     |                  |     |    |              |             |
| Diego Pérez    |     |                  |     |    |              |             |
| Angie Ramírez  |     |                  |     |    |              |             |

Máquina de referencia: (por definir)

---

## Referencias

1. Indyk, P., y Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *STOC '98*, 604–613. https://doi.org/10.1145/276698.276876
2. Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. *STOC '02*, 380–388. https://doi.org/10.1145/509907.509965
3. Datar, M., Immorlica, N., Indyk, P., y Mirrokni, V. S. (2004). Locality-sensitive hashing scheme based on p-stable distributions. *SCG '04*, 253–262. https://doi.org/10.1145/997817.997857
4. Slaney, M., y Casey, M. (2008). Locality-sensitive hashing for finding nearest neighbors [Lecture notes]. *IEEE Signal Processing Magazine*, 25(2), 128–131. https://doi.org/10.1109/MSP.2007.914237
5. Leskovec, J., Rajaraman, A., y Ullman, J. D. (2014). *Mining of Massive Datasets*.
6. Xie, H., Li, J., Zhang, Q., y Wang, Y. (2016). Comparison among dimensionality reduction techniques based on random projection for cancer classification. *Computational Biology and Chemistry*, 65, 165–172. https://doi.org/10.1016/j.compbiolchem.2016.09.010
7. Nguyen, H., et al. (2021). SMRT: Randomized data transformation for cancer subtyping and big data analysis. *Frontiers in Oncology*, 11, 725133. https://doi.org/10.3389/fonc.2021.725133
8. Wu, X., Wang, J., y Wan, S. (2026). RPSLearner: A novel approach based on random projection and deep stacking learning for categorizing non-small cell lung cancer. *Advanced Intelligent Systems*, 8(2), e202500635. https://doi.org/10.1002/aisy.202500635
9. Hoadley, K. A., et al. (2018). Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33 types of cancer. *Cell*, 173(2), 291–304. https://doi.org/10.1016/j.cell.2018.03.022
10. Weinstein, J. N., y The Cancer Genome Atlas Research Network. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. *Nature Genetics*, 45(10), 1113–1120. https://doi.org/10.1038/ng.2764

**Recursos de datos**

- NCI Genomic Data Commons. *PanCanAtlas Publications*. https://gdc.cancer.gov/about-data/publications/pancanatlas
- cBioPortal for Cancer Genomics. *Datasets*. https://www.cbioportal.org/datasets
