# Perfil de los datos y aplicabilidad de las medidas de similitud

Generado por `scripts/perfil_datos.py` (módulo `tcgaknn.perfil`). Cada sección resume `results/perfil/<tipo>.json`.

- **Estadísticos por rasgo**: exactos sobre las filas leídas del stream (todas, salvo que se indique límite).
- **Comparación de medidas**: sobre un *sketch* S × F (S muestras estratificadas por tipo de cáncer, una por paciente, códigos 01/03/06; F rasgos por reservoir sampling). Pureza@k = fracción de los k vecinos (leave-one-out) con el mismo tipo; voto@k = exactitud del voto por mayoría.
- **Contraste relativo** cercano a 0 = distancias concentradas (vecinos poco distinguibles). **Hubness** (asimetría de N_10) > 1 = unos pocos puntos aparecen en muchas listas de vecinos.
- **Referencia** = coseno sobre la última variante (transformado + z-score + L2), la verdad de terreno del plan (§2.5).
- Jaccard y Hamming usan la binarización indicada por variante.

Pendiente: número de copias (`copy_number`, `absolute_segments`) no se perfila: requiere llevar segmentos a genes.

## expression

- Archivo: `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` (feature); filas leídas: 20.531 (archivo completo; truncado = False); columnas: 11.069; lectura 522 s, análisis 61 s.
- Valores: NA 1,748 %, ceros 12,6 %, negativos 0,560 %, enteros 13,1 %, en [0,1] 17,0 %; rango [-0,99, 9.924.360]; rasgos con NA 4.196, con varianza 0 219.
- Varianza por rasgo (cruda): mediana 130.249, q99/q01 = 14.214.136.190; transformada q99/q01 = 1.489; el 1 % de rasgos con más varianza concentra el 92,8 % de la varianza cruda. Asimetría mediana por rasgo: cruda 5,40, transformada 0,14.
- Norma por muestra (pasada completa, 10.215 muestras elegibles): η² norma cruda vs tipo = 0,42, transformada = 0,45.
- _Hay pocos negativos (0.560 %): artefacto de normalización; recortar a 0 antes de log2(x+1)._
- _Escalas muy heterogéneas entre rasgos (var q99/q01 = 1.42e+10): L2/L1/coseno crudos los dominan unos pocos rasgos; z-score por rasgo recomendado._
- _El 1 % de rasgos de mayor varianza concentra el 93 % de la varianza total cruda._
- _La transformación reduce la asimetría mediana por rasgo de 5.40 a 0.14._
- _NA = 1.748 % de las celdas (4196 rasgos con algún NA): imputar (media del rasgo = 0 tras z-score) o descartar rasgos._
- Sketch: S = 400 muestras, 33 tipos, F = 20000 rasgos; pureza al azar = 0,028; transformación: log2(max(x,0)+1).

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,60 (0,69) | 0,78 (0,82) | 0,75 (0,81) | 0,75 (0,81) |
| L2 | 0,55 (0,66) | 0,78 (0,82) | 0,71 (0,79) | 0,75 (0,81) |
| L1 | 0,69 (0,76) | 0,76 (0,82) | 0,73 (0,81) | 0,75 (0,82) |
| pearson | 0,60 (0,69) | 0,79 (0,82) | 0,75 (0,82) | 0,75 (0,82) |
| spearman | 0,78 (0,82) | 0,78 (0,82) | 0,73 (0,81) | 0,73 (0,81) |
| jaccard | 0,77 (0,82) | 0,77 (0,82) | 0,74 (0,79) | 0,74 (0,79) |
| hamming | 0,77 (0,81) | 0,77 (0,81) | 0,72 (0,79) | 0,72 (0,79) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 6,46 [0,33] | 5,24 [0,29] | 2,08 [0,15] | 2,08 [0,15] |
| L2 | 35,65 [1,06] | 1,47 [0,15] | 1,68 [0,16] | 0,76 [0,08] |
| L1 | 6,54 [0,31] | 1,53 [0,16] | 1,46 [0,16] | 0,94 [0,09] |
| pearson | 6,19 [0,32] | 4,80 [0,27] | 2,07 [0,14] | 2,07 [0,14] |
| spearman | 5,16 [0,28] | 4,57 [0,26] | 2,10 [0,16] | 2,10 [0,16] |
| jaccard | 1,75 [0,19] | 1,75 [0,19] | 0,46 [0,05] | 0,46 [0,05] |
| hamming | 1,96 [0,21] | 1,96 [0,21] | 1,74 [0,17] | 1,74 [0,17] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 1,77 [54] | 1,09 [41] | 0,56 [28] | 0,56 [28] |
| L2 | 3,43 [95] | 0,97 [40] | 3,45 [76] | 0,56 [28] |
| L1 | 2,09 [63] | 0,73 [37] | 1,98 [54] | 0,44 [27] |
| pearson | 1,70 [52] | 0,81 [36] | 0,56 [28] | 0,56 [28] |
| spearman | 0,89 [41] | 1,04 [41] | 0,47 [26] | 0,47 [26] |
| jaccard | 1,31 [49] | 1,31 [49] | 0,65 [32] | 0,65 [32] |
| hamming | 1,08 [37] | 1,08 [37] | 2,24 [51] | 2,24 [51] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,42 | 0,65 | 1,00 | 1,00 |
| L2 | 0,34 | 0,66 | 0,64 | 1,00 |
| L1 | 0,55 | 0,69 | 0,67 | 0,87 |
| pearson | 0,42 | 0,67 | 0,94 | 0,94 |
| spearman | 0,68 | 0,70 | 0,87 | 0,87 |
| jaccard | 0,52 | 0,52 | 0,77 | 0,77 |
| hamming | 0,53 | 0,53 | 0,63 | 0,63 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 20.000 | 25 | 3 | 9,7 | 0,98 | 0,24 | 0,03 | x > 1 (gen expresado: log2(x+1) > 1) (0,82) |
| transformado | 20.000 | 178 | 11 | 16,1 | 0,02 | 0,57 | 0,05 | x > 1 (gen expresado: log2(x+1) > 1) (0,82) |
| transformado+z | 19.738 | 196 | 15 | 17,1 | 0,19 | 0,58 | 0,06 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,14) |
| transformado+z+L2 | 19.738 | 208 | 17 | 17,1 | 0,00 | 0,10 | 0,03 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,14) |

Acuerdo entre medidas en `transformado+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,87, coseno|pearson 0,94, coseno|spearman 0,87, coseno|jaccard 0,77, coseno|hamming 0,63.

## methylation_450k

- Archivo: `jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv` (feature); filas leídas: 25.000 (límite --max-rows 25.000; truncado = True); columnas: 9.664; lectura 552 s, análisis 57 s.
- **Sesgo por límite**: solo las primeras filas en el orden del archivo (de `cg00000029` a `cg01470744`); los estadísticos y el reservoir cubren solo ese prefijo.
- Valores: NA 0,126 %, ceros 0,0 %, negativos 0,000 %, enteros 0,0 %, en [0,1] 100,0 %; rango [0,00, 1,00]; rasgos con NA 7.862, con varianza 0 0.
- Varianza por rasgo (cruda): mediana 0,018, q99/q01 = 2.272; transformada q99/q01 = 64,87; el 1 % de rasgos con más varianza concentra el 3,9 % de la varianza cruda. Asimetría mediana por rasgo: cruda -0,06, transformada -0,08.
- Norma por muestra (pasada completa, 8.819 muestras elegibles): η² norma cruda vs tipo = 0,44, transformada = 0,64.
- _Todos los valores son >= 0: los vectores crudos viven en un ortante y el coseno crudo queda comprimido (ángulos pequeños); centrar (z-score) antes de coseno/SimHash._
- _Valores en [0, 1] (proporciones beta): varianza heterocedástica; M-value o z-score antes de L2/coseno; la binarización beta > 0,5 tiene sentido biológico._
- _Escalas muy heterogéneas entre rasgos (var q99/q01 = 2.27e+03): L2/L1/coseno crudos los dominan unos pocos rasgos; z-score por rasgo recomendado._
- _NA = 0.126 % de las celdas (7862 rasgos con algún NA): imputar (media del rasgo = 0 tras z-score) o descartar rasgos._
- Sketch: S = 400 muestras, 33 tipos, F = 20000 rasgos; pureza al azar = 0,028; transformación: M-value log2(b/(1-b)), b en [0.001, 0.999].

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,62 (0,72) | 0,65 (0,74) | 0,72 (0,79) | 0,72 (0,79) |
| L2 | 0,61 (0,71) | 0,64 (0,72) | 0,61 (0,72) | 0,72 (0,79) |
| L1 | 0,66 (0,72) | 0,67 (0,75) | 0,65 (0,75) | 0,71 (0,77) |
| pearson | 0,64 (0,72) | 0,67 (0,76) | 0,75 (0,80) | 0,75 (0,80) |
| spearman | 0,68 (0,79) | 0,68 (0,79) | 0,74 (0,82) | 0,74 (0,82) |
| jaccard | 0,61 (0,69) | 0,61 (0,69) | 0,65 (0,69) | 0,65 (0,69) |
| hamming | 0,62 (0,71) | 0,62 (0,71) | 0,32 (0,28) | 0,32 (0,28) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 7,22 [0,42] | 10,06 [0,44] | 1,64 [0,15] | 1,64 [0,15] |
| L2 | 2,67 [0,23] | 2,44 [0,21] | 2,75 [0,22] | 0,63 [0,08] |
| L1 | 3,77 [0,26] | 3,09 [0,23] | 3,10 [0,23] | 0,83 [0,10] |
| pearson | 6,78 [0,39] | 6,42 [0,37] | 1,53 [0,13] | 1,53 [0,13] |
| spearman | 5,61 [0,34] | 5,62 [0,34] | 1,61 [0,15] | 1,61 [0,15] |
| jaccard | 5,21 [0,35] | 5,21 [0,35] | 0,32 [0,06] | 0,32 [0,06] |
| hamming | 4,62 [0,33] | 4,62 [0,33] | 3,77 [0,41] | 3,77 [0,41] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 3,00 [98] | 2,41 [75] | 1,21 [41] | 1,21 [41] |
| L2 | 2,82 [89] | 2,52 [73] | 2,59 [93] | 1,21 [41] |
| L1 | 1,86 [58] | 1,78 [60] | 1,79 [57] | 0,56 [36] |
| pearson | 2,78 [81] | 2,50 [79] | 0,75 [37] | 0,75 [37] |
| spearman | 2,75 [85] | 2,74 [85] | 0,61 [31] | 0,61 [31] |
| jaccard | 2,16 [64] | 2,16 [64] | 1,75 [59] | 1,75 [59] |
| hamming | 2,25 [68] | 2,25 [68] | 6,50 [322] | 6,50 [322] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,39 | 0,44 | 1,00 | 1,00 |
| L2 | 0,39 | 0,42 | 0,42 | 1,00 |
| L1 | 0,47 | 0,49 | 0,49 | 0,80 |
| pearson | 0,40 | 0,42 | 0,73 | 0,73 |
| spearman | 0,44 | 0,44 | 0,70 | 0,70 |
| jaccard | 0,40 | 0,40 | 0,45 | 0,45 |
| hamming | 0,42 | 0,42 | 0,19 | 0,19 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 20.000 | 196 | 13 | 17,9 | 0,09 | 0,46 | 0,06 | beta > 0,5 (sonda metilada; equivale a M > 0) (0,48) |
| transformado | 20.000 | 202 | 13 | 17,5 | 0,07 | 0,60 | 0,07 | beta > 0,5 (sonda metilada; equivale a M > 0) (0,48) |
| transformado+z | 20.000 | 220 | 15 | 20,3 | 0,28 | 0,34 | 0,05 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |
| transformado+z+L2 | 20.000 | 252 | 20 | 20,6 | 0,00 | 0,04 | 0,03 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |

Acuerdo entre medidas en `transformado+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,80, coseno|pearson 0,73, coseno|spearman 0,70, coseno|jaccard 0,45, coseno|hamming 0,19.

## methylation_27k_450k

- Archivo: `jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv` (feature); filas leídas: 18.000 (límite --max-rows 18.000; truncado = True); columnas: 12.039; lectura 536 s, análisis 55 s.
- **Sesgo por límite**: solo las primeras filas en el orden del archivo (de `cg00000292` a `cg21949305`); los estadísticos y el reservoir cubren solo ese prefijo.
- Valores: NA 0,076 %, ceros 0,0 %, negativos 0,000 %, enteros 0,0 %, en [0,1] 100,0 %; rango [0,00, 1,00]; rasgos con NA 8.620, con varianza 0 0.
- Varianza por rasgo (cruda): mediana 0,008, q99/q01 = 1.687; transformada q99/q01 = 54,26; el 1 % de rasgos con más varianza concentra el 4,7 % de la varianza cruda. Asimetría mediana por rasgo: cruda 2,83, transformada 0,77.
- Norma por muestra (pasada completa, 10.728 muestras elegibles): η² norma cruda vs tipo = 0,37, transformada = 0,53.
- _Todos los valores son >= 0: los vectores crudos viven en un ortante y el coseno crudo queda comprimido (ángulos pequeños); centrar (z-score) antes de coseno/SimHash._
- _Valores en [0, 1] (proporciones beta): varianza heterocedástica; M-value o z-score antes de L2/coseno; la binarización beta > 0,5 tiene sentido biológico._
- _Escalas muy heterogéneas entre rasgos (var q99/q01 = 1.69e+03): L2/L1/coseno crudos los dominan unos pocos rasgos; z-score por rasgo recomendado._
- _La transformación reduce la asimetría mediana por rasgo de 2.83 a 0.77._
- _NA = 0.076 % de las celdas (8620 rasgos con algún NA): imputar (media del rasgo = 0 tras z-score) o descartar rasgos._
- Sketch: S = 400 muestras, 33 tipos, F = 18000 rasgos; pureza al azar = 0,028; transformación: M-value log2(b/(1-b)), b en [0.001, 0.999].

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,59 (0,69) | 0,60 (0,71) | 0,63 (0,66) | 0,63 (0,66) |
| L2 | 0,56 (0,67) | 0,60 (0,69) | 0,54 (0,66) | 0,63 (0,66) |
| L1 | 0,64 (0,72) | 0,62 (0,73) | 0,57 (0,71) | 0,63 (0,68) |
| pearson | 0,61 (0,70) | 0,62 (0,70) | 0,69 (0,72) | 0,69 (0,72) |
| spearman | 0,58 (0,68) | 0,58 (0,67) | 0,67 (0,72) | 0,67 (0,72) |
| jaccard | 0,59 (0,68) | 0,59 (0,68) | 0,59 (0,64) | 0,59 (0,64) |
| hamming | 0,58 (0,67) | 0,58 (0,67) | 0,31 (0,31) | 0,31 (0,31) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 4,98 [0,40] | 4,77 [0,40] | 1,49 [0,15] | 1,49 [0,15] |
| L2 | 1,52 [0,22] | 1,42 [0,20] | 1,59 [0,20] | 0,58 [0,08] |
| L1 | 1,91 [0,23] | 1,60 [0,20] | 1,93 [0,22] | 0,81 [0,10] |
| pearson | 4,92 [0,39] | 4,23 [0,36] | 1,33 [0,13] | 1,33 [0,13] |
| spearman | 3,31 [0,29] | 3,30 [0,29] | 1,41 [0,14] | 1,41 [0,14] |
| jaccard | 2,84 [0,27] | 2,84 [0,27] | 0,33 [0,06] | 0,33 [0,06] |
| hamming | 2,87 [0,31] | 2,87 [0,31] | 4,25 [0,41] | 4,25 [0,41] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 3,35 [97] | 2,48 [73] | 2,15 [49] | 2,15 [49] |
| L2 | 3,18 [86] | 2,77 [81] | 3,71 [118] | 2,15 [49] |
| L1 | 1,90 [51] | 1,89 [58] | 2,41 [65] | 1,09 [35] |
| pearson | 3,22 [91] | 3,00 [90] | 2,88 [54] | 2,88 [54] |
| spearman | 2,31 [74] | 2,31 [74] | 2,51 [55] | 2,51 [55] |
| jaccard | 3,62 [93] | 3,62 [93] | 2,07 [53] | 2,07 [53] |
| hamming | 3,54 [98] | 3,54 [98] | 6,60 [312] | 6,60 [312] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,30 | 0,34 | 1,00 | 1,00 |
| L2 | 0,30 | 0,33 | 0,35 | 1,00 |
| L1 | 0,38 | 0,41 | 0,40 | 0,74 |
| pearson | 0,31 | 0,33 | 0,67 | 0,67 |
| spearman | 0,32 | 0,32 | 0,62 | 0,62 |
| jaccard | 0,31 | 0,31 | 0,46 | 0,46 |
| hamming | 0,31 | 0,31 | 0,22 | 0,22 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 18.000 | 194 | 12 | 18,1 | 0,10 | 0,43 | 0,06 | beta > 0,5 (sonda metilada; equivale a M > 0) (0,23) |
| transformado | 18.000 | 204 | 12 | 20,1 | 0,06 | 0,49 | 0,06 | beta > 0,5 (sonda metilada; equivale a M > 0) (0,23) |
| transformado+z | 18.000 | 236 | 17 | 23,2 | 0,25 | 0,42 | 0,05 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |
| transformado+z+L2 | 18.000 | 260 | 23 | 23,7 | 0,00 | 0,07 | 0,02 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |

Acuerdo entre medidas en `transformado+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,74, coseno|pearson 0,67, coseno|spearman 0,62, coseno|jaccard 0,46, coseno|hamming 0,22.

## mirna

- Archivo: `pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv` (feature); filas leídas: 743 (archivo completo; truncado = False); columnas: 10.824; lectura 19 s, análisis 6 s.
- Valores: NA 0,000 %, ceros 21,6 %, negativos 1,507 %, enteros 21,6 %, en [0,1] 38,2 %; rango [-0,79, 8.418.657]; rasgos con NA 0, con varianza 0 0.
- Varianza por rasgo (cruda): mediana 621,171, q99/q01 = 5.377.216.394; transformada q99/q01 = 95,10; el 1 % de rasgos con más varianza concentra el 86,5 % de la varianza cruda. Asimetría mediana por rasgo: cruda 10,79, transformada 0,89.
- Norma por muestra (pasada completa, 10.072 muestras elegibles): η² norma cruda vs tipo = 0,44, transformada = 0,48.
- _Datos con signo (1.5 % negativos): ya centrados o en escala log-ratio; coseno/correlación aplican sin transformar; Jaccard requiere elegir un umbral._
- _Escalas muy heterogéneas entre rasgos (var q99/q01 = 5.38e+09): L2/L1/coseno crudos los dominan unos pocos rasgos; z-score por rasgo recomendado._
- _El 1 % de rasgos de mayor varianza concentra el 86 % de la varianza total cruda._
- _La transformación reduce la asimetría mediana por rasgo de 10.79 a 0.89._
- Sketch: S = 400 muestras, 32 tipos, F = 743 rasgos; pureza al azar = 0,029; transformación: log2(max(x,0)+1).

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,50 (0,56) | 0,79 (0,85) | 0,76 (0,81) | 0,76 (0,81) |
| L2 | 0,49 (0,55) | 0,78 (0,86) | 0,72 (0,81) | 0,76 (0,81) |
| L1 | 0,59 (0,68) | 0,77 (0,83) | 0,74 (0,81) | 0,76 (0,81) |
| pearson | 0,50 (0,56) | 0,79 (0,86) | 0,78 (0,84) | 0,78 (0,84) |
| spearman | 0,77 (0,82) | 0,77 (0,82) | 0,77 (0,83) | 0,77 (0,83) |
| jaccard | 0,66 (0,76) | 0,66 (0,76) | 0,69 (0,75) | 0,69 (0,75) |
| hamming | 0,66 (0,79) | 0,66 (0,79) | 0,53 (0,56) | 0,53 (0,56) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 38,90 [0,61] | 8,72 [0,42] | 3,00 [0,19] | 3,00 [0,19] |
| L2 | 15,94 [0,56] | 2,23 [0,23] | 2,42 [0,27] | 1,00 [0,10] |
| L1 | 5,67 [0,34] | 2,42 [0,26] | 2,42 [0,28] | 1,30 [0,13] |
| pearson | 38,84 [0,61] | 8,50 [0,43] | 2,54 [0,16] | 2,54 [0,16] |
| spearman | 5,00 [0,37] | 5,18 [0,39] | 2,33 [0,17] | 2,33 [0,17] |
| jaccard | 1,68 [0,19] | 1,68 [0,19] | 0,56 [0,07] | 0,56 [0,07] |
| hamming | 2,14 [0,24] | 2,14 [0,24] | 4,75 [0,45] | 4,75 [0,45] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,70 [29] | 1,98 [50] | 0,69 [34] | 0,69 [34] |
| L2 | 0,75 [33] | 1,67 [46] | 2,02 [56] | 0,69 [34] |
| L1 | 0,83 [29] | 1,31 [42] | 1,54 [44] | 0,55 [32] |
| pearson | 0,70 [29] | 1,56 [44] | 0,58 [30] | 0,58 [30] |
| spearman | 2,40 [54] | 2,31 [53] | 0,49 [27] | 0,49 [27] |
| jaccard | 1,71 [54] | 1,71 [54] | 0,52 [29] | 0,52 [29] |
| hamming | 1,70 [57] | 1,70 [57] | 7,05 [243] | 7,05 [243] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | transformado | transformado+z | transformado+z+L2 |
|---|---|---|---|---|
| coseno | 0,29 | 0,60 | 1,00 | 1,00 |
| L2 | 0,27 | 0,61 | 0,61 | 1,00 |
| L1 | 0,36 | 0,63 | 0,61 | 0,80 |
| pearson | 0,29 | 0,59 | 0,78 | 0,78 |
| spearman | 0,55 | 0,55 | 0,68 | 0,68 |
| jaccard | 0,47 | 0,47 | 0,56 | 0,56 |
| hamming | 0,47 | 0,47 | 0,36 | 0,36 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 743 | 9 | 2 | 6,3 | 0,44 | 0,60 | 0,07 | x > 1 (miRNA expresado: log2(x+1) > 1) (0,61) |
| transformado | 743 | 70 | 7 | 14,8 | 0,07 | 0,63 | 0,06 | x > 1 (miRNA expresado: log2(x+1) > 1) (0,61) |
| transformado+z | 743 | 99 | 8 | 15,9 | 0,32 | 0,75 | 0,09 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |
| transformado+z+L2 | 743 | 117 | 12 | 15,8 | 0,00 | 0,24 | 0,03 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |

Acuerdo entre medidas en `transformado+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,80, coseno|pearson 0,78, coseno|spearman 0,68, coseno|jaccard 0,56, coseno|hamming 0,36.

## paradigm

- Archivo: `merge_merged_reals.tar.gz` (feature); filas leídas: 19.503 (archivo completo; truncado = False); columnas: 9.829; lectura 393 s, análisis 80 s.
- Valores: NA 0,000 %, ceros 19,2 %, negativos 44,941 %, enteros 19,2 %, en [0,1] 54,4 %; rango [-18,50, 16,58]; rasgos con NA 0, con varianza 0 1.290.
- Varianza por rasgo (cruda): mediana 0,231, q99/q01 = 159.270.679.833.643.814.618.169.325.125.632; transformada q99/q01 = 159.270.679.833.643.814.618.169.325.125.632; el 1 % de rasgos con más varianza concentra el 13,0 % de la varianza cruda. Asimetría mediana por rasgo: cruda -0,18, transformada -0,18.
- Norma por muestra (pasada completa, 9.807 muestras elegibles): η² norma cruda vs tipo = 0,35, transformada = 0,35.
- _Datos con signo (44.9 % negativos): ya centrados o en escala log-ratio; coseno/correlación aplican sin transformar; Jaccard requiere elegir un umbral._
- _Escalas muy heterogéneas entre rasgos (var q99/q01 = 1.59e+32): L2/L1/coseno crudos los dominan unos pocos rasgos; z-score por rasgo recomendado._
- Sketch: S = 400 muestras, 33 tipos, F = 19503 rasgos; pureza al azar = 0,028; transformación: ninguna.

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 0,67 (0,72) | 0,67 (0,74) | 0,67 (0,74) |
| L2 | 0,68 (0,75) | 0,67 (0,74) | 0,67 (0,74) |
| L1 | 0,69 (0,76) | 0,69 (0,74) | 0,68 (0,74) |
| pearson | 0,68 (0,73) | 0,70 (0,75) | 0,70 (0,75) |
| spearman | 0,69 (0,76) | 0,70 (0,76) | 0,70 (0,76) |
| jaccard | 0,66 (0,77) | 0,64 (0,72) | 0,64 (0,72) |
| hamming | 0,68 (0,74) | 0,54 (0,55) | 0,54 (0,55) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 2,10 [0,20] | 1,42 [0,14] | 1,42 [0,14] |
| L2 | 0,75 [0,10] | 0,74 [0,10] | 0,55 [0,07] |
| L1 | 1,22 [0,13] | 1,26 [0,14] | 1,06 [0,12] |
| pearson | 1,82 [0,17] | 1,22 [0,12] | 1,22 [0,12] |
| spearman | 1,59 [0,15] | 1,37 [0,13] | 1,37 [0,13] |
| jaccard | 0,71 [0,10] | 0,52 [0,07] | 0,52 [0,07] |
| hamming | 1,35 [0,15] | 2,09 [0,28] | 2,09 [0,28] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 0,73 [31] | 1,21 [41] | 1,21 [41] |
| L2 | 0,46 [27] | 0,52 [32] | 1,21 [41] |
| L1 | 0,52 [28] | 0,59 [33] | 0,70 [30] |
| pearson | 0,48 [28] | 0,51 [28] | 0,51 [28] |
| spearman | 0,54 [29] | 0,53 [28] | 0,53 [28] |
| jaccard | 2,99 [75] | 2,69 [63] | 2,69 [63] |
| hamming | 0,73 [31] | 5,74 [162] | 5,74 [162] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 0,67 | 1,00 | 1,00 |
| L2 | 0,74 | 0,72 | 1,00 |
| L1 | 0,78 | 0,75 | 0,86 |
| pearson | 0,72 | 0,81 | 0,81 |
| spearman | 0,79 | 0,79 | 0,79 |
| jaccard | 0,65 | 0,63 | 0,63 |
| hamming | 0,73 | 0,40 | 0,40 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 19.503 | 234 | 17 | 22,5 | 0,09 | 0,47 | 0,05 | x > 0 (vía/entidad activada) (0,35) |
| crudo+z | 18.536 | 251 | 30 | 25,9 | 0,09 | 0,38 | 0,06 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,16) |
| crudo+z+L2 | 18.536 | 256 | 30 | 25,9 | 0,00 | 0,09 | 0,03 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,16) |

Acuerdo entre medidas en `crudo+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,86, coseno|pearson 0,81, coseno|spearman 0,79, coseno|jaccard 0,63, coseno|hamming 0,40.

## rppa

- Archivo: `TCGA-RPPA-pancan-clean.txt` (sample); filas leídas: 7.790 (archivo completo; truncado = False); columnas: 7.790; lectura 10 s, análisis 3 s.
- Valores: NA 3,051 %, ceros 0,1 %, negativos 48,633 %, enteros 0,1 %, en [0,1] 32,4 %; rango [-8,47, 14,14]; rasgos con NA 9, con varianza 0 0.
- Varianza por rasgo (cruda): mediana 0,272, q99/q01 = 57,63; transformada q99/q01 = 57,63; el 1 % de rasgos con más varianza concentra el 6,5 % de la varianza cruda. Asimetría mediana por rasgo: cruda 0,70, transformada 0,70.
- Etiqueta del archivo vs CDR en la muestra: {'coincide': 375, 'difiere': 25}.
- _Datos con signo (48.6 % negativos): ya centrados o en escala log-ratio; coseno/correlación aplican sin transformar; Jaccard requiere elegir un umbral._
- _NA = 3.051 % de las celdas (9 rasgos con algún NA): imputar (media del rasgo = 0 tras z-score) o descartar rasgos._
- Sketch: S = 400 muestras, 32 tipos, F = 198 rasgos; pureza al azar = 0,029; transformación: ninguna.

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 0,66 (0,75) | 0,67 (0,76) | 0,67 (0,76) |
| L2 | 0,64 (0,74) | 0,62 (0,74) | 0,67 (0,76) |
| L1 | 0,65 (0,76) | 0,63 (0,75) | 0,68 (0,76) |
| pearson | 0,66 (0,74) | 0,67 (0,74) | 0,67 (0,74) |
| spearman | 0,65 (0,76) | 0,67 (0,75) | 0,67 (0,75) |
| jaccard | 0,51 (0,67) | 0,54 (0,65) | 0,54 (0,65) |
| hamming | 0,51 (0,67) | 0,42 (0,56) | 0,42 (0,56) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 6,03 [0,31] | 2,84 [0,17] | 2,84 [0,17] |
| L2 | 2,54 [0,21] | 2,80 [0,21] | 0,96 [0,09] |
| L1 | 2,96 [0,21] | 3,15 [0,22] | 1,12 [0,10] |
| pearson | 6,17 [0,31] | 2,86 [0,17] | 2,86 [0,17] |
| spearman | 5,85 [0,30] | 2,80 [0,18] | 2,80 [0,18] |
| jaccard | 1,95 [0,18] | 0,54 [0,07] | 0,54 [0,07] |
| hamming | 2,66 [0,21] | 3,15 [0,29] | 3,15 [0,29] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 1,56 [44] | 0,72 [32] | 0,72 [32] |
| L2 | 2,31 [68] | 2,76 [74] | 0,72 [32] |
| L1 | 1,81 [56] | 2,15 [58] | 0,60 [29] |
| pearson | 1,48 [43] | 0,65 [32] | 0,65 [32] |
| spearman | 1,28 [40] | 0,50 [31] | 0,50 [31] |
| jaccard | 1,43 [47] | 0,88 [30] | 0,88 [30] |
| hamming | 1,46 [50] | 7,67 [273] | 7,67 [273] |

**Acuerdo con la referencia (Jaccard de los 10-NN)**

| métrica | crudo | crudo+z | crudo+z+L2 |
|---|---|---|---|
| coseno | 0,51 | 1,00 | 1,00 |
| L2 | 0,48 | 0,53 | 1,00 |
| L1 | 0,54 | 0,54 | 0,75 |
| pearson | 0,51 | 0,90 | 0,90 |
| spearman | 0,51 | 0,74 | 0,74 |
| jaccard | 0,36 | 0,44 | 0,44 |
| hamming | 0,36 | 0,29 | 0,29 |

**Dimensión intrínseca y normas por variante**

| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |
|---|---|---|---|---|---|---|---|---|
| crudo | 198 | 51 | 8 | 12,5 | 0,13 | 0,52 | 0,08 | x > 0 (proteína sobre la mediana normalizada) (0,51) |
| crudo+z | 198 | 73 | 11 | 12,5 | 0,25 | 0,25 | 0,04 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |
| crudo+z+L2 | 198 | 78 | 12 | 11,6 | 0,00 | 0,04 | 0,03 | z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia) (0,13) |

Acuerdo entre medidas en `crudo+z+L2` (Jaccard de 10-NN): coseno|L2 1,00, coseno|L1 0,75, coseno|pearson 0,90, coseno|spearman 0,74, coseno|jaccard 0,44, coseno|hamming 0,29.

## mutations

- Eventos leídos: 3.600.963 (archivo completo); muestras en el MAF: 10.295; claves no contiguas: 0; FILTER = PASS: 80,7 %; lectura 448 s.
- Genes mutados no silenciosos por paciente (10.145 pacientes con etiqueta): mediana 58, q05 8, q95 683, máx 11.211; CV 2,97; sin ninguna: 0,2 %; ≥ 500 genes: 7,1 %; ≥ 1000: 3,1 %; η²(log tamaño ~ tipo) = 0,51.
- Más hipermutados: UCEC 11211, PAAD 8387, UCEC 8284, UCEC 8266, UCEC 7957.
- Sketch: S = 400 pacientes, 33 tipos; pureza al azar = 0,028.

**Pureza@5 de vecinos (voto@10 entre paréntesis)**

| métrica | todas | no_silenciosas | no_silenciosas_top500 |
|---|---|---|---|
| jaccard | 0,16 (0,25) | 0,17 (0,23) | 0,12 (0,18) |
| coseno_binario | 0,13 (0,18) | 0,14 (0,21) | 0,10 (0,14) |
| hamming | 0,03 (0,04) | 0,03 (0,03) | 0,04 (0,05) |

**Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]**

| métrica | todas | no_silenciosas | no_silenciosas_top500 |
|---|---|---|---|
| jaccard | 0,03 [0,01] | 0,04 [0,01] | 0,20 [0,03] |
| coseno_binario | 0,08 [0,02] | 0,09 [0,02] | 0,48 [0,07] |
| hamming | 146,59 [2,23] | 175,30 [2,46] | 54,12 [1,50] |

**Hubness: asimetría de N_10  [máx N_10]**

| métrica | todas | no_silenciosas | no_silenciosas_top500 |
|---|---|---|---|
| jaccard | 2,57 [62] | 2,50 [75] | 2,00 [58] |
| coseno_binario | 7,79 [298] | 7,61 [203] | 4,88 [123] |
| hamming | 6,09 [397] | 6,14 [397] | 6,33 [363] |

| variante | universo de genes | densidad | tamaño mediano | vacíos | genes en 1 sola muestra | ρ(tamaño, N_10) Hamming | ρ(tamaño, N_10) Jaccard |
|---|---|---|---|---|---|---|---|
| todas | 18.550 | 0,0125 | 67 | 0,0 % | 14,2 % | -0,22 | 0,17 |
| no_silenciosas | 16.147 | 0,0095 | 43 | 0,2 % | 22,9 % | -0,18 | 0,18 |
| no_silenciosas_top500 | 500 | 0,0398 | 8 | 5,8 % | 0,0 % | -0,52 | 0,50 |

