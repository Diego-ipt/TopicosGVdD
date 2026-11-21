

## 2. Metas de calidad fijadas antes de la próxima evaluación real

Se volvieron a fijar los umbrales originales en la única constante
`QUALITY_POLICY` de `experimento_streaming.py`, validada por pruebas:

| Medida | Regla | Tipo |
|---|---|---|
| accuracy | pérdida vs su exacto ≤ 0.02 | estricto |
| F1 macro | pérdida vs su exacto ≤ 0.03 | estricto |
| sensibilidad por clase (soporte de consultas ≥10) | pérdida máxima ≤ 0.05 | estricto cuando aplica |
| recall@k | ≥ 0.90 | estricto |
| cobertura | ≥ 0.90 | estricto |
| fracción candidatos | ≤ 0.50 | eficiencia estricta |

**No se eleva una configuración a “exitosa” cuando incumple las metas.**
Si ninguna alternativa pasa, se registra `qualified=false`. Aun así se
puede congelar una alternativa para *diagnóstico* y se informa el motivo
`selection_reason`. El test verifica que los umbrales del archivo de
configuración congelada coincidan con los del protocolo.

Estos criterios deben ser fijados ANTES de la nueva corrida real. **Advertencia
ética/metodológica:** haber inspeccionado un test anterior ya constituye exposición
al test, incluso si restauramos los límites originales. Para una confirmación
estrictamente independiente, se necesitaría un conjunto de test no observado
previamente; si no existe, informar la limitación y no declarar confirmación
independiente. El mínimo de recall 0.90 puede ser poco factible; si nadie
lo logra, así se informa, sin bajarlo retroactivamente.

## 3. Benchmark pareado con proyección online (nuevo)

El benchmark antiguo `--exact-benchmark-raw` sigue existiendo, pero su costo
por consulta **amortizado** sobre todo val/test no debe dividirse por la
latencia de consultas ANN sobre sketches precalculados.

El nuevo **`scripts/benchmark_pareado.py`**:

1. Comprueba cohorte A1∩A2∩A3, split y parámetros de train.
2. Relee A1 **una vez por ejecución**, independientemente de n, métricas y
   métodos, y crea una matriz temporal memmap únicamente para este benchmark.
3. Para cada n y ambas métricas mide `ExactKNN.query` de **una sola consulta**,
   contra una base densa precargada en RAM.
4. Con las mismas consultas mide `proyección z @ Ω/A + búsqueda LSH/E2LSH +
   reranking`, usando los índices/configuraciones congelados y una matriz de
   proyección precargada. Así incorpora el costo de proyectar al paciente.
5. Comprueba además el acuerdo del exacto denso con el oráculo streaming.
6. Genera un CSV independiente `pareado_test.csv` con medianas por fase, n,
   método, réplica, `n_queries`, memoria de base/índice/pesos y
   `speedup_vs_dense_exact`.

**Escenario comparado:** ambos métodos parten de un vector de genes **ya
transformado y estandarizado** de la misma consulta, con la base e índice
precargados en RAM. **NO** incluye E/S, parsing de A1, log2/estandarización
del nuevo paciente, ni entrenamiento/construcción de índices, ni generación
inicial de pesos aleatorios. Por eso es una comparación razonable de
**tiempo de búsqueda online** en ese escenario, **no** un speedup end-to-end
desde un archivo crudo. La consulta exacta tampoco recupera datos de disco.
El entrenamiento principal de LSH y E2LSH sigue siendo streaming; la matriz
n×genes solo existe temporalmente en el benchmark opcional.

```bash
# Una corrida de prueba sin buscar en todo test:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python scripts/benchmark_pareado.py --data data/processed --raw data/raw \
  --selection results/escalabilidad/configuraciones_congeladas.json \
  --out results/escalabilidad/pareado_test.csv \
  --phase test --queries 32 --max-dense-mib 2048

# Para reporte final aumentar --queries a 0 (todas) si el costo lo permite.
python scripts/graficar_escalabilidad.py \
  --csv results/escalabilidad/test_final.csv \
  --pareado results/escalabilidad/pareado_test.csv
```

**Costos y límites**: El benchmark denso requiere más RAM, pero limita la
estimación con `--max-dense-mib` y guarda el Z temporal en un archivo que se
elimina al terminar. El costo de I/O y reconstrucción se informa por separado
del tiempo de consulta; no hay nueva lectura de A1 por n/métrica/fase en
una sola ejecución. Reducir tamaño de base o aumentar memoria si no cabe.
Para comparaciones reproducibles, fijar número de consultas, semilla,
hardware y número de hilos BLAS/numba; no extrapolar desde demo a TCGA.

## Verificación

- Nueva prueba explícita de `n` base en CSV y `n_queries` por separado.
- Test de umbrales originales y recall ≥ 0.90.
- Benchmark pareado integrado en demo, misma n y mismas consultas, comparación
  con oráculo streaming y rechazo de umbrales alterados.
- No se ha ejecutado todavía un benchmark completo sobre TCGA real.
