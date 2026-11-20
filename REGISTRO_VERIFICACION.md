# Registro de verificación del código corregido

**Fecha:** 2026-11-19. **Fuente auditada:** ZIP `TopicosGVdD-main.zip` subido por el usuario.

## Pruebas automáticas

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q
```

**Última suite completa realizada:** 242 pruebas aprobadas, 14 omitidas, 0 fallidas, 6 advertencias. Las omitidas son pruebas de integración con datos originales ubicados en una ruta externa; las advertencias proceden de pruebas sintéticas previas que calculan `nanmean/nanstd` sobre slices vacíos.

**Tests nuevos (`tests/test_protocolo_validado.py`, 8):** preparación A1+A2+A3 sintética, cruces de alícuota, descarte por A3, división por paciente sin solapamientos, flujo validación→configuraciones congeladas→test, rechazo de `dataset_id` cambiado, valores de validación no influyen en estadísticos de training, plataforma GA no se usa para A1 HiSeq, L2 contra `scipy.cdist` y contabilidad de memoria PCA, guardas por métrica, y columna de abstenciones en matriz de confusión.

## Ejecuciones integradas con datos artificiales

```bash
python scripts/generar_demo.py --out demo/raw
PYTHONPATH=src python scripts/preparar_cohorte.py demo/raw --out demo/processed
PYTHONPATH=src python scripts/experimentar.py validar --data demo/processed --out demo/results --methods rp simhash simhash_rp lsh --k 3
PYTHONPATH=src python scripts/experimentar.py test --data demo/processed --out demo/results
```

**Resultado:** PASS, las cinco familias (exact/RP/SimHash/SimHash+RP/LSH) generaron reportes de validación y de test.

**Validaciones adicionales:** PASS en datos artificiales para PCA + E2LSH p2 aproximando coseno; E2LSH p1 aproximando L1; RP + E2LSH p2 aproximando L2. Ningún error de ejecución en esas grillas reducidas. Estas cifras **no** estiman la calidad de recuperación en TCGA real.

## Limitaciones / pendientes antes de autorizar experimentos finales

- **No fueron proporcionados A1, A2 y A3 originales en el ZIP.** El pipeline completo no se ha validado sobre ellos ni se ha comprobado la política de anotaciones/calidad con sus valores reales.
- Los scripts originales `experimento_completo.py` y `experimento_escalabilidad.py` siguen siendo **exploratorios**. Los nuevos scripts son los únicos designados para entrenamiento, validación y test independiente.
- La selección de representaciones/hiperparámetros con las consultas de validación es válida, pero debe elegirse una representación definitiva **antes del test** y registrarla en el informe.
- Las pruebas sintéticas validan comportamiento, no sensibilidad clínica. Resultados por tipo de cáncer con soporte pequeño requieren incertidumbre adecuada y no sustentan certeza de diagnóstico.
- Las mediciones de latencia y memoria del nuevo runner son de una corrida y estimaciones de arreglos residentes, respectivamente. El protocolo del README de 5 semillas y múltiples repeticiones debe completarse antes de afirmar estabilidad de hardware.
