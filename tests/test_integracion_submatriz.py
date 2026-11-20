"""Pruebas de integración con submatriz preparada real.

Estas pruebas NO dependen de TCGA descargado para el pytest habitual.
Para activarlas explícitamente:
    TCGA_PROCESSED_DIR=data/submatriz/processed pytest -q tests/test_integracion_submatriz.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from tcgaknn.preparacion import load_prepared


@pytest.fixture(scope="module")
def real_data():
    directory = os.environ.get("TCGA_PROCESSED_DIR")
    if not directory:
        pytest.skip("Definir TCGA_PROCESSED_DIR para probar la submatriz REAL")
    path = Path(directory)
    if not path.is_dir():
        pytest.fail(f"TCGA_PROCESSED_DIR no existe: {path}")
    X, splits, meta = load_prepared(path, verify_hash=True)
    return path, X, splits, meta


def test_formato_etiquetas_y_particiones(real_data):
    path, X, splits, meta = real_data
    with (path / "pacientes.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["patient_id"] for r in rows]
    assert len(ids) == len(set(ids)), "Un paciente fue indexado más de una vez"
    assert len(ids) == X.shape[0] == meta["cohort_total"]
    assert X.shape[1] == meta["genes_used"]
    assert len(splits["labels"]) == len(ids)
    assert [r["type"] for r in rows] == splits["labels"].tolist()
    sets = [set(splits[k].tolist()) for k in ("train", "val", "test")]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1, 3))
    assert set.union(*sets) == set(range(len(rows)))
    for i, part in enumerate(("train", "val", "test")):
        assert set(splits[part].tolist()) == {int(r["row"]) for r in rows if r["split"] == part}
    assert X.shape[1] >= 2


def test_cohorte_filtros_y_valores_finitos(real_data):
    path, X, splits, meta = real_data
    assert np.isfinite(np.asarray(X)).all(), "La matriz tiene NaN o Inf"
    with (path / "auditoria_muestras.csv").open(encoding="utf-8", newline="") as f:
        audit = list(csv.DictReader(f))
    valid = [row for row in audit if row["included"].lower() == "true"]
    assert len(valid) == meta["cohort_total"]
    assert set(row["sample_id"] for row in valid) == set(splits["sample_ids"].tolist())
    assert len(set(row["patient_id"] for row in valid)) == len(valid)
    assert all(row["reason"] == "incluida" for row in valid)


def test_estadisticas_e_invariantes(real_data):
    path, X, splits, meta = real_data
    tr = splits["train"]
    assert len(tr) >= 3
    assert np.all(np.isfinite(splits["gene_mean_train"]))
    assert np.all(splits["gene_std_train"] > 0)
    if meta["zscore"]:
        # Con imputación de media TRAIN, cada gen de entrenamiento debe estar centrado.
        mean = np.asarray(X[tr]).mean(axis=0)
        np.testing.assert_allclose(mean, 0.0, atol=5e-4)
