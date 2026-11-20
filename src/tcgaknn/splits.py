"""Particiones fijas por paciente: train/base, validation y test.

Cada paciente aparece una sola vez en la cohorte y exactamente en un conjunto.
El estratificado garantiza al menos una muestra de cada clase por conjunto;
para estimaciones robustas por clase hacen falta muchas más observaciones.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(patient_ids: list[str], labels: np.ndarray, *, seed: int = 2026,
               val_frac: float = 0.15, test_frac: float = 0.15) -> Split:
    labels = np.asarray(labels)
    if len(patient_ids) != len(labels) or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Debe haber exactamente una fila por paciente")
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Proporciones inválidas")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for label in sorted(set(labels.tolist())):
        pos = np.flatnonzero(labels == label)
        n = pos.size
        if n < 3:
            raise ValueError(f"La clase {label} tiene solo {n} pacientes: se requieren al menos 3")
        rng.shuffle(pos)
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        if n_val + n_test >= n:
            n_val = n_test = 1
        parts["val"].extend(pos[:n_val].tolist())
        parts["test"].extend(pos[n_val:n_val + n_test].tolist())
        parts["train"].extend(pos[n_val + n_test:].tolist())
    split = Split(*(np.asarray(sorted(parts[k]), dtype=np.int64) for k in ("train", "val", "test")))
    assert_disjoint(split, len(patient_ids))
    return split


def assert_disjoint(split: Split, n: int) -> None:
    a, b, c = (set(getattr(split, part).tolist()) for part in ("train", "val", "test"))
    if a & b or a & c or b & c or (a | b | c) != set(range(n)):
        raise AssertionError("Particiones superpuestas o incompletas")
