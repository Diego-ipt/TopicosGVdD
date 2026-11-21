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


def stratified_nested_order(train: np.ndarray, labels: np.ndarray, *, seed: int = 2026) -> np.ndarray:
    """Permutación anidada: representación inicial de clases y luego cuotas.

    Se selecciona primero una muestra de cada clase, cuando hay suficientes
    pacientes. Posteriormente se añade la clase con mayor déficit respecto a
    su frecuencia dentro de train. Nunca se utilizan val/test. Las bases B_n
    siguen siendo prefijos de la MISMA permutación, sin re-muestreo por n.
    """
    train = np.asarray(train, dtype=np.int64)
    labels = np.asarray(labels)
    if train.ndim != 1 or len(np.unique(train)) != len(train):
        raise ValueError("train debe contener índices únicos en una dimensión")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[train])
    groups = []
    for label in classes:
        indices = train[labels[train] == label].copy()
        rng.shuffle(indices)
        groups.append(indices)
    # Desempate reproducible, independiente del orden alfabético de las clases.
    priority = rng.permutation(len(groups))
    counts = np.zeros(len(groups), dtype=np.int64)
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    result = []
    for cls in priority:
        result.append(int(groups[cls][0]))
        counts[cls] += 1
    while len(result) < len(train):
        t = len(result) + 1
        deficits = t * sizes / len(train) - counts
        deficits[counts >= sizes] = -np.inf
        cls = int(priority[np.argmax(deficits[priority])])
        result.append(int(groups[cls][counts[cls]]))
        counts[cls] += 1
    return np.asarray(result, dtype=np.int64)


def assert_disjoint(split: Split, n: int) -> None:
    a, b, c = (set(getattr(split, part).tolist()) for part in ("train", "val", "test"))
    if a & b or a & c or b & c or (a | b | c) != set(range(n)):
        raise AssertionError("Particiones superpuestas o incompletas")
