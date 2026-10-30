from __future__ import annotations

import numpy as np
import pytest

from tcgaknn.algoritmos.base import (
    StreamingProjectionAccumulator,
    default_feature_ids,
    l2_normalize,
    omega_matrix,
    omega_row,
    topk,
)


def test_topk_ties_and_padding() -> None:
    s = np.array([[0.5, 0.9, 0.9, 0.1, 0.9]])
    idx, val = topk(s, 2)
    assert idx.tolist() == [[1, 2]]  # empate en 0.9: gana el índice menor, también en la frontera
    idx, val = topk(s, 7)
    assert idx[0, 5:].tolist() == [-1, -1] and np.isnan(val[0, 5:]).all()
    idx, _ = topk(np.array([[3.0, np.nan, 1.0]]), 2, largest=False)
    assert idx.tolist() == [[2, 0]]  # NaN es el peor


def test_topk_matches_stable_argsort() -> None:
    rng = np.random.default_rng(0)
    S = rng.integers(0, 5, size=(20, 50)).astype(np.float32)  # muchos empates
    idx, _ = topk(S, 7)
    ref = np.argsort(-S, axis=1, kind="stable")[:, :7]
    np.testing.assert_array_equal(idx, ref)


def test_l2_normalize_zero_row() -> None:
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    N = l2_normalize(X)
    np.testing.assert_allclose(N[0], [0.6, 0.8], rtol=1e-6)
    assert (N[1] == 0).all()


def test_omega_deterministic_by_id() -> None:
    a = omega_row("TP53|7157", 64, seed=1)
    np.testing.assert_array_equal(a, omega_row("TP53|7157", 64, seed=1))
    assert not np.array_equal(a, omega_row("TP53|7157", 64, seed=2))
    assert not np.array_equal(a, omega_row("BRCA1|672", 64, seed=1))
    O = omega_matrix(["a", "b"], 8, seed=3)
    np.testing.assert_array_equal(O[1], omega_row("b", 8, seed=3))


@pytest.mark.parametrize("batch_size", [1, 7, 256])
def test_streaming_accumulator_equals_in_memory(batch_size: int) -> None:
    rng = np.random.default_rng(1)
    n, d, cols = 40, 300, 64
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + rng.normal(size=d)
    X[rng.random((n, d)) < 0.02] = np.nan
    X[:, 5] = 2.0  # rasgo constante: se descarta
    fit_mask = np.arange(n) < 30
    ids = default_feature_ids(d)

    acc = StreamingProjectionAccumulator(n, cols, seed=7, fit_mask=fit_mask, batch_size=batch_size)
    for j in range(d):
        acc.update(ids[j], X[:, j])
    assert acc.n_skipped == 1 and acc.n_features == d - 1

    # Versión en memoria con la misma regla
    ref = X[fit_mask]
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    keep = sd > 1e-12
    Z = np.nan_to_num((X[:, keep] - mu[keep]) / sd[keep])
    O = omega_matrix([i for i, k in zip(ids, keep) if k], cols, seed=7)
    P = l2_normalize(Z) @ O[:, :32] / np.sqrt(32)
    np.testing.assert_allclose(acc.projections(32), P, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(acc.sign_bits(48), (Z @ O[:, :48]) >= 0)
