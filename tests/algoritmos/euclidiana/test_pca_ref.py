from __future__ import annotations

import numpy as np
import pytest

from tcgaknn.algoritmos.base import KNNIndex, l2_normalize, topk
from tcgaknn.algoritmos.euclidiana.pca_ref import PCAKNN


def _low_rank(n: int, d: int, r: int, noise: float, seed: int) -> np.ndarray:
    """Datos de rango ≈ r: n perfiles en un subespacio de dimensión r más ruido."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r)) * np.linspace(3.0, 1.0, r)
    V = rng.standard_normal((r, d))
    return (U @ V + noise * rng.standard_normal((n, d))).astype(np.float32)


def _exact_cosine(X: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    return topk(l2_normalize(Q) @ l2_normalize(X).T, k)[0]


def _recall(a: np.ndarray, e: np.ndarray) -> float:
    return float(np.mean([len(set(x[x >= 0]) & set(y)) for x, y in zip(a, e)])) / e.shape[1]


@pytest.fixture(scope="module")
def low_rank_data() -> tuple[np.ndarray, np.ndarray]:
    X = _low_rank(600, 2000, r=20, noise=0.05, seed=0)
    return X[:500], X[500:]


def test_interface_and_shapes(low_rank_data) -> None:
    X, Q = low_rank_data
    idx = PCAKNN(m=32, seed=0).fit(X)
    assert isinstance(idx, KNNIndex) and idx.score_kind == "distance"
    I, S, st = idx.query(Q, 10)
    assert I.shape == (100, 10) and I.dtype == np.int64 and S.dtype == np.float32
    assert st["n_candidates"].dtype == np.int64 and (st["n_candidates"] == 500).all()
    assert (np.diff(S, axis=1) >= 0).all()  # distancias crecientes
    assert idx.fit_time_s > 0 and idx.Y_.dtype == np.float32
    p = idx.pca_
    itemsize = p.components_.itemsize  # sklearn conserva float32 si la entrada es float32
    assert st["index_bytes"] == itemsize * (32 * 2000 + 2000) + 4 * 500 * 32


def test_recall_high_on_low_rank(low_rank_data) -> None:
    """Con m ≥ rango, PCA conserva casi toda la geometría: recall@10 alto.

    Con m = 5 < r = 20 se pierde información y el recall cae claramente:
    el test falla si el índice ignora las componentes o el orden de distancia.
    """
    X, Q = low_rank_data
    exact = _exact_cosine(X, Q, 10)
    r_hi = _recall(PCAKNN(m=32, seed=1).fit(X).query(Q, 10)[0], exact)
    r_lo = _recall(PCAKNN(m=5, seed=1).fit(X).query(Q, 10)[0], exact)
    assert r_hi >= 0.95
    assert r_lo < r_hi - 0.2


def test_full_rank_distances_equal_original() -> None:
    """Con m = rango completo, la distancia en ℝ^m es exactamente ‖u − v‖ entre filas unitarias."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 40)).astype(np.float32)
    Q = rng.standard_normal((7, 40)).astype(np.float32)
    idx = PCAKNN(m=40, seed=0, svd_solver="full").fit(X)
    I, S, _ = idx.query(Q, 60)
    Xn, Qn = l2_normalize(X).astype(np.float64), l2_normalize(Q).astype(np.float64)
    D = np.sqrt(((Qn[:, None, :] - Xn[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(S, np.take_along_axis(D, I, axis=1), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(I[:, :10], topk(D, 10, largest=False)[0])


def test_self_query_and_padding(low_rank_data) -> None:
    X, _ = low_rank_data
    idx = PCAKNN(m=32, seed=0).fit(X[:50])
    I, S, _ = idx.query(X[:50], 1)
    np.testing.assert_array_equal(I[:, 0], np.arange(50))
    assert np.all(S[:, 0] < 1e-3)
    I, S, _ = idx.query(X[:3], 55)  # k > n: relleno -1 / NaN
    assert (I[:, 50:] == -1).all() and np.isnan(S[:, 50:]).all()
    assert idx.m_effective == 32


def test_m_clamped_when_larger_than_rank() -> None:
    X = np.random.default_rng(0).standard_normal((20, 300)).astype(np.float32)
    idx = PCAKNN(m=256, seed=0).fit(X)
    assert idx.m_effective == 20 and idx.Y_.shape == (20, 20)


def test_deterministic_same_seed(low_rank_data) -> None:
    X, Q = low_rank_data
    a = PCAKNN(m=16, seed=7).fit(X).query(Q, 10)
    b = PCAKNN(m=16, seed=7).fit(X).query(Q, 10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_refit_time(low_rank_data) -> None:
    X, Q = low_rank_data
    idx = PCAKNN(m=32, seed=0).fit(X[:400])
    before = idx.query(Q, 10)[0]
    t = idx.refit_time(X[400:440])  # agregar 10 % de pacientes (E10)
    assert isinstance(t, float) and t > 0 and idx.last_refit_time_s == t
    np.testing.assert_array_equal(idx.query(Q, 10)[0], before)  # sin update no cambia
    idx.refit_time(X[400:440], update=True)
    assert idx.Y_.shape == (440, 32)
    I, S, _ = idx.query(X[420:421], 1)
    assert I[0, 0] == 420 and S[0, 0] < 1e-3  # el paciente agregado es buscable
    with pytest.raises(RuntimeError):
        PCAKNN(m=8, keep_data=False).fit(X[:50]).refit_time(X[50:60])
