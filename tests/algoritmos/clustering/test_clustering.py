from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans as SkMiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from tcgaknn.algoritmos.base import default_feature_ids, omega_matrix
from tcgaknn.algoritmos.clustering.clustering import (
    MiniBatchKMeans,
    ari,
    hamming_to_pm1,
    kmeans_plusplus_init,
)


def _blobs(n: int, K: int, d: int, sep: float, sd: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """K nubes gaussianas con centros N(0, sep²) y desviación sd; etiquetas balanceadas."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(K, d))
    y = rng.permutation(np.arange(n) % K)
    X = centers[y] + rng.normal(0, sd, size=(n, d))
    return X.astype(np.float32), y


# --------------------------------------------------------------------------
# k-means++
# --------------------------------------------------------------------------


def test_kmeanspp_one_center_per_blob() -> None:
    """Con nubes muy separadas, el muestreo D² cae en una nube nueva en cada paso."""
    X, y = _blobs(1000, 10, 8, sep=50.0, sd=1.0, seed=0)
    for s in range(10):
        C, idx = kmeans_plusplus_init(X, 10, np.random.default_rng(s), n_local_trials=1)
        assert C.shape == (10, 8) and C.dtype == np.float32 and idx.dtype == np.int64
        np.testing.assert_array_equal(C, X[idx])
        assert len(set(y[idx].tolist())) == 10


def test_kmeanspp_deterministic_and_degenerate() -> None:
    X, _ = _blobs(300, 5, 4, sep=10.0, sd=1.0, seed=1)
    a = kmeans_plusplus_init(X, 5, np.random.default_rng(3))[1]
    b = kmeans_plusplus_init(X, 5, np.random.default_rng(3))[1]
    np.testing.assert_array_equal(a, b)
    # Menos puntos distintos que K: no debe fallar (D² = 0 ⇒ relleno uniforme)
    Xd = np.repeat(np.eye(3, dtype=np.float32), 10, axis=0)
    C, _ = kmeans_plusplus_init(Xd, 5, np.random.default_rng(0))
    assert C.shape == (5, 3)
    with pytest.raises(ValueError):
        kmeans_plusplus_init(Xd, 31, np.random.default_rng(0))


# --------------------------------------------------------------------------
# Paso de Sculley
# --------------------------------------------------------------------------


def _sculley_reference(C0: np.ndarray, v0: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Algoritmo 1 de Sculley (2010) literal: caché de asignaciones y bucle punto a punto."""
    C, v = C0.astype(np.float64).copy(), v0.copy()
    near = [int(np.argmin(((C - x) ** 2).sum(1))) for x in M]
    for x, c in zip(M.astype(np.float64), near):
        v[c] += 1
        eta = 1.0 / v[c]
        C[c] = (1 - eta) * C[c] + eta * x
    return C, v


def test_vectorized_step_equals_sculley_loop() -> None:
    rng = np.random.default_rng(5)
    X, _ = _blobs(600, 6, 5, sep=4.0, sd=1.5, seed=5)
    C0 = X[rng.choice(600, 6, replace=False)]
    model = MiniBatchKMeans(6, init=C0)
    C_ref, v_ref = C0.astype(np.float64), np.zeros(6, dtype=np.int64)
    for s in range(0, 600, 100):  # varios lotes: v[c] se acumula entre lotes
        model.partial_fit(X[s : s + 100])
        C_ref, v_ref = _sculley_reference(C_ref, v_ref, X[s : s + 100])
        np.testing.assert_allclose(model._C, C_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(model.counts_, v_ref)
    assert model.n_seen_ == 600 and model.n_iter_ == 6


# --------------------------------------------------------------------------
# Calidad del clustering
# --------------------------------------------------------------------------


def test_well_separated_blobs_ari() -> None:
    X, y = _blobs(3000, 8, 20, sep=10.0, sd=1.0, seed=2)
    m = MiniBatchKMeans(8, batch_size=256, n_iter=100, seed=0).fit(X)
    assert ari(y, m.labels_) > 0.9
    np.testing.assert_array_equal(m.predict(X), m.labels_)
    assert m.inertia(X) == pytest.approx(m.inertia_, rel=1e-9)


def test_33_types_in_projected_space() -> None:
    """Escenario del proyecto: K = 33 tipos en ℝ^64 (proyecciones RP)."""
    X, y = _blobs(33 * 60, 33, 64, sep=3.0, sd=1.0, seed=4)
    m = MiniBatchKMeans(33, batch_size=256, n_iter=200, seed=1).fit(X)
    assert ari(y, m.labels_) > 0.9


def test_inertia_comparable_to_sklearn() -> None:
    """Nubes solapadas (problema no trivial): la inercia media sobre 5 semillas
    queda dentro de un 10 % de la de sklearn.MiniBatchKMeans (mismo algoritmo
    de Sculley y misma siembra k-means++; sklearn solo agrega reasignación de
    centros vacíos y otro criterio de parada). Una tasa de aprendizaje errónea
    (p. ej. η fija) o una asignación mal hecha la alejan mucho más."""
    X, _ = _blobs(4000, 12, 16, sep=2.5, sd=1.5, seed=6)
    ours = [MiniBatchKMeans(12, batch_size=256, n_iter=300, seed=s).fit(X).inertia_ for s in range(5)]
    sk = [
        SkMiniBatchKMeans(12, batch_size=256, max_iter=50, n_init=1, random_state=s).fit(X).inertia_
        for s in range(5)
    ]
    assert np.mean(ours) <= 1.10 * np.mean(sk)
    assert np.mean(ours) >= 0.90 * np.mean(sk)
    # Control negativo: centros al azar (sin entrenar) dan una inercia mucho peor
    rand = MiniBatchKMeans(12, init=np.random.default_rng(0).normal(size=(12, 16)) * 2.5)
    rand.partial_fit(X[:12])
    assert rand.inertia(X) > 1.5 * np.mean(sk)


def test_partial_fit_streaming() -> None:
    """Lotes de un stream de pacientes (2 pasadas en orden aleatorio) bastan para recuperar las nubes."""
    X, y = _blobs(3000, 8, 20, sep=10.0, sd=1.0, seed=2)
    rng = np.random.default_rng(0)
    m = MiniBatchKMeans(8, seed=0)
    for _ in range(2):
        perm = rng.permutation(3000)
        m.fit_stream(X[perm[s : s + 256]] for s in range(0, 3000, 256))
    assert m.n_seen_ == 6000 and m.counts_.sum() == 6000
    assert ari(y, m.predict(X)) > 0.9
    with pytest.raises(ValueError):
        MiniBatchKMeans(8).partial_fit(X[:5])  # el primer lote necesita ≥ K filas


def test_tol_stops_early_and_determinism() -> None:
    X, _ = _blobs(2000, 5, 10, sep=10.0, sd=1.0, seed=7)
    a = MiniBatchKMeans(5, n_iter=500, seed=3, tol=1e-4).fit(X)
    b = MiniBatchKMeans(5, n_iter=500, seed=3, tol=1e-4).fit(X)
    assert a.n_iter_ < 500
    np.testing.assert_array_equal(a.centers_, b.centers_)
    assert MiniBatchKMeans(5, n_iter=500, seed=3).fit(X).n_iter_ == 500  # tol = 0: sin parada


# --------------------------------------------------------------------------
# Firmas SimHash y ARI
# --------------------------------------------------------------------------


def _pack(bits: np.ndarray) -> np.ndarray:
    """Empaquetado de §5.4 (simhash.signatures)."""
    p = np.packbits(bits, axis=1, bitorder="little")
    return np.ascontiguousarray(p).view(np.uint64)


def test_hamming_to_pm1_identity() -> None:
    rng = np.random.default_rng(0)
    bits = rng.random((50, 128)) < 0.5
    P = hamming_to_pm1(bits)
    assert P.dtype == np.float32 and set(np.unique(P).tolist()) == {-1.0, 1.0}
    ham = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    sq = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.testing.assert_array_equal(sq, 4 * ham)  # ‖x − y‖² = 4·Ham
    np.testing.assert_array_equal(P @ P.T, 128 - 2 * ham)
    np.testing.assert_array_equal(hamming_to_pm1(_pack(bits), 128), P)  # empaquetado uint64
    np.testing.assert_array_equal(hamming_to_pm1(bits.astype(np.uint8)), P)
    np.testing.assert_array_equal(hamming_to_pm1(_pack(bits), 100), P[:, :100])
    with pytest.raises(ValueError):
        hamming_to_pm1(np.array([[0, 2, 1]]))


def test_mbk_on_simhash_signatures() -> None:
    """Clusters de tipos sobreviven a SimHash: MBK sobre {−1,+1}^256 recupera las nubes.

    En este espacio k-means (incluso Lloyd completo de sklearn con n_init=1)
    cae a veces en un óptimo local (dos nubes fusionadas); con n_init = 5
    reinicios basta con que uno acierte. Se verifica además que el reinicio
    elige la corrida de menor inercia.
    """
    X, y = _blobs(1200, 6, 300, sep=1.0, sd=0.6, seed=8)
    O = omega_matrix(default_feature_ids(300), 256, seed=0)
    bits = (X @ O) >= 0
    P = hamming_to_pm1(_pack(bits), 256)
    m = MiniBatchKMeans(6, n_iter=150, seed=0, n_init=5).fit(P)
    assert ari(y, m.labels_) > 0.9
    singles = [MiniBatchKMeans(6, n_iter=150, seed=s).fit(P).inertia_ for s in range(5)]
    assert m.inertia_ <= min(singles) * 1.001
    assert m.inertia(P) == pytest.approx(m.inertia_, rel=1e-9)


def test_ari_wrapper() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    assert ari(y, y) == 1.0
    assert ari(y, np.array([5, 5, 9, 9, 7, 7])) == 1.0  # invariante a renombrar
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 10, 5000), rng.integers(0, 10, 5000)
    assert abs(ari(a, b)) < 0.01
    assert ari(a, b) == adjusted_rand_score(a, b)
    with pytest.raises(ValueError):
        ari(y, y[:3])


def test_hamming_to_pm1_b_too_large_and_packed_roundtrip() -> None:
    """b mayor que los bits disponibles es error (antes se truncaba en silencio);
    el desempaquetado coincide bit a bit con el empaquetado little-endian."""
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(5, 70)).astype(bool)
    by = np.packbits(bits, axis=1, bitorder="little")
    by = np.concatenate([by, np.zeros((5, 16 - by.shape[1]), np.uint8)], axis=1)
    packed = by.view("<u8").astype(np.uint64)
    np.testing.assert_array_equal(hamming_to_pm1(packed, b=70), np.where(bits, 1.0, -1.0))
    with pytest.raises(ValueError):
        hamming_to_pm1(packed, b=129)
    with pytest.raises(ValueError):
        hamming_to_pm1(bits, b=71)


def test_nonfinite_input_rejected() -> None:
    X = np.random.default_rng(0).normal(size=(20, 3))
    X[4, 1] = np.nan
    with pytest.raises(ValueError):
        MiniBatchKMeans(K=2).fit(X)
    with pytest.raises(ValueError):
        MiniBatchKMeans(K=2).partial_fit(X)
    with pytest.raises(ValueError):
        kmeans_plusplus_init(X, 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        MiniBatchKMeans(K=30).fit(np.zeros((5, 3)))  # K > n
