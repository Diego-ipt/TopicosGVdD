from __future__ import annotations

import math

import numpy as np
import pytest

from tcgaknn.algoritmos.base import (
    KNNIndex,
    StreamingProjectionAccumulator,
    default_feature_ids,
    l2_normalize,
    omega_matrix,
    topk,
)
from tcgaknn.algoritmos.euclidiana.projection import (
    ProjectedKNN,
    RandomProjection,
    achlioptas_row,
    distortion_stats,
    jl_eps_for,
    jl_min_dim,
    project,
    random_matrix,
)


# --------------------------------------------------------------------------
# Cota JL
# --------------------------------------------------------------------------


def test_jl_min_dim() -> None:
    # Tabla del plan §3.2 (Dasgupta–Gupta con ceil)
    assert [jl_min_dim(11000, e) for e in (0.5, 0.3, 0.2)] == [447, 1034, 2148]
    assert [jl_min_dim(1000, e) for e in (0.5, 0.3, 0.2)] == [332, 768, 1595]
    # Es la misma fórmula que sklearn, que trunca en vez de redondear hacia arriba
    sk = pytest.importorskip("sklearn.random_projection")
    for e in (0.5, 0.3, 0.2):
        assert jl_min_dim(11000, e) == int(sk.johnson_lindenstrauss_min_dim(11000, eps=e)) + 1
    with pytest.raises(ValueError):
        jl_min_dim(100, 1.0)


def test_jl_eps_for_inverts_bound() -> None:
    n = 11000
    prev = 1.0
    for m in (256, 447, 1034, 2148, 5000):
        eps = jl_eps_for(n, m)
        assert eps is not None and 0 < eps < prev  # decrece con m
        assert jl_min_dim(n, eps) <= m  # cumple la cota
        assert jl_min_dim(n, eps * (1 - 1e-6)) > m  # y es el mínimo
        prev = eps
    assert jl_eps_for(n, 1034) == pytest.approx(0.3, abs=1e-3)
    # Sin ε < 1 posible si m < 24 ln n (≈ 223,3 para n = 11.000)
    assert jl_eps_for(n, 223) is None and jl_eps_for(n, 224) is not None


# --------------------------------------------------------------------------
# Distorsión (JL)
# --------------------------------------------------------------------------


def _clustered(n: int, d: int, C: int, r: int, seed: int) -> np.ndarray:
    """Clusters con un subespacio latente propio de dimensión r, más ruido."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, C, n)
    centers = rng.normal(size=(C, d))
    bases = rng.normal(size=(C, r, d))
    lat = rng.normal(size=(n, r))
    X = centers[lab] + 0.6 * np.einsum("nr,nrd->nd", lat, bases[lab]) + 0.2 * rng.normal(size=(n, d))
    return X.astype(np.float32)


@pytest.mark.parametrize("kind", ["gaussian", "achlioptas"])
def test_jl_distortion(kind: str) -> None:
    n, d = 200, 400  # d chico: generar Ω fila a fila domina el tiempo
    X = l2_normalize(_clustered(n, d, C=5, r=3, seed=3))
    for eps in (0.3, 0.5):
        m = jl_min_dim(n, eps)
        Xp = RandomProjection(m, kind=kind, seed=11).fit_transform(X)
        st = distortion_stats(X, Xp, eps_list=(eps,))
        assert st["n_pairs"] == n * (n - 1) // 2
        # E[ρ] = 1. Con una sola R los pares no son independientes (las diferencias
        # viven en un subespacio de baja dimensión), así que se promedia sobre semillas.
        means = [distortion_stats(X, RandomProjection(m, kind=kind, seed=s).fit_transform(X),
                                  eps_list=(eps,))["mean"] for s in range(6)]
        assert abs(np.mean(means) - 1) < 0.02
        # m·ρ ~ χ²_m (gaussiana): Var[ρ] = 2/m; Achlioptas tiene el mismo 4º momento
        assert st["std"] == pytest.approx(math.sqrt(2 / m), rel=0.15)
        assert st["pct_outside"][eps] <= 5.0  # ≥ 95 % de los pares dentro de 1 ± ε
        assert st["p1"] > 1 - eps and st["p99"] < 1 + eps
    # Con m muy chico la distorsión sí se nota (el test discrimina)
    st = distortion_stats(X, RandomProjection(16, kind=kind, seed=11).fit_transform(X), eps_list=(0.2,))
    assert st["pct_outside"][0.2] > 40.0  # P(|χ²₁₆/16 − 1| > 0,2) ≈ 0,57


def test_distortion_stats_known_cases() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 10))
    st = distortion_stats(X, X)
    assert st["mean"] == pytest.approx(1) and st["std"] == pytest.approx(0, abs=1e-12)
    assert all(v == 0 for v in st["pct_outside"].values())
    st = distortion_stats(X, 2 * X, eps_list=(0.5,))
    assert st["mean"] == pytest.approx(4) and st["pct_outside"][0.5] == 100.0
    # Pares degenerados (filas idénticas) se excluyen y se cuentan
    Xd = np.vstack([X, X[:1]])
    st = distortion_stats(Xd, Xd)
    assert st["n_degenerate"] == 1 and st["n_pairs"] == 31 * 30 // 2 - 1
    # Muestreo de pares y pares explícitos
    st = distortion_stats(X, X, max_pairs=100, rng=np.random.default_rng(1), return_rho=True)
    assert st["n_pairs"] == 100 and st["rho"].shape == (100,)
    st = distortion_stats(X, 3 * X, pairs=np.array([[0, 1], [2, 5]]))
    assert st["n_pairs"] == 2 and st["mean"] == pytest.approx(9)


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------


def test_gaussian_matrix_uses_shared_omega() -> None:
    d, m = 300, 48
    rp = RandomProjection(m, seed=5).fit(d)
    O = omega_matrix(default_feature_ids(d), m, seed=5)
    np.testing.assert_allclose(rp.components_, O / np.sqrt(m), rtol=1e-6)
    np.testing.assert_array_equal(random_matrix(d, m, seed=5), O)
    # Las primeras m columnas no dependen de cols (misma Ω que un acumulador más ancho)
    np.testing.assert_array_equal(RandomProjection(m, seed=5, cols=200).fit(d).components_, rp.components_)
    # row() reproduce la fila de R de un rasgo
    np.testing.assert_array_equal(rp.row("f17"), rp.components_[17])
    X = np.random.default_rng(0).normal(size=(5, d)).astype(np.float32)
    np.testing.assert_allclose(rp.transform(X), project(X, O, m), rtol=1e-5, atol=1e-6)


def test_achlioptas_entries_and_norms() -> None:
    d, m = 2000, 256
    rp = RandomProjection(m, kind="achlioptas", seed=2).fit(d)
    R = rp.components_
    S = R * np.sqrt(m / 3)
    vals = np.unique(np.round(S, 5))
    assert set(vals.tolist()) <= {-1.0, 0.0, 1.0}
    N = S.size  # 512.000 entradas: error estándar de la frecuencia ≈ 6e-4
    assert (S > 0).sum() / N == pytest.approx(1 / 6, abs=0.005)
    assert (S < 0).sum() / N == pytest.approx(1 / 6, abs=0.005)
    assert (S == 0).sum() / N == pytest.approx(2 / 3, abs=0.005)
    # Matriz propia: distinta de Ω, determinista por (semilla, ID)
    assert not np.allclose(R, RandomProjection(m, seed=2).fit(d).components_)
    np.testing.assert_array_equal(rp.row("f3"), R[3])
    np.testing.assert_array_equal(achlioptas_row("f3", 512, 2)[:m], achlioptas_row("f3", m, 2))
    assert not np.array_equal(achlioptas_row("f3", m, 2), achlioptas_row("f3", m, 3))

    # Preserva normas en esperanza: E‖Ru‖² = ‖u‖², Var = 2/m
    rng = np.random.default_rng(4)
    U = l2_normalize(rng.normal(size=(2000, d)))
    r = (rp.transform(U) ** 2).sum(axis=1)
    assert r.mean() == pytest.approx(1, abs=0.03)
    assert r.std() == pytest.approx(math.sqrt(2 / m), rel=0.15)
    # Vectores canónicos (máximo Σu⁴): ‖R e_j‖² = 3·Bin(m, 1/3)/m, media 1
    r1 = (R**2).sum(axis=1)
    assert r1.mean() == pytest.approx(1, abs=0.03)


def test_achlioptas_streaming_by_rows() -> None:
    rng = np.random.default_rng(6)
    n, d, m = 20, 150, 32
    ids = [f"gen{j}" for j in range(d)]
    X = rng.normal(size=(n, d)).astype(np.float32)
    rp = RandomProjection(m, kind="achlioptas", seed=9, feature_ids=ids).fit(d)
    Y = np.zeros((n, m))
    for j, f in enumerate(ids):  # una fila de rasgo a la vez, sin guardar R
        Y += np.outer(X[:, j], rp.row(f))
    np.testing.assert_allclose(rp.transform(X), Y, rtol=1e-4, atol=1e-5)


def test_projection_equals_streaming_accumulator() -> None:
    rng = np.random.default_rng(1)
    n, d, cols, m = 40, 300, 64, 32
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + rng.normal(size=d)
    X[rng.random((n, d)) < 0.02] = np.nan
    X[:, 5] = 2.0  # constante: el acumulador la descarta
    fit_mask = np.arange(n) < 30
    ids = [f"g{j}|{j * 7}" for j in range(d)]
    acc = StreamingProjectionAccumulator(n, cols, seed=7, fit_mask=fit_mask, batch_size=17)
    for j in range(d):
        acc.update(ids[j], X[:, j])

    ref = X[fit_mask]
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    keep = sd > 1e-12
    Z = np.nan_to_num((X[:, keep] - mu[keep]) / sd[keep])
    kept = [f for f, k in zip(ids, keep) if k]
    assert kept == acc.feature_ids

    expected = acc.projections(m)
    rp = RandomProjection(m, seed=7, feature_ids=kept).fit(Z.shape[1])
    np.testing.assert_allclose(rp.transform(l2_normalize(Z)), expected, rtol=1e-4, atol=1e-5)
    rp_wide = RandomProjection(m, seed=7, feature_ids=kept, cols=cols).fit(Z)
    np.testing.assert_array_equal(rp_wide.components_, rp.components_)
    # ProjectedKNN normaliza internamente: su base proyectada es la del acumulador
    knn = ProjectedKNN(m, seed=7, feature_ids=kept).fit(Z)
    np.testing.assert_allclose(knn.P_, expected, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# ProjectedKNN
# --------------------------------------------------------------------------


def _recall(exact: np.ndarray, approx: np.ndarray, q: np.ndarray, k: int) -> float:
    hits = []
    for i in range(len(q)):
        a = approx[i][approx[i] != q[i]][:k]
        hits.append(len(set(exact[i, :k].tolist()) & set(a.tolist())) / k)
    return float(np.mean(hits))


def test_projected_knn_recall_clusters() -> None:
    X = _clustered(2000, 2000, C=20, r=3, seed=0)
    Xn = l2_normalize(X)
    q = np.arange(200)
    S = Xn[q] @ Xn.T
    S[np.arange(len(q)), q] = -np.inf  # excluye la propia consulta
    exact, _ = topk(S, 10)
    rec = {}
    for m in (32, 256):
        idx, _, _ = ProjectedKNN(m, seed=1).fit(X).query(X[q], 11)
        rec[m] = _recall(exact, idx, q, 10)
    assert rec[256] >= 0.9
    assert rec[256] - rec[32] >= 0.05  # el recall crece con m
    idx, _, _ = ProjectedKNN(256, kind="achlioptas", seed=1).fit(X).query(X[q], 11)
    assert _recall(exact, idx, q, 10) >= 0.9


def test_projected_knn_interface() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 120)).astype(np.float32)
    knn = ProjectedKNN(64, seed=0).fit(X)
    assert isinstance(knn, KNNIndex) and knn.score_kind == "distance"
    idx, dist, st = knn.query(X[:7], 5)
    assert idx.dtype == np.int64 and dist.dtype == np.float32 and idx.shape == (7, 5)
    np.testing.assert_array_equal(idx[:, 0], np.arange(7))  # cada punto es su vecino más cercano
    assert np.all(dist[:, 0] < 1e-3) and np.all(np.diff(dist, axis=1) >= 0)
    assert st["n_candidates"].tolist() == [50] * 7
    assert st["index_bytes"] == 50 * 64 * 4 and st["matrix_bytes"] == 120 * 64 * 4
    assert st["t_project_s"] >= 0 and st["t_search_s"] >= 0
    # Distancia = ‖q' − p'‖ en ℝ^m sobre filas normalizadas
    P = knn.transform(X)
    np.testing.assert_allclose(dist[3], np.linalg.norm(P[3] - P[idx[3]], axis=1), atol=1e-4)
    # Invariante a la escala de las filas (normalize=True)
    idx2, dist2, _ = ProjectedKNN(64, seed=0).fit(5 * X).query(X[:7] * 0.1, 5)
    np.testing.assert_array_equal(idx2, idx)
    # Relleno si k > n
    idx3, dist3, _ = knn.query(X[:2], 60)
    assert (idx3[:, 50:] == -1).all() and np.isnan(dist3[:, 50:]).all()
    # Bloques de consultas no cambian el resultado
    idx4, _, _ = ProjectedKNN(64, seed=0, query_block=3).fit(X).query(X[:7], 5)
    np.testing.assert_array_equal(idx4, idx)


# --------------------------------------------------------------------------
# Casos borde (revisión adversarial)
# --------------------------------------------------------------------------


def test_projected_knn_rejects_nan() -> None:
    # l2_normalize volvía nula una fila con NaN: el paciente quedaba a distancia 0
    # de cualquier fila nula sin aviso.
    X = np.random.default_rng(0).normal(size=(6, 20)).astype(np.float32)
    Xn = X.copy()
    Xn[2, 3] = np.nan
    with pytest.raises(ValueError):
        ProjectedKNN(8).fit(Xn)
    knn = ProjectedKNN(8).fit(X)
    with pytest.raises(ValueError):
        knn.query(Xn[2:3], 2)
    with pytest.raises(ValueError):
        knn.query(X[:1], -1)


def test_projected_knn_edge_sizes() -> None:
    X = np.random.default_rng(1).normal(size=(5, 20)).astype(np.float32)
    # base vacía: todo relleno
    idx, dist, st = ProjectedKNN(8).fit(X[:0]).query(X[:2], 3)
    assert (idx == -1).all() and np.isnan(dist).all() and st["n_candidates"].tolist() == [0, 0]
    knn = ProjectedKNN(8, seed=3).fit(X)
    # consulta 1D, k = 0, cero consultas
    assert knn.query(X[0], 2)[0].shape == (1, 2)
    assert knn.query(X[:2], 0)[0].shape == (2, 0)
    assert knn.query(X[:0], 2)[0].shape == (0, 2)
    # n = 1
    i1, d1, _ = ProjectedKNN(8).fit(X[:1]).query(X[:1], 3)
    assert i1.tolist() == [[0, -1, -1]]
    # determinismo y dtypes de entrada
    a = ProjectedKNN(8, seed=3).fit(X.astype(np.float64)).query(X[:3], 4)
    b = knn.query(X[:3], 4)
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(ValueError):
        knn.query(X[:, :10], 2)  # d distinto


def test_distortion_stats_nonfinite_and_bad_pairs() -> None:
    X = np.random.default_rng(2).normal(size=(5, 4))
    Xb = X.copy()
    Xb[0, 0] = np.nan
    st = distortion_stats(Xb, Xb)
    assert st["n_nonfinite"] == 4 and st["n_degenerate"] == 0 and st["n_pairs"] == 6
    with pytest.raises(ValueError):
        distortion_stats(X, X, pairs=np.array([[0, -1]]))
    with pytest.raises(ValueError):
        distortion_stats(X, X, pairs=np.array([[0, 5]]))


def test_achlioptas_scale_and_gaussian_scale() -> None:
    # Escalas exactas: √(3/m) y 1/√m (romperlas cambia E‖Ru‖²)
    rp = RandomProjection(12, kind="achlioptas", seed=0).fit(50)
    assert set(np.unique(np.abs(rp.components_)).round(6).tolist()) <= {0.0, round(math.sqrt(3 / 12), 6)}
    assert RandomProjection(12, seed=0).scale == pytest.approx(1 / math.sqrt(12))
