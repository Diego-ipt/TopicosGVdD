"""Pruebas de E2LSH (Datar et al. 2004): fórmulas de colisión, índice e interfaz KNNIndex."""

import numpy as np
import pytest
from scipy import integrate, stats

from tcgaknn.algoritmos.base import StreamingProjectionAccumulator, default_feature_ids
from tcgaknn.algoritmos.lp.e2lsh import (
    E2LSH,
    candidate_probability,
    collision_probability,
    rho,
    suggest_w,
)


def _p_integral(c, w, p):
    """p(c) = ∫₀^w (1/c) f_p(t/c) (1 − t/w) dt, con f_p densidad de |X| (definición del paper)."""
    f = (lambda x: 2 * stats.norm.pdf(x)) if p == 2 else (lambda x: 2 * stats.cauchy.pdf(x))
    return integrate.quad(lambda t: f(t / c) / c * (1 - t / w), 0, w)[0]


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("c", [0.3, 1.0, 2.5, 8.0])
@pytest.mark.parametrize("w", [1.0, 4.0])
def test_closed_form_matches_integral(p, c, w):
    assert collision_probability(c, w, p) == pytest.approx(_p_integral(c, w, p), abs=1e-9)


def test_known_values():
    # u = w/c = 4: gaussiana ≈ 0.8005, Cauchy ≈ 0.6186
    assert collision_probability(1.0, 4.0, 2) == pytest.approx(0.80052, abs=1e-4)
    assert collision_probability(1.0, 4.0, 1) == pytest.approx(0.61858, abs=1e-4)
    assert collision_probability(0.0, 4.0, 2) == 1.0
    assert collision_probability(np.array([0.0, 1e300]), 4.0, 1)[1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("p", [1, 2])
def test_monotone_decreasing(p):
    c = np.linspace(0.01, 50, 2000)
    pc = collision_probability(c, 4.0, p)
    assert np.all(np.diff(pc) < 0)
    assert np.all((pc > 0) & (pc < 1))
    # crece con w
    assert np.all(collision_probability(c, 8.0, p) > pc)


@pytest.mark.parametrize("p", [1, 2])
def test_monte_carlo_collision(p):
    """Colisión empírica de una función h (vía el índice) vs fórmula, en ℓp."""
    rng = np.random.default_rng(1)
    d, w, F = 20, 4.0, 4000
    idx = E2LSH(p=p, w=w, r=1, L=F, seed=3, by_feature_id=False)
    u = rng.standard_normal(d)
    for c in [0.5, 2.0, 6.0]:
        delta = rng.standard_normal(d)
        delta *= c / np.linalg.norm(delta, ord=p)
        codes = idx.hash_codes(np.stack([u, u + delta]).astype(np.float32))
        emp = np.mean(codes[0, :, 0] == codes[1, :, 0])
        assert emp == pytest.approx(collision_probability(c, w, p), abs=0.03)


def test_rho_and_s_curve():
    assert rho(2.0, 4.0, 2) < 0.5  # ℓ2: ρ < 1/c
    assert rho(2.0, 4.0, 1) < 1.0
    pc = collision_probability(1.0, 4.0, 2)
    assert candidate_probability(1.0, 4.0, 2, 3, 5) == pytest.approx(1 - (1 - pc**3) ** 5)


def test_suggest_w():
    assert suggest_w(np.array([1.0, 2.0, 3.0, np.nan, 0.0])) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        suggest_w(np.array([0.0, np.nan]))


def _data(n=300, d=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


@pytest.mark.parametrize("p", [1, 2])
def test_high_recall_large_L(p):
    X = _data()
    k = 5
    diff = X[:20, None, :] - X[None, :, :]
    D = np.sqrt((diff**2).sum(-1)) if p == 2 else np.abs(diff).sum(-1)
    true = np.argsort(D, axis=1, kind="stable")[:, :k]
    knn = np.sort(D, axis=1)[:, k]
    w = suggest_w(knn, factor=4.0) if p == 2 else suggest_w(knn, factor=2.0)
    idx = E2LSH(p=p, w=w, r=2, L=60, seed=1).fit(X)
    I, S, st = idx.query(X[:20], k)
    rec = np.mean([len(set(I[i]) & set(true[i])) / k for i in range(20)])
    assert rec > 0.9
    assert I[:, 0].tolist() == list(range(20))  # el propio punto, distancia 0
    assert np.allclose(S[:, 0], 0)
    ok = I >= 0
    np.testing.assert_allclose(S[ok], np.take_along_axis(D, np.where(ok, I, 0), 1)[ok], rtol=1e-5)
    assert np.all(np.diff(S, axis=1)[np.isfinite(np.diff(S, axis=1))] >= 0)


def test_interface_and_edge_cases():
    X = _data(10, 5)
    idx = E2LSH(r=1, L=4, w=100.0, seed=0).fit(X)
    I, S, st = idx.query(X[:3], 15)  # k > n
    assert I.dtype == np.int64 and S.dtype == np.float32 and I.shape == (3, 15)
    assert np.all(I[:, 10:] == -1) and np.all(np.isnan(S[:, 10:]))
    assert st["n_candidates"].shape == (3,)
    assert E2LSH.score_kind == "distance"
    # consulta 1D y vacía
    assert idx.query(X[0], 2)[0].shape == (1, 2)
    assert idx.query(np.zeros((0, 5)), 2)[0].shape == (0, 2)
    # n = 0 y n = 1
    I0, S0, _ = E2LSH(r=1, L=2).fit(np.zeros((0, 5))).query(X[:2], 3)
    assert np.all(I0 == -1) and np.all(np.isnan(S0))
    I1, _, _ = E2LSH(r=1, L=2).fit(X[:1]).query(X[:1], 3)
    assert I1[0].tolist() == [0, -1, -1]
    # filas nulas: todas colisionan entre sí
    Z = np.zeros((4, 5), np.float32)
    Iz, Sz, _ = E2LSH(r=2, L=3).fit(Z).query(Z[:1], 4)
    assert Iz[0].tolist() == [0, 1, 2, 3] and np.all(Sz == 0)
    # NaN se rechaza (en X y en proyecciones)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        E2LSH().fit(Xn)
    with pytest.raises(ValueError):
        E2LSH(r=1, L=2, rerank="none").fit_projections(np.full((3, 2), np.nan))
    with pytest.raises(ValueError):
        idx.query(np.zeros((1, 6)), 2)
    with pytest.raises(ValueError):
        E2LSH(p=3)


def test_determinism_and_dtypes():
    X = _data(100, 8)
    a = E2LSH(seed=7).fit(X).query(X[:10], 5)
    b = E2LSH(seed=7).fit(X.astype(np.float64)).query(X[:10], 5)
    np.testing.assert_array_equal(a[0], b[0])
    c = E2LSH(seed=8).fit(X)
    assert not np.array_equal(c.hash_codes(X), E2LSH(seed=7).fit(X).hash_codes(X))


def test_floor_and_beta():
    idx = E2LSH(p=2, w=2.0, r=1, L=3, seed=0)
    assert np.all((idx.beta >= 0) & (idx.beta < 2.0))
    Y = np.array([[-0.1, 0.0, 5.0]])
    codes = idx.codes_from_projections(Y)[0, :, 0]
    np.testing.assert_array_equal(codes, np.floor((Y[0] + idx.beta) / 2.0).astype(np.int64))
    # overflow: proyecciones enormes no revientan
    big = idx.codes_from_projections(np.array([[1e300, -1e300, 0.0]]))
    assert big.dtype == np.int64


@pytest.mark.parametrize("rerank", ["d", "proj", "none"])
def test_rerank_modes_and_streaming(rerank):
    X = _data(80, 12, seed=2)
    ids = default_feature_ids(12)
    idx = E2LSH(p=2, w=8.0, r=2, L=10, seed=4, rerank=rerank).fit(X, feature_ids=ids)
    ref = idx.query(X[:5], 4)
    acc = StreamingProjectionAccumulator(80, idx.n_funcs, seed=idx.a_seed, standardize=False)
    for j, f in enumerate(ids):
        acc.update(f, X[:, j])
    Y = acc.raw()
    np.testing.assert_allclose(Y, idx.projections(X), rtol=1e-6, atol=1e-6)
    idx2 = E2LSH(p=2, w=8.0, r=2, L=10, seed=4, rerank=rerank)
    idx2.fit_projections(Y, X=X if rerank == "d" else None, feature_ids=ids)
    out = idx2.query(X[:5], 4)
    np.testing.assert_array_equal(out[0], ref[0])
    assert out[0][:, 0].tolist() == list(range(5))


def test_l1_variant_uses_cauchy():
    idx = E2LSH(p=1, r=4, L=500, seed=0)
    A = idx.a_matrix(default_feature_ids(4)).ravel()
    # colas pesadas: hay valores muy grandes que una normal no produce
    assert np.abs(A).max() > 50
    assert np.median(np.abs(A)) == pytest.approx(1.0, abs=0.1)  # mediana |Cauchy| = 1
