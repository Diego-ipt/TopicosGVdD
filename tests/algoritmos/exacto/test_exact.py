from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from tcgaknn.algoritmos.base import KNNIndex, default_feature_ids
from tcgaknn.algoritmos.exacto.exact import ExactKNN, StreamingExactKNN, l1_accumulate

SK_METRIC = {"cosine": "cosine", "correlation": "correlation", "l2": "euclidean", "l1": "manhattan"}


def _clustered(rng: np.random.Generator, n: int, d: int, n_clusters: int = 8) -> np.ndarray:
    """Datos con estructura (centros + ruido + escala y desplazamiento por muestra)."""
    centers = rng.normal(size=(n_clusters, d)) * 2.0
    lab = rng.integers(0, n_clusters, size=n)
    X = centers[lab] + rng.normal(size=(n, d))
    X = X * rng.uniform(0.5, 2.0, size=(n, 1)) + rng.normal(size=(n, 1)) * 3.0
    return X.astype(np.float32)


def _assert_same_neighbors(idx, val, ref_idx, ref_val, rtol, atol) -> None:
    """Índices iguales salvo empates numéricos (mismo valor dentro de la tolerancia)."""
    np.testing.assert_allclose(val, ref_val, rtol=rtol, atol=atol)
    diff = idx != ref_idx
    # Donde difieren los índices, los valores deben estar empatados.
    assert diff.mean() < 0.01, f"{diff.mean():.3%} de los índices difieren"


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    X = _clustered(rng, 560, 2000)
    return X[:500], X[500:]


@pytest.mark.parametrize("metric", ["cosine", "correlation", "l2", "l1"])
def test_matches_sklearn(data, metric: str) -> None:
    X, Q = data
    k = 10
    ex = ExactKNN(metric, block_size=17).fit(X)
    assert isinstance(ex, KNNIndex)
    idx, val, stats = ex.query(Q, k)
    assert idx.dtype == np.int64 and val.dtype == np.float32 and idx.shape == (60, k)
    nn = NearestNeighbors(algorithm="brute", metric=SK_METRIC[metric]).fit(X.astype(np.float64))
    ref_d, ref_i = nn.kneighbors(Q.astype(np.float64), n_neighbors=k)
    if metric in ("cosine", "correlation"):
        assert ex.score_kind == "similarity"
        ref_v = 1.0 - ref_d  # sklearn devuelve distancia = 1 − similitud
        _assert_same_neighbors(idx, val, ref_i, ref_v, rtol=0, atol=2e-5)
        assert (np.diff(val, axis=1) <= 0).all()
    else:
        assert ex.score_kind == "distance"
        _assert_same_neighbors(idx, val, ref_i, ref_d, rtol=1e-5, atol=1e-4)
        assert (np.diff(val, axis=1) >= 0).all()
    np.testing.assert_array_equal(stats["n_candidates"], np.full(60, 500))
    assert stats["metric"] == metric and stats["index_bytes"] > 0


def test_cosine_is_scale_invariant_and_correlation_is_pearson(data) -> None:
    X, Q = data
    ex = ExactKNN("cosine").fit(X)
    i1, v1, _ = ex.query(Q, 5)
    i2, v2, _ = ex.query(Q * 7.5, 5)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(v1, v2, atol=1e-6)
    # correlación = Pearson de numpy entre la consulta y sus vecinos
    ec = ExactKNN("correlation").fit(X)
    idx, val, _ = ec.query(Q[:5], 3)
    for i in range(5):
        for j in range(3):
            r = np.corrcoef(Q[i].astype(np.float64), X[idx[i, j]].astype(np.float64))[0, 1]
            assert abs(val[i, j] - r) < 1e-5
    # la correlación ignora un desplazamiento constante por muestra; el coseno no
    shifted = Q[:5] + 50.0
    i_c, _, _ = ec.query(shifted, 3)
    np.testing.assert_array_equal(i_c, idx)
    i_cos, _, _ = ex.query(shifted, 3)
    assert not np.array_equal(i_cos, ex.query(Q[:5], 3)[0])


@pytest.mark.parametrize("metric", ["cosine", "l2", "l1", "correlation"])
def test_block_size_does_not_change_result(data, metric: str) -> None:
    X, Q = data
    a = ExactKNN(metric, block_size=1).fit(X).query(Q[:20], 7)
    b = ExactKNN(metric, block_size=1024).fit(X).query(Q[:20], 7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("metric", ["cosine", "l2", "l1", "correlation"])
def test_padding_and_ties(metric: str) -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(3, 6)).astype(np.float32)
    X = np.vstack([base, base])  # filas 3, 4, 5 duplican a 0, 1, 2
    idx, val, _ = ExactKNN(metric).fit(X).query(base[1], 8)
    # Los duplicados empatan: gana el índice menor y la copia viene justo después.
    assert idx[0, :2].tolist() == [1, 4]
    assert idx[0, 6:].tolist() == [-1, -1] and np.isnan(val[0, 6:]).all()
    assert sorted(idx[0, :6].tolist()) == list(range(6))


def test_l2_precision_with_large_offset() -> None:
    """Con un desplazamiento común grande, ‖q‖²+‖x‖²−2q·x en float32 pierde toda la
    precisión; centrar en la media de la base y refinar en float64 la recupera."""
    rng = np.random.default_rng(5)
    n, d = 300, 512
    offset = rng.normal(size=d) * 1e4
    delta = rng.normal(size=(n, d)) * np.linspace(1e-2, 1.0, n)[:, None]
    X = (offset + delta).astype(np.float32)
    q = (offset + rng.normal(size=d) * 1e-3).astype(np.float32)
    ref = np.sqrt(((X.astype(np.float64) - q.astype(np.float64)) ** 2).sum(axis=1))
    ref_idx = np.argsort(ref, kind="stable")[:10]
    idx, val, _ = ExactKNN("l2").fit(X).query(q, 10)
    np.testing.assert_array_equal(idx[0], ref_idx)
    np.testing.assert_allclose(val[0], ref[ref_idx], rtol=1e-5)
    # Control: la fórmula ingenua en float32 sí falla en este caso.
    naive = (q @ q) + np.einsum("ij,ij->i", X, X) - 2 * (X @ q)
    assert not np.array_equal(np.argsort(naive, kind="stable")[:10], ref_idx)


def test_l1_kernel_matches_cdist() -> None:
    rng = np.random.default_rng(9)
    A = rng.normal(size=(13, 77)).astype(np.float32)
    B = rng.normal(size=(29, 77)).astype(np.float32)
    out = np.full((13, 29), 1.5)
    l1_accumulate(A, B, out)
    np.testing.assert_allclose(out - 1.5, cdist(A.astype(np.float64), B.astype(np.float64), "cityblock"), rtol=1e-10)


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _materialize(X, fit_mask, transform=None):
    """La misma regla del stream aplicada a la matriz completa (muestras × rasgos)."""
    V = X.astype(np.float64)
    if transform is not None:
        V = transform(V)
    ref = V[fit_mask] if fit_mask is not None else V
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    nvalid = (~np.isnan(ref)).sum(axis=0)
    keep = (nvalid >= 2) & np.isfinite(sd) & (sd > 1e-12)
    Z = np.nan_to_num((V[:, keep] - mu[keep]) / sd[keep])
    return Z.astype(np.float32), int((~keep).sum())


@pytest.mark.parametrize("batch_size", [1, 13, 256])
def test_streaming_matches_in_memory(batch_size: int) -> None:
    rng = np.random.default_rng(11)
    n, d = 120, 400
    X = np.abs(_clustered(rng, n, d)) * 10.0
    X[rng.random((n, d)) < 0.03] = np.nan
    X[:, 7] = 3.0  # constante: se descarta
    X[:, 9] = np.nan  # todo NA: se descarta
    perm = rng.permutation(n)
    query_idx, base_idx = perm[:25], perm[25:]
    fit_mask = np.zeros(n, bool)
    fit_mask[base_idx] = True
    tf = lambda v: np.log2(np.maximum(v, 0) + 1)  # noqa: E731

    sx = StreamingExactKNN(n, query_idx, base_idx, fit_mask=fit_mask, transform=tf, batch_size=batch_size)
    for fid, j in zip(default_feature_ids(d), range(d)):
        sx.update(fid, X[:, j])
    Z, skipped = _materialize(X, fit_mask, tf)
    assert sx.n_skipped == skipped == 2 and sx.n_features == d - 2

    for metric in ("cosine", "correlation", "l2", "l1"):
        ref_i, ref_v, _ = ExactKNN(metric).fit(Z[base_idx]).query(Z[query_idx], 10)
        idx, val = sx.result(metric, 10)
        np.testing.assert_array_equal(idx, ref_i, err_msg=metric)
        tol = dict(atol=2e-5) if metric in ("cosine", "correlation") else dict(rtol=1e-5)
        np.testing.assert_allclose(val, ref_v, **tol, err_msg=metric)
        # Índices de muestra originales
        gi, _ = sx.result(metric, 10, as_sample_idx=True)
        np.testing.assert_array_equal(gi, base_idx[ref_i])


def test_streaming_matches_dense_scores_and_exclude_self() -> None:
    rng = np.random.default_rng(12)
    n, d = 40, 60
    X = rng.normal(size=(n, d)).astype(np.float32)
    q_idx = np.arange(0, 10)
    b_idx = np.arange(0, n)  # Q ⊂ B
    sx = StreamingExactKNN(n, q_idx, b_idx, metrics=("cosine", "l1"), standardize=False, batch_size=8)
    for j in range(d):
        sx.update(f"g{j}", X[:, j])
    Xd = X.astype(np.float64)
    np.testing.assert_allclose(sx.scores("l1"), cdist(Xd[q_idx], Xd, "cityblock"), rtol=1e-10)
    np.testing.assert_allclose(sx.scores("cosine"), 1 - cdist(Xd[q_idx], Xd, "cosine"), atol=1e-6)
    idx, _ = sx.result("cosine", 3)
    np.testing.assert_array_equal(idx[:, 0], q_idx)  # sin exclusión, el más similar es sí mismo
    idx, _ = sx.result("cosine", 3, exclude_self=True)
    assert not (idx == q_idx[:, None]).any()
    with pytest.raises(ValueError):
        sx.result("l2", 3)  # no se acumuló


def test_streaming_matches_projection_accumulator_rule() -> None:
    """Las normas acumuladas son las mismas que las del acumulador de proyecciones (misma regla)."""
    from tcgaknn.algoritmos.base import StreamingProjectionAccumulator

    rng = np.random.default_rng(13)
    n, d = 30, 80
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d)
    X[rng.random((n, d)) < 0.05] = np.nan
    X[:, 3] = 1.0
    fm = np.arange(n) < 20
    acc = StreamingProjectionAccumulator(n, 8, seed=0, fit_mask=fm)
    sx = StreamingExactKNN(n, np.arange(n), np.arange(n), metrics=("cosine",), fit_mask=fm)
    for j, fid in enumerate(default_feature_ids(d)):
        assert acc.update(fid, X[:, j]) == sx.update(fid, X[:, j])
    acc.raw()
    sx.scores("cosine")
    np.testing.assert_allclose(sx.sq_q, acc.sqnorm, rtol=1e-6)
    assert sx.n_skipped == acc.n_skipped == 1


# --------------------------------------------------------------------------
# Casos borde (revisión adversarial)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "correlation", "l2", "l1"])
def test_edge_sizes(metric: str) -> None:
    """Base vacía, base de una fila, k > n, k = 0 y cero consultas."""
    Q = np.ones((2, 4), np.float32)
    idx, val, st = ExactKNN(metric).fit(np.zeros((0, 4), np.float32)).query(Q, 3)
    assert idx.shape == (2, 3) and (idx == -1).all() and np.isnan(val).all()
    np.testing.assert_array_equal(st["n_candidates"], [0, 0])
    idx, val, _ = ExactKNN(metric).fit(np.arange(4.0)[None]).query(Q, 3)
    assert idx[:, 0].tolist() == [0, 0] and (idx[:, 1:] == -1).all() and np.isnan(val[:, 1:]).all()
    ex = ExactKNN(metric).fit(np.random.default_rng(0).normal(size=(5, 4)))
    idx, val, _ = ex.query(Q, 0)
    assert idx.shape == (2, 0) and val.shape == (2, 0)
    idx, val, st = ex.query(np.zeros((0, 4)), 2)
    assert idx.shape == (0, 2) and idx.dtype == np.int64 and val.dtype == np.float32
    with pytest.raises(ValueError):
        ex.query(Q, -1)
    with pytest.raises(ValueError):
        ex.query(np.ones((1, 5)), 2)


@pytest.mark.parametrize("metric", ["cosine", "correlation", "l2", "l1"])
def test_rejects_nan(metric: str) -> None:
    X = np.random.default_rng(0).normal(size=(5, 4))
    bad = X.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError):
        ExactKNN(metric).fit(bad)
    with pytest.raises(ValueError):
        ExactKNN(metric).fit(X).query(np.full((1, 4), np.inf), 2)
    with pytest.raises(RuntimeError):
        ExactKNN(metric).query(X, 2)


def test_null_rows_have_zero_similarity() -> None:
    X = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0], [2.0, 2.0, 2.0]], np.float32)
    idx, val, _ = ExactKNN("cosine").fit(X).query(np.zeros((1, 3)), 4)
    assert idx[0].tolist() == [0, 1, 2, 3] and (val == 0).all()  # fila nula: todo 0, empate por índice
    # fila constante en correlación (varianza nula) también vale 0
    idx, val, _ = ExactKNN("correlation").fit(X).query(np.array([[1.0, 2.0, 4.0]]), 4)
    sc = dict(zip(idx[0].tolist(), val[0].tolist()))
    assert sc[1] == 0 and sc[3] == 0


def test_streaming_validation() -> None:
    with pytest.raises(ValueError):
        StreamingExactKNN(4, [0], [1], batch_size=0)
    with pytest.raises(ValueError):
        StreamingExactKNN(4, [0], [4])
    sx = StreamingExactKNN(4, [0], [1, 2])
    with pytest.raises(ValueError):
        sx.update("g", np.zeros(3))
