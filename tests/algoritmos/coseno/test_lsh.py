"""Pruebas de LSH por bandas (plan §3.4 y §5.10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcgaknn.algoritmos.base import (
    StreamingProjectionAccumulator,
    default_feature_ids,
    l2_normalize,
    omega_matrix,
    topk,
)
from tcgaknn.algoritmos.coseno.lsh import BandedLSH, band_codes, s_curve, threshold_angle


# --------------------------------------------------------------------------
# Utilidades de las pruebas
# --------------------------------------------------------------------------


def clustered(n: int, d: int, n_clusters: int, seed: int, noise: float = 1.0) -> np.ndarray:
    """Datos con grupos (como tejidos): centro aleatorio + ruido gaussiano."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 1.5
    lab = rng.integers(0, n_clusters, size=n)
    return (centers[lab] + noise * rng.standard_normal((n, d))).astype(np.float32)


def exact_cosine(X: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    return topk(l2_normalize(Q) @ l2_normalize(X).T, k)[0]


def recall_at_k(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Definición del plan (§4.2.1): los -1 (faltantes) cuentan como no recuperados."""
    hits = [len(set(a[:k][a[:k] >= 0].tolist()) & set(e[:k].tolist())) for a, e in zip(approx, exact)]
    return float(np.mean(hits)) / k


def pairs_at_angle(n: int, d: int, theta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n pares (u, v) unitarios con ángulo exactamente θ y orientación aleatoria."""
    U = l2_normalize(rng.standard_normal((n, d)))
    W = rng.standard_normal((n, d)).astype(np.float32)
    W -= np.sum(W * U, axis=1, keepdims=True) * U
    W = l2_normalize(W)
    V = np.cos(theta) * U + np.sin(theta) * W
    return U, V.astype(np.float32)


def naive_candidates(bits_db: np.ndarray, bits_q: np.ndarray, L: int, r: int) -> list[set[int]]:
    """Referencia literal del plan (§5.4): L diccionarios de listas, clave = bytes de la banda."""
    tables = []
    for t in range(L):
        tab: dict[bytes, list[int]] = {}
        keys = np.packbits(bits_db[:, t * r : (t + 1) * r], axis=1)
        for i, key in enumerate(map(bytes, keys)):
            tab.setdefault(key, []).append(i)
        tables.append(tab)
    out = []
    for q in bits_q:
        c: set[int] = set()
        for t in range(L):
            c.update(tables[t].get(bytes(np.packbits(q[t * r : (t + 1) * r])), ()))
        out.append(c)
    return out


# --------------------------------------------------------------------------
# Curva S
# --------------------------------------------------------------------------


def test_s_curve_matches_plan_table() -> None:
    deg = np.deg2rad
    # Valores de la tabla de §3.4 (redondeados a 3 decimales)
    assert s_curve(deg(45), 16, 12) == pytest.approx(0.403, abs=1e-3)
    assert s_curve(deg(30), 16, 12) == pytest.approx(0.851, abs=1e-3)
    assert s_curve(deg(60), 32, 8) == pytest.approx(0.720, abs=1e-3)
    assert s_curve(deg(20), 8, 16) == pytest.approx(0.732, abs=1e-3)
    assert s_curve(deg(90), 64, 4) == pytest.approx(0.984, abs=1e-3)
    assert math.degrees(threshold_angle(16, 12)) == pytest.approx(37, abs=0.5)
    assert math.degrees(threshold_angle(32, 8)) == pytest.approx(63, abs=0.5)
    assert math.degrees(threshold_angle(64, 4)) == pytest.approx(116, abs=0.5)
    th = np.linspace(0, np.pi, 50)
    s = s_curve(th, 16, 12)
    assert s[0] == 1.0 and s[-1] == 0.0 and np.all(np.diff(s) <= 0)
    # Subir r vuelve la curva más selectiva; subir L la eleva.
    # Más bits por banda => curva más selectiva; se compara donde no hay subdesbordamiento a 0 o 1
    inner = (s > 1e-9) & (s < 1 - 1e-9)
    assert np.all(s_curve(th, 16, 16)[inner] < s[inner])
    assert np.all(s_curve(th, 32, 12)[inner] > s[inner])


def test_band_codes_by_hand() -> None:
    bits = np.array([[1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=bool)
    # r = 3: banda 0 = bits 0..2 con pesos 1, 2, 4
    np.testing.assert_array_equal(band_codes(bits, 2, 3), [[5, 3], [0, 7]])
    full = np.ones((1, 64), dtype=bool)
    assert band_codes(full, 1, 64)[0, 0] == np.uint64(2**64 - 1)


@pytest.mark.parametrize("L,r", [(8, 4), (16, 8)])
def test_candidate_probability_matches_s_curve(L: int, r: int) -> None:
    """Monte Carlo: fracción de pares a ángulo θ que resultan candidatos ≈ s_curve(θ).

    Con 600 pares por ángulo (4 semillas de Ω × 150) el error estándar es
    ≤ √(0,25/600) ≈ 0,020; se exige |error| < 0,07 (≈ 3,5 σ) en cada ángulo y
    un error medio < 0,03. Un error de implementación (bandas solapadas, OR en
    vez de AND dentro de la banda, bits mal cortados) desplaza la curva mucho más.
    """
    d, per = 128, 150
    thetas = np.deg2rad([10, 25, 40, 55, 70, 85, 100, 120])
    hits = np.zeros(thetas.size)
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        U, V = zip(*(pairs_at_angle(per, d, th, rng) for th in thetas))
        U, V = np.concatenate(U), np.concatenate(V)
        idx = BandedLSH(L, r, seed=seed, rerank=None).fit(U)
        cands = idx.candidates(V)
        got = np.array([i in set(c.tolist()) for i, c in enumerate(cands)])
        hits += got.reshape(thetas.size, per).sum(axis=1)
    emp = hits / (4 * per)
    theo = s_curve(thetas, L, r)
    assert np.max(np.abs(emp - theo)) < 0.07, (emp, theo)
    assert np.mean(np.abs(emp - theo)) < 0.03, (emp, theo)


# --------------------------------------------------------------------------
# Estructura de tablas y re-ranking
# --------------------------------------------------------------------------


def test_candidates_match_naive_dicts() -> None:
    X = clustered(400, 300, 6, seed=1)
    Q = np.concatenate([clustered(30, 300, 6, seed=2), X[:5]])
    L, r, seed = 8, 6, 3
    idx = BandedLSH(L, r, seed=seed).fit(X)
    Om = omega_matrix(default_feature_ids(300), L * r, seed)
    ref = naive_candidates(l2_normalize(X) @ Om >= 0, l2_normalize(Q) @ Om >= 0, L, r)
    got = idx.candidates(Q)
    sizes = [c.size for c in got]
    assert 0 < np.median(sizes) < 400  # caso no trivial
    for g, e in zip(got, ref):
        assert np.all(np.diff(g) > 0)  # ordenado y sin duplicados
        assert set(g.tolist()) == e


@pytest.mark.parametrize("rerank", ["d", "m", None])
def test_query_equals_bruteforce_on_candidates(rerank: str | None) -> None:
    X = clustered(500, 400, 8, seed=4)
    Q = clustered(40, 400, 8, seed=5)
    k, L, r, m, seed = 10, 6, 8, 64, 9
    idx = BandedLSH(L, r, seed=seed, rerank=rerank, m=m).fit(X)
    I, S, st = idx.query(Q, k)
    assert I.dtype == np.int64 and S.dtype == np.float32 and I.shape == (40, k)
    assert idx.score_kind == ("distance" if rerank == "m" else "similarity")
    cands = idx.candidates(Q)
    np.testing.assert_array_equal(st["n_candidates"], [c.size for c in cands])
    Om = omega_matrix(default_feature_ids(400), max(L * r, m), seed)
    Xn, Qn = l2_normalize(X), l2_normalize(Q)
    for i, c in enumerate(cands):
        if rerank == "d":
            s = Xn[c] @ Qn[i]
            order = np.lexsort((c, -s))
        elif rerank == "m":
            P, p = Xn[c] @ Om[:, :m] / np.sqrt(m), Qn[i] @ Om[:, :m] / np.sqrt(m)
            s = np.linalg.norm(P - p, axis=1)
            order = np.lexsort((c, s))
        else:
            bx, bq = Xn[c] @ Om[:, : L * r] >= 0, Qn[i] @ Om[:, : L * r] >= 0
            eq = bx == bq
            bands = eq.reshape(c.size, L, r).all(axis=2).sum(axis=1)
            ham = (~eq).sum(axis=1)
            s = bands.astype(np.float64)
            order = np.lexsort((c, ham, -bands))
        top = order[:k]
        kk = top.size
        np.testing.assert_array_equal(I[i, :kk], c[top])
        assert (I[i, kk:] == -1).all() and np.isnan(S[i, kk:]).all()
        np.testing.assert_allclose(S[i, :kk], s[top], rtol=1e-4, atol=1e-5)


def test_degenerate_recall() -> None:
    """r = 1, L = 64: P_cand(90°) = 1 − 2⁻⁶⁴, casi todo es candidato ⇒ recall@10 ≥ 0,99."""
    X = clustered(500, 2000, 10, seed=6)
    Q = clustered(50, 2000, 10, seed=7)
    I, _, st = BandedLSH(64, 1, seed=0).fit(X).query(Q, 10)
    assert recall_at_k(I, exact_cosine(X, Q, 10), 10) >= 0.99
    assert np.median(st["n_candidates"]) > 0.95 * 500


@pytest.mark.parametrize("rerank", ["d", "m", None])
def test_duplicate_found(rerank: str | None) -> None:
    """Un duplicado (incluso escalado por c > 0) coincide en todas las bandas: 100 %."""
    X = clustered(400, 300, 5, seed=8)
    X = np.concatenate([X, X[7:8]])  # X[400] es copia exacta de X[7]
    idx = BandedLSH(4, 24, seed=1, rerank=rerank, m=64).fit(X)  # muy selectivo
    q = np.arange(0, 400, 7)
    I, S, st = idx.query(3.7 * X[q], 5)
    cands = idx.candidates(X[q])
    assert all(i in set(c.tolist()) for i, c in zip(q, cands))
    assert np.median(st["n_candidates"]) < 20  # el índice sí es selectivo
    assert set(cands[1].tolist()) >= {7, 400}  # q[1] = 7: ambas copias colisionan
    if rerank == "d":
        np.testing.assert_allclose(S[:, 0], 1.0, atol=1e-5)
    if rerank == "m":
        np.testing.assert_allclose(S[:, 0], 0.0, atol=1e-4)
    if rerank is None:
        np.testing.assert_array_equal(S[:, 0], 4)
    first = I[:, 0]
    assert np.all((first == q) | ((q == 7) & (first == 400)))


def test_short_candidates_pad_and_recall() -> None:
    X = clustered(300, 200, 4, seed=10)
    rng = np.random.default_rng(11)
    Q = np.stack([X[3], rng.standard_normal(200).astype(np.float32)])
    k = 5
    idx = BandedLSH(1, 32, seed=2).fit(X)  # una sola banda de 32 bits
    I, S, st = idx.query(Q, k)
    np.testing.assert_array_equal(I, [[3, -1, -1, -1, -1], [-1] * 5])
    assert np.isnan(S[0, 1:]).all() and np.isnan(S[1]).all()
    np.testing.assert_array_equal(st["n_candidates"], [1, 0])
    np.testing.assert_array_equal(st["short"], [True, True])
    np.testing.assert_array_equal(st["empty"], [False, True])
    # Faltantes = no recuperados: q0 recupera 1 de 5 y q1 ninguno
    assert recall_at_k(I, exact_cosine(X, Q, k), k) == pytest.approx((1 / 5 + 0) / 2)

    # LSH+fallback: completa con los k más cercanos en Hamming y re-ordena por coseno
    fb = BandedLSH(1, 32, seed=2, fallback=True).fit(X)
    I2, S2, st2 = fb.query(Q, k)
    assert (I2 >= 0).all() and I2[0, 0] == 3
    np.testing.assert_array_equal(st2["n_candidates"], [1, 0])
    assert st2["n_fallback"][1] == k and st2["n_fallback"][0] >= k - 1
    Om = omega_matrix(default_feature_ids(200), 32, 2)
    bx, bq = l2_normalize(X) @ Om >= 0, l2_normalize(Q) @ Om >= 0
    for i, base_c in enumerate([{3}, set()]):
        ham = (bx != bq[i]).sum(axis=1)
        near = set(np.lexsort((np.arange(300), ham))[:k].tolist())
        c = np.array(sorted(base_c | near))
        s = l2_normalize(X)[c] @ l2_normalize(Q)[i]
        np.testing.assert_array_equal(I2[i], c[np.lexsort((c, -s))][:k])


# --------------------------------------------------------------------------
# Streaming, inserción y anidamiento
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rerank", ["d", "m", None])
def test_fit_bits_equivalent_to_fit(rerank: str | None) -> None:
    n, d, L, r, m, seed = 300, 250, 8, 8, 64, 4
    X = clustered(n, d, 5, seed=12)
    Q = clustered(30, d, 5, seed=13)
    ref = BandedLSH(L, r, seed=seed, rerank=rerank, m=m).fit(X)
    I0, S0, st0 = ref.query(Q, 10)

    # Desde el acumulador en streaming (sin estandarizar: Z = X), fila de rasgo a fila de rasgo
    ids = default_feature_ids(d)
    acc = StreamingProjectionAccumulator(n, max(L * r, m), seed, standardize=False, batch_size=32)
    for j in range(d):
        acc.update(ids[j], X[:, j])
    bits = acc.sign_bits(L * r)
    # Signo calculado en float64 (acumulador) vs float32 (fit): solo pueden diferir valores ~0
    Xbits, _ = ref.hash(X)
    assert np.mean(bits == Xbits) > 0.9999
    data = {"d": X, "m": acc.projections(m), None: None}[rerank]
    idx = BandedLSH(L, r, seed=seed, rerank=rerank, m=m).fit_bits(bits, rerank_data=data)
    I1, S1, st1 = idx.query(Q, 10)
    np.testing.assert_array_equal(I1, I0)
    np.testing.assert_allclose(S1, S0, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(st1["n_candidates"], st0["n_candidates"])

    # Consultas ya hasheadas
    qb, qx = ref.hash(Q)
    I2, S2, _ = idx.query_bits(qb, 10, rerank_data=None if rerank is None else (Q if rerank == "d" else qx))
    np.testing.assert_array_equal(I2, I0)


def test_insert_equals_fit() -> None:
    X = clustered(600, 150, 6, seed=14)
    Q = np.concatenate([clustered(25, 150, 6, seed=15), X[450:455]])
    for rerank in ["d", "m", None]:
        full = BandedLSH(10, 6, seed=5, rerank=rerank, m=32, fallback=True).fit(X)
        I0, S0, st0 = full.query(Q, 8)
        inc = BandedLSH(10, 6, seed=5, rerank=rerank, m=32, fallback=True).fit(X[:400])
        np.testing.assert_array_equal(inc.insert(X[400:500]), np.arange(400, 500))
        inc.insert(X[500:])
        assert inc.n == 600 and inc.n_built == 400  # en el búfer pendiente
        for _ in range(2):
            I1, S1, st1 = inc.query(Q, 8)
            np.testing.assert_array_equal(I1, I0)
            np.testing.assert_allclose(S1, S0, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(st1["n_candidates"], st0["n_candidates"])
            inc.rebuild()  # segunda vuelta: ya fundido en las tablas
        assert inc.n_built == 600
    # La refusión automática ocurre cuando el búfer supera max(1.024, n_built/4)
    Y = clustered(1600, 32, 4, seed=16)
    big = BandedLSH(4, 8, seed=0).fit(Y[:100])
    big.insert(Y[100:1100])
    assert big.n_built == 100
    big.insert(Y[1100:1200])
    assert big.n_built == 1200
    np.testing.assert_array_equal(big.bucket_sizes(0).sum(), 1200)


def test_nested_with_simhash_and_rp() -> None:
    """Bits de LSH = primeros L·r bits de SimHash con Ω de 1.024 columnas; re-ranking m = RP."""
    X = clustered(100, 120, 3, seed=17)
    L, r, m, seed = 16, 12, 256, 6
    idx = BandedLSH(L, r, seed=seed, rerank="m", m=m)
    bits, P = idx.hash(X)
    Y = l2_normalize(X) @ omega_matrix(default_feature_ids(120), 1024, seed)
    clear = np.abs(Y[:, : L * r]) > 1e-4
    np.testing.assert_array_equal(bits[clear], (Y[:, : L * r] >= 0)[clear])
    np.testing.assert_allclose(P, Y[:, :m] / np.sqrt(m), rtol=1e-4, atol=1e-5)


def test_bucket_sizes_and_determinism() -> None:
    X = clustered(300, 100, 3, seed=18)
    a = BandedLSH(6, 5, seed=3).fit(X)
    sizes = [a.bucket_sizes(t) for t in range(6)]
    assert all(s.sum() == 300 and (s > 0).all() and s.size <= 2**5 for s in sizes)
    assert a.bucket_sizes().size == sum(s.size for s in sizes)
    summ = a.bucket_summary()
    assert summ["n_buckets"] == a.bucket_sizes().size and summ["max"] >= summ["median"]
    b = BandedLSH(6, 5, seed=3).fit(X)
    c = BandedLSH(6, 5, seed=4).fit(X)
    Q = clustered(20, 100, 3, seed=19)
    np.testing.assert_array_equal(a.query(Q, 5)[0], b.query(Q, 5)[0])
    assert not np.array_equal(a.codes, c.codes)
    ib = a.index_bytes()
    assert ib["codes"] == 300 * 6 * 8 and ib["rerank_data"] == 300 * 100 * 4


# --------------------------------------------------------------------------
# Casos borde (revisión adversarial)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("rerank", ["d", "m", None])
def test_edge_sizes(rerank: str | None, fallback: bool) -> None:
    """n = 0, n = 1 y k > n: forma (nq, k), relleno -1/NaN y nunca índices repetidos."""
    rng = np.random.default_rng(20)
    X = rng.standard_normal((5, 20)).astype(np.float32)
    kw = dict(rerank=rerank, m=8, fallback=fallback)
    I, S, st = BandedLSH(4, 3, **kw).fit(X[:0]).query(X[:2], 3)
    assert (I == -1).all() and np.isnan(S).all() and (st["n_candidates"] == 0).all()
    I, S, _ = BandedLSH(4, 3, **kw).fit(X[:1]).query(X[:2], 3)
    assert I[0, 0] == 0 and (I[:, 1:] == -1).all() and np.isnan(S[:, 1:]).all()
    if fallback:
        assert I[1, 0] == 0
    I, S, _ = BandedLSH(4, 3, **kw).fit(X).query(X, 10)
    assert I.shape == (5, 10) and I.dtype == np.int64 and S.dtype == np.float32
    assert (I[:, 5:] == -1).all()
    for row, srow in zip(I, S):
        v = row[row >= 0]
        assert v.size == np.unique(v).size and np.isfinite(srow[: v.size]).all()
        if fallback:
            assert v.size == 5  # con respaldo, k > n devuelve toda la base


def test_null_and_nan_rows_do_not_crash() -> None:
    rng = np.random.default_rng(21)
    X = rng.standard_normal((6, 30)).astype(np.float32)
    X[0] = 0.0
    X[1, 4] = np.nan
    I, S, _ = BandedLSH(4, 3, fallback=True).fit(X).query(X, 3)
    assert (I >= 0).all() and not np.isnan(S).any()


def test_dimension_mismatch_raises() -> None:
    X = clustered(50, 40, 2, seed=22)
    idx = BandedLSH(4, 4).fit(X)
    with pytest.raises(ValueError):
        idx.query(X[:, :30], 3)
    with pytest.raises(ValueError):
        idx.insert(X[:, :30])
    idx.fit(X[:, :30])  # un nuevo fit sí puede cambiar d
    assert idx.query(X[:3, :30], 2)[0].shape == (3, 2)
