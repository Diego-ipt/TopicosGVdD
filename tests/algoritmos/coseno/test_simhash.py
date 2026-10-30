from __future__ import annotations

import numpy as np
import pytest

from tcgaknn.algoritmos.base import (
    StreamingProjectionAccumulator,
    default_feature_ids,
    l2_normalize,
    omega_matrix,
    topk,
)
from tcgaknn.algoritmos.coseno.simhash import (
    SimHashKNN,
    angle_estimate,
    cosine_estimate,
    hamming,
    hamming_matrix,
    pack_bits,
    sign_bits,
    signatures,
    unpack_bits,
)


def _pair_at_angle(rng: np.random.Generator, d: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    w = rng.standard_normal(d)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    return u, np.cos(theta) * u + np.sin(theta) * w


def _clustered(seed: int, n: int, nq: int, d: int, n_clusters: int = 20, noise: float = 1.5):
    rng = np.random.default_rng(seed)
    cent = rng.standard_normal((n_clusters, d))
    lab = rng.integers(0, n_clusters, n + nq)
    X = (cent[lab] + noise * rng.standard_normal((n + nq, d))).astype(np.float32)
    return X[:n], X[n:]


def _recall(idx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(r)) / ref.shape[1] for a, r in zip(idx, ref)]))


# --------------------------------------------------------------------------
# Empaquetado y Hamming
# --------------------------------------------------------------------------


@pytest.mark.parametrize("b", [64, 128, 320])
def test_pack_roundtrip(b: int) -> None:
    rng = np.random.default_rng(b)
    bits = rng.random((37, b)) < 0.5
    P = pack_bits(bits)
    assert P.dtype == np.uint64 and P.shape == (37, b // 64)
    np.testing.assert_array_equal(unpack_bits(P, b), bits)
    np.testing.assert_array_equal(unpack_bits(P), bits)


def test_pack_layout_little_endian() -> None:
    # bit j de la firma = bit (j mod 64) de la palabra j // 64
    for j in [0, 1, 7, 8, 63, 64, 100, 127]:
        bits = np.zeros((1, 128), dtype=bool)
        bits[0, j] = True
        P = pack_bits(bits)[0]
        assert P[j // 64] == np.uint64(1) << np.uint64(j % 64)
        assert P[1 - j // 64] == 0
    with pytest.raises(ValueError):
        pack_bits(np.zeros((2, 100), dtype=bool))


def test_hamming_vs_count_nonzero() -> None:
    rng = np.random.default_rng(1)
    b = 256
    Qb = rng.random((13, b)) < 0.5
    Db = rng.random((50, b)) < 0.3
    ref = np.array([[np.count_nonzero(q != x) for x in Db] for q in Qb])
    Qp, Dp = pack_bits(Qb), pack_bits(Db)
    np.testing.assert_array_equal(hamming_matrix(Qp, Dp), ref)
    # bloques diminutos: el resultado no depende del tamaño de bloque
    np.testing.assert_array_equal(hamming_matrix(Qp, Dp, block_elems=7), ref)
    np.testing.assert_array_equal(hamming(Qp[3], Dp), ref[3])
    np.testing.assert_array_equal(hamming(Qp, Dp), ref)


def test_sign_bits_definition_and_omega_forms() -> None:
    rng = np.random.default_rng(2)
    n, d, b = 30, 40, 128
    X = rng.standard_normal((n, d)).astype(np.float32)
    ids = default_feature_ids(d)
    O = omega_matrix(ids, 192, seed=5)  # más columnas que b: se usan las primeras b
    ref = (X.astype(np.float64) @ O[:, :b].astype(np.float64)) >= 0
    np.testing.assert_array_equal(sign_bits(X, O, b), ref)
    np.testing.assert_array_equal(sign_bits(X, (ids, 5), b), ref)
    # escalar filas por un positivo no cambia la firma
    np.testing.assert_array_equal(sign_bits(X * 3.7, O, b), ref)
    with pytest.raises(ValueError):
        sign_bits(X, O, 100)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        sign_bits(Xn, O, b)


# --------------------------------------------------------------------------
# Propiedad de Charikar
# --------------------------------------------------------------------------


def test_angle_estimate() -> None:
    """Plan §5.10: π·Ham/b aproxima θ con error absoluto medio < 0,05 rad (b = 1.024).

    Desviación teórica de θ̂: π·√(p(1−p)/b) ≤ π·0,5/32 ≈ 0,049 rad (p = 1/2), y
    el error absoluto medio de una normal es ≈ 0,8σ, así que el promedio sobre
    θ ∈ [0,1; 3,0] queda bastante por debajo de 0,05.
    """
    rng = np.random.default_rng(0)
    d, b = 500, 1024
    Omega = omega_matrix(default_feature_ids(d), b, seed=1)
    errs = []
    for theta in np.linspace(0.1, 3.0, 20):
        u, v = _pair_at_angle(rng, d, theta)
        su, sv = signatures(np.stack([u, v]).astype(np.float32), Omega, b)
        est = angle_estimate(hamming(su, sv[None, :])[0], b)
        errs.append(abs(est - theta))
    assert np.mean(errs) < 0.05
    assert np.isclose(cosine_estimate(256, 1024), np.cos(np.pi / 4))
    assert angle_estimate(0, 64) == 0.0 and np.isclose(angle_estimate(64, 64), np.pi)


@pytest.mark.parametrize("theta", [0.3, np.pi / 4, np.pi / 2, 2.2, 2.9])
def test_bit_collision_probability(theta: float) -> None:
    """Monte Carlo de P[bit igual] = 1 − θ/π.

    40 pares × 2.048 hiperplanos = 81.920 bits; σ = √(p(1−p)/N) ≤ 0,0018. Se
    exige |p̂ − p| < 0,01 (≥ 5σ): falla si la regla de signo o Ω están mal
    (p. ej. bits sesgados darían p̂ ≠ 1 − θ/π para θ = π/2).
    """
    rng = np.random.default_rng(int(theta * 1000))
    d, b, pairs = 200, 2048, 40
    Omega = omega_matrix(default_feature_ids(d), b, seed=11)
    eq = 0
    for _ in range(pairs):
        u, v = _pair_at_angle(rng, d, theta)
        bits = sign_bits(np.stack([u, v]), Omega, b)
        eq += np.count_nonzero(bits[0] == bits[1])
    p_hat = eq / (pairs * b)
    assert abs(p_hat - (1 - theta / np.pi)) < 0.01


# --------------------------------------------------------------------------
# Ω compartida: anidamiento y streaming
# --------------------------------------------------------------------------


def test_nested_omega() -> None:
    """Los primeros b bits con b pequeño coinciden con los de un b mayor (misma Ω)."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 80)).astype(np.float32)
    small = SimHashKNN(b=128, seed=9).fit(X)
    big = SimHashKNN(b=512, seed=9).fit(X)
    np.testing.assert_array_equal(unpack_bits(small.packed, 128), unpack_bits(big.packed, 512)[:, :128])
    np.testing.assert_array_equal(small.Omega, big.Omega[:, :128])
    other = SimHashKNN(b=128, seed=10).fit(X)
    assert not np.array_equal(other.packed, small.packed)


def test_consistent_with_streaming_accumulator() -> None:
    """Firmas en memoria = ``StreamingProjectionAccumulator.sign_bits(b)`` (misma semilla e IDs)."""
    rng = np.random.default_rng(5)
    n, d, cols, b, seed = 120, 400, 384, 256, 21
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + rng.normal(size=d)
    X[rng.random((n, d)) < 0.02] = np.nan
    ids = [f"GEN{j}|{1000 + j}" for j in range(d)]

    acc = StreamingProjectionAccumulator(n, cols, seed=seed, batch_size=64)
    for j in range(d):
        acc.update(ids[j], X[:, j])

    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    Z = np.nan_to_num((X - mu) / sd).astype(np.float32)
    mem = SimHashKNN(b=b, seed=seed, feature_ids=ids).fit(Z)
    stream_bits = acc.sign_bits(b)
    np.testing.assert_array_equal(unpack_bits(mem.packed, b), stream_bits)

    # Índice construido desde las firmas del streaming: mismas respuestas.
    fs = SimHashKNN.from_signatures(pack_bits(acc.sign_bits(cols)), b, seed=seed, feature_ids=ids)
    i1, s1, _ = mem.query(Z[:20], 7)
    i2, s2, _ = fs.query_packed(pack_bits(stream_bits[:20]), 7)
    i3, s3, _ = fs.query(Z[:20], 7)  # firma consultas crudas con Ω regenerada por ID
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(i1, i3)
    np.testing.assert_array_equal(s1, s2)

    # Con re-ranking sobre las proyecciones RP del mismo acumulador (plan §3.4, rerank="m")
    P = acc.projections(cols)
    fr = SimHashKNN.from_signatures(pack_bits(stream_bits), b, base=P, rerank_factor=n)
    ir, sr, _ = fr.query_packed(pack_bits(stream_bits[:20]), 7, Q_base=P[:20])
    ref_idx, ref_s = topk(l2_normalize(P[:20]) @ l2_normalize(P).T, 7)
    np.testing.assert_array_equal(ir, ref_idx)
    np.testing.assert_allclose(sr, ref_s, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------


def test_self_query_ties_padding_and_stats() -> None:
    rng = np.random.default_rng(6)
    n, d, b = 40, 60, 128
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[25] = X[3] * 2.0  # misma dirección que la fila 3: Hamming 0 con ella
    m = SimHashKNN(b=b, seed=3).fit(X)
    idx, s, st = m.query(X[[3, 25, 10]], 3)
    assert idx[0, :2].tolist() == [3, 25]  # empate en Hamming 0: índice menor primero
    assert idx[1, :2].tolist() == [3, 25]
    assert idx[2, 0] == 10
    np.testing.assert_allclose(s[:, 0], 1.0)
    # el score es cos(π·Ham/b) del vecino devuelto
    H = hamming_matrix(signatures(X[[10]], m.Omega, b), m.packed)[0]
    np.testing.assert_allclose(s[2], np.cos(np.pi * H[idx[2]] / b), rtol=1e-6)
    assert idx.dtype == np.int64 and s.dtype == np.float32
    assert st["n_candidates"].tolist() == [n] * 3 and st["n_candidates"].dtype == np.int64
    assert st["index_bytes"] == n * b // 8 and st["rerank"] is False

    idx, s, _ = m.query(X[:2], n + 5)  # k > n: relleno
    assert (idx[:, n:] == -1).all() and np.isnan(s[:, n:]).all()
    assert sorted(idx[0, :n].tolist()) == list(range(n))

    mr = SimHashKNN(b=b, rerank_factor=2, seed=3).fit(X)
    idx, s, st = mr.query(X[:2], n + 5)  # f·k ≥ n: re-ranking de todos
    assert (idx[:, n:] == -1).all() and np.isnan(s[:, n:]).all()
    assert st["rerank"] is True and st["base_bytes"] == n * d * 4
    assert st["n_reranked"].tolist() == [n, n]


def test_rerank_with_all_candidates_is_exact() -> None:
    """Si f·k ≥ n el re-ranking examina todo: debe coincidir con el coseno exacto."""
    B, Q = _clustered(7, n=300, nq=25, d=50)
    m = SimHashKNN(b=64, rerank_factor=30, seed=2).fit(B)
    idx, s, _ = m.query(Q, 10)
    ref_idx, ref_s = topk(l2_normalize(Q) @ l2_normalize(B).T, 10)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(s, ref_s, rtol=1e-5, atol=1e-6)


def test_recall_increases_with_b_and_rerank() -> None:
    """recall@10 contra el k-NN exacto por coseno en datos sintéticos con clústeres.

    Valores observados (semilla fija): b = 64, 256, 1.024 → ≈ 0,14, 0,31, 0,49
    sin re-ranking; b = 256 con f = 10 → ≈ 0,96. Con 100 consultas × 10 vecinos
    el error estándar de recall es ≈ 0,015, así que márgenes de 0,05 entre
    niveles de b y de 0,3 para el re-ranking son holgados pero significativos.
    """
    B, Q = _clustered(3, n=2000, nq=100, d=300)
    ref, _ = topk(l2_normalize(Q) @ l2_normalize(B).T, 10)
    r = {b: _recall(SimHashKNN(b=b, seed=1).fit(B).query(Q, 10)[0], ref) for b in (64, 256, 1024)}
    assert r[64] + 0.05 < r[256] < r[1024] - 0.05
    r_rr = _recall(SimHashKNN(b=256, rerank_factor=10, seed=1).fit(B).query(Q, 10)[0], ref)
    assert r_rr > r[256] + 0.3 and r_rr > 0.85
    r_rr5 = _recall(SimHashKNN(b=256, rerank_factor=5, seed=1).fit(B).query(Q, 10)[0], ref)
    assert r[256] < r_rr5 < r_rr


def test_determinism() -> None:
    B, Q = _clustered(8, n=200, nq=10, d=40)
    a = SimHashKNN(b=128, rerank_factor=3, seed=4).fit(B).query(Q, 5)
    c = SimHashKNN(b=128, rerank_factor=3, seed=4).fit(B).query(Q, 5)
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_array_equal(a[1], c[1])


@pytest.mark.parametrize("f", [0, 2])
def test_empty_base_and_degenerate_k(f: int) -> None:
    """Base vacía, k = 0 y nq = 0 devuelven formas correctas con relleno."""
    X = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
    idx, sc, st = SimHashKNN(64, f).fit(X[:0]).query(X[:2], 3)
    assert idx.shape == (2, 3) and (idx == -1).all() and np.isnan(sc).all()
    m = SimHashKNN(64, f).fit(X)
    assert m.query(X[:2], 0)[0].shape == (2, 0)
    assert m.query(X[:0], 3)[0].shape == (0, 3)
    idx, sc, _ = m.query(X[:1], 10)
    assert idx.dtype == np.int64 and sc.dtype == np.float32
    assert sorted(idx[0, :5]) == list(range(5)) and (idx[0, 5:] == -1).all()
