from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.stats import binom

from tcgaknn.algoritmos.base import KNNIndex, topk
from tcgaknn.algoritmos.jaccard.minhash import (
    EMPTY,
    MERSENNE_P,
    MinHash,
    MinHashLSH,
    StreamingMinHash,
    StreamingWeightedMinHash,
    WeightedMinHash,
    exact_jaccard,
    jaccard_estimate,
    lsh_threshold,
    mulmod_p,
    s_curve,
    token_hash,
    weighted_jaccard,
    weighted_jaccard_estimate,
)
from tcgaknn.stream import Record


def _pair_with_jaccard(rng: np.random.Generator, union: int, inter: int) -> tuple[np.ndarray, np.ndarray]:
    """Dos conjuntos de hashes aleatorios con |A ∪ B| = union y |A ∩ B| = inter (exacto)."""
    v = rng.integers(0, 2**64 - 1, size=union, dtype=np.uint64, endpoint=False)
    assert np.unique(v).size == union
    half = (union - inter) // 2
    common, only_a, only_b = v[:inter], v[inter : inter + half], v[inter + half :]
    return np.concatenate([common, only_a]), np.concatenate([common, only_b])


def _expected_abs_error(J: float, K: int) -> float:
    """E|Bin(K, J)/K − J|: error absoluto esperado de un estimador MinHash ideal."""
    k = np.arange(K + 1)
    return float((binom.pmf(k, K, J) * np.abs(k / K - J)).sum())


# --------------------------------------------------------------------------
# Hash y aritmética modular
# --------------------------------------------------------------------------


def test_token_hash_stable_blake2b() -> None:
    ref = int.from_bytes(hashlib.blake2b(b"TP53", digest_size=8).digest(), "little")
    assert token_hash("TP53") == ref == token_hash(b"TP53")
    assert token_hash("TP53") != token_hash("TP63")
    assert 0 <= token_hash("x") < 2**64


def test_mulmod_matches_python_ints() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, MERSENNE_P, 5000, dtype=np.uint64)
    x = rng.integers(0, MERSENNE_P, 5000, dtype=np.uint64)
    edge = [0, 1, 2**29 - 1, 2**32 - 1, 2**32, 2**33 + 7, 2**60, MERSENNE_P - 1]
    a = np.concatenate([a, np.repeat(edge, len(edge)).astype(np.uint64)])
    x = np.concatenate([x, np.tile(edge, len(edge)).astype(np.uint64)])
    got = mulmod_p(a, x)
    ref = [int(ai) * int(xi) % MERSENNE_P for ai, xi in zip(a, x)]
    assert got.tolist() == ref


def test_hash_values_match_universal_formula() -> None:
    mh = MinHash(64, seed=3)
    toks = [f"g{i}" for i in range(50)] + ["TP53", "", "ñandú"]
    H = mh.hash_values(np.array([token_hash(t) for t in toks], dtype=np.uint64))
    for j, t in enumerate(toks):
        x = token_hash(t) % MERSENNE_P
        ref = [(int(a) * x + int(b)) % MERSENNE_P for a, b in zip(mh.a, mh.b)]
        assert H[j].tolist() == ref
    sig = mh.signature(toks + toks[:10])  # repetidos no cambian la firma
    np.testing.assert_array_equal(sig, H.min(axis=0))


# --------------------------------------------------------------------------
# Estimador de Jaccard
# --------------------------------------------------------------------------


def test_identical_disjoint_and_empty_sets() -> None:
    mh = MinHash(256, seed=1)
    A = {f"gen{i}" for i in range(80)}
    B = {f"otro{i}" for i in range(80)}
    assert jaccard_estimate(mh.signature(A), mh.signature(set(A))) == 1.0
    # hashes distintos mód p ⇒ h_i distintos (h es biyectiva en Z_p): el estimador da 0 exacto
    assert jaccard_estimate(mh.signature(A), mh.signature(B)) == 0.0
    e = mh.signature(set())
    assert (e == EMPTY).all()
    assert exact_jaccard(set(), set()) == 1.0 == jaccard_estimate(e, mh.signature([]))
    assert exact_jaccard(set(), A) == 0.0 == jaccard_estimate(e, mh.signature(A))
    assert mh.signatures([]).shape == (0, 256)


def test_estimate_error_matches_binomial_monte_carlo() -> None:
    """El error se comporta como Bin(K, J)/K: insesgado y con E|error| = O(1/√K)."""
    rng = np.random.default_rng(42)
    K, trials, U = 128, 120, 100
    mh = MinHash(K, seed=7)
    errs, expected = [], []
    for inter in range(10, 100, 10):  # J = 10/100 … 90/100 (U − I par)
        J = inter / U
        for _ in range(trials):
            a, b = _pair_with_jaccard(rng, U, inter)
            errs.append(jaccard_estimate(mh.signature_from_hashes(a), mh.signature_from_hashes(b)) - J)
            expected.append(_expected_abs_error(J, K))
    errs = np.array(errs)
    # Sesgo: |media| ≤ 4 errores estándar (σ por par ≤ 0,5/√K).
    assert abs(errs.mean()) < 4 * 0.5 / np.sqrt(K) / np.sqrt(errs.size)
    # E|error| empírico vs teórico: con 1.080 pares el error relativo de la media es ≈ 2–3 %.
    ratio = np.abs(errs).mean() / np.mean(expected)
    assert 0.85 < ratio < 1.15, ratio


def test_error_scales_as_inverse_sqrt_k() -> None:
    rng = np.random.default_rng(5)
    small, large = MinHash(32, seed=1), MinHash(512, seed=1)
    e_s, e_l = [], []
    for _ in range(300):
        a, b = _pair_with_jaccard(rng, 60, 30)  # J = 0,5
        e_s.append(abs(jaccard_estimate(small.signature_from_hashes(a), small.signature_from_hashes(b)) - 0.5))
        e_l.append(abs(jaccard_estimate(large.signature_from_hashes(a), large.signature_from_hashes(b)) - 0.5))
    ratio = np.mean(e_s) / np.mean(e_l)
    assert 3.0 < ratio < 5.3, ratio  # teórico √(512/32) = 4


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def test_streaming_equals_batch_and_is_order_independent() -> None:
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(300)]
    samples = [f"TCGA-{i:02d}" for i in range(25)]
    keys = rng.choice(samples, size=2000)
    toks = rng.choice(genes, size=2000)  # con repetidos
    truth: dict[str, set[str]] = {}
    for k, t in zip(keys, toks):
        truth.setdefault(str(k), set()).add(str(t))
    mh = MinHash(96, seed=11)

    results = []
    for order_seed in range(3):
        perm = np.random.default_rng(order_seed).permutation(len(keys))
        smh = StreamingMinHash(96, seed=11, cache_tokens=order_seed != 1)
        for i in perm:
            smh.update(str(keys[i]), str(toks[i]))
        results.append(smh.signatures())
    smh = StreamingMinHash(96, seed=11)
    half = len(keys) // 2
    smh.update_many([str(k) for k in keys[:half]], [str(t) for t in toks[:half]])
    smh.update_many([str(k) for k in keys[half:]], [str(t) for t in toks[half:]])
    results.append(smh.signatures())

    for skeys, S in results:
        assert sorted(skeys) == sorted(truth) and S.dtype == np.uint64
        for k, row in zip(skeys, S):
            np.testing.assert_array_equal(row, mh.signature(truth[k]))


def test_streaming_from_event_records() -> None:
    def ev(sample: str, gene: str) -> Record:
        return Record("mutations", "event", "mc3.maf.gz", key=sample, fields={"Hugo_Symbol": gene})

    recs = [ev("S1", "TP53"), ev("S2", "KRAS"), ev("S1", "PIK3CA"), ev("S1", "TP53"), ev("S3", ""),
            Record("mutations", "schema", "mc3.maf.gz"), ev("S2", "TP53")]
    smh = StreamingMinHash(64, seed=2)
    used = [smh.update_record(r) for r in recs]
    assert used == [True, True, True, True, False, False, True]
    keys, S = smh.signatures()
    assert keys == ["S1", "S2", "S3"]
    mh = MinHash(64, seed=2)
    np.testing.assert_array_equal(S[0], mh.signature({"TP53", "PIK3CA"}))
    np.testing.assert_array_equal(S[1], mh.signature({"KRAS", "TP53"}))
    assert (S[2] == EMPTY).all()  # muestra sin mutaciones válidas = conjunto vacío


# --------------------------------------------------------------------------
# LSH por bandas
# --------------------------------------------------------------------------


def test_s_curve_formula_matches_mmds_table() -> None:
    # MMDS, cap. 3, figura 3.7 (b = 20, r = 5).
    s = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    book = np.array([0.006, 0.047, 0.186, 0.470, 0.802, 0.975, 0.9996])
    np.testing.assert_allclose(s_curve(s, 20, 5), book, atol=6e-4)
    assert abs(lsh_threshold(20, 5) - 0.5493) < 1e-3
    assert MinHashLSH(20, 5).s_curve(0.0) == 0.0 and MinHashLSH(20, 5).s_curve(1.0) == 1.0


def test_candidate_probability_follows_s_curve() -> None:
    rng = np.random.default_rng(9)
    bands, rows, trials, U = 20, 5, 200, 100
    lsh_seed = 4
    mh = MinHash(bands * rows, seed=lsh_seed)
    for inter in (20, 30, 40, 50, 60, 70, 80):
        s = inter / U
        A, B = zip(*(_pair_with_jaccard(rng, U, inter) for _ in range(trials)))
        SA = np.stack([mh.signature_from_hashes(a) for a in A])
        SB = np.stack([mh.signature_from_hashes(b) for b in B])
        lsh = MinHashLSH(bands, rows, seed=lsh_seed).fit(SB)
        hit = np.mean([i in set(lsh.candidates(SA[i]).tolist()) for i in range(trials)])
        p = s_curve(s, bands, rows)
        tol = 4 * np.sqrt(p * (1 - p) / trials) + 0.02
        assert abs(hit - p) <= tol, (s, hit, p)


def _random_gene_sets(rng: np.random.Generator, n: int) -> list[set[str]]:
    """Conjuntos con estructura de grupos (prototipos con ruido), como perfiles de mutación."""
    protos = [set(rng.choice(400, size=40, replace=False).tolist()) for _ in range(5)]
    out = []
    for i in range(n):
        base = protos[i % 5]
        keep = {g for g in base if rng.random() < 0.8}
        extra = set(rng.choice(400, size=8).tolist())
        out.append({f"G{g}" for g in keep | extra})
    return out


def test_lsh_query_matches_bruteforce_when_all_are_candidates() -> None:
    rng = np.random.default_rng(1)
    base = _random_gene_sets(rng, 120)
    queries = _random_gene_sets(rng, 15)
    # r = 1 y 128 bandas: P[candidato] = 1 − (1 − s)^128 ≈ 1 para s ≥ 0,1.
    lsh = MinHashLSH(bands=128, rows=1, seed=0).fit(base)
    assert isinstance(lsh, KNNIndex) and lsh.score_kind == "similarity"
    idx, sc, stats = lsh.query(queries, k=10)
    assert idx.dtype == np.int64 and sc.dtype == np.float32 and stats["rerank_space"] == "jaccard_exact"
    exact = np.array([[exact_jaccard(q, b) for b in base] for q in queries])
    ref_idx, ref_val = topk(exact, 10)
    np.testing.assert_allclose(sc, ref_val, rtol=1e-6)
    np.testing.assert_array_equal(idx, ref_idx)
    assert (stats["n_candidates"] >= 10).all()


def test_lsh_estimate_mode_duplicates_and_padding() -> None:
    rng = np.random.default_rng(2)
    base = _random_gene_sets(rng, 60)
    mh = MinHash(16 * 4, seed=5)
    S = mh.signatures(base)
    lsh = MinHashLSH(bands=16, rows=4, seed=5).fit(S)
    idx, sc, stats = lsh.query(S[[3, 17]], k=5)
    assert stats["rerank_space"] == "jaccard_estimate"
    assert idx[:, 0].tolist() == [3, 17] and (sc[:, 0] == 1.0).all()  # el duplicado colisiona y gana
    # Los scores son el Jaccard estimado con la firma completa
    np.testing.assert_allclose(sc[0, idx[0] >= 0], jaccard_estimate(S[3], S[idx[0][idx[0] >= 0]]), rtol=1e-6)
    # Consulta sin candidatos: todo relleno; con pocos candidatos: -1 y NaN al final
    lone = mh.signature({"NUNCA_VISTO_1", "NUNCA_VISTO_2"})
    idx, sc, stats = lsh.query(lone, k=3)
    assert idx.tolist() == [[-1, -1, -1]] and np.isnan(sc).all() and stats["n_candidates"].tolist() == [0]
    idx, sc, stats = MinHashLSH(16, 4, seed=5).fit(S[:2]).query(S[[0]], k=4)
    assert idx[0, 0] == 0 and (idx[0, stats["n_candidates"][0]:] == -1).all()
    assert np.isnan(sc[0, stats["n_candidates"][0]:]).all()


def test_lsh_empty_sets_and_errors() -> None:
    base = [{"A", "B"}, set(), {"C"}, set()]
    lsh = MinHashLSH(8, 2, seed=0).fit(base)
    idx, sc, _ = lsh.query([set()], k=3)
    assert idx[0, :2].tolist() == [1, 3] and sc[0, :2].tolist() == [1.0, 1.0] and idx[0, 2] == -1
    with pytest.raises(ValueError):
        MinHashLSH(8, 2, rerank="exact").fit(MinHash(16).signatures(base)).query([set()], k=1)
    with pytest.raises(ValueError):
        MinHashLSH(8, 2).fit(np.zeros((3, 10), dtype=np.uint64))


# --------------------------------------------------------------------------
# MinHash ponderado (ICWS)
# --------------------------------------------------------------------------


def test_weighted_jaccard_exact() -> None:
    assert weighted_jaccard({"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}) == pytest.approx(1 / 6)
    assert weighted_jaccard(np.array([1.0, np.nan, 2.0]), np.array([1.0, 5.0, -1.0])) == pytest.approx(1 / 8)
    assert weighted_jaccard({}, {}) == 1.0 and weighted_jaccard({"a": 0.0}, {"b": 2.0}) == 0.0


def test_icws_unbiased_and_reduces_to_jaccard() -> None:
    rng = np.random.default_rng(8)
    K, d = 256, 60
    wmh = WeightedMinHash(K, seed=3)
    ids = [f"f{j}" for j in range(d)]
    errs, sds = [], []
    for _ in range(60):
        x = rng.gamma(1.0, 2.0, d) * (rng.random(d) < 0.7)
        y = np.where(rng.random(d) < 0.5, x * rng.uniform(0.5, 2.0, d), rng.gamma(1.0, 2.0, d)) * (rng.random(d) < 0.8)
        J = weighted_jaccard(x, y)
        est = weighted_jaccard_estimate(wmh.signature(dict(zip(ids, x))), wmh.signature(dict(zip(ids, y))))
        errs.append(est - J)
        sds.append(np.sqrt(J * (1 - J) / K))
    errs, sds = np.array(errs), np.array(sds)
    assert abs(errs.mean()) < 4 * np.sqrt((sds**2).mean() / errs.size)  # insesgado
    assert np.abs(errs).mean() < 1.3 * np.sqrt(2 / np.pi) * sds.mean()  # dispersión binomial
    # Pesos 0/1: Jaccard generalizado = Jaccard de conjuntos
    a, b = {f"g{i}": 1.0 for i in range(0, 60)}, {f"g{i}": 1.0 for i in range(30, 90)}
    est = weighted_jaccard_estimate(wmh.signature(a), wmh.signature(b))
    assert abs(est - exact_jaccard(a, b)) < 4 * np.sqrt(1 / 3 * 2 / 3 / K)
    # Idénticos → 1; soportes disjuntos → 0; vector nulo → EMPTY
    assert weighted_jaccard_estimate(wmh.signature(a), wmh.signature(dict(a))) == 1.0
    assert weighted_jaccard_estimate(wmh.signature({"x": 2.0}), wmh.signature({"y": 2.0})) == 0.0
    assert (wmh.signature({"x": 0.0, "y": np.nan}) == EMPTY).all()


def test_icws_streaming_equals_in_memory() -> None:
    rng = np.random.default_rng(4)
    n, d, K = 12, 40, 64
    X = rng.lognormal(0.0, 1.0, (n, d)) * (rng.random((n, d)) < 0.6)
    X[0] = 0.0  # muestra sin soporte
    X[1, 3] = np.nan
    ids = [f"f{j}" for j in range(d)]
    sw = StreamingWeightedMinHash(n, K, seed=9)
    for j in rng.permutation(d):  # orden arbitrario de rasgos
        sw.update(ids[j], X[:, j])
    S = sw.signatures()
    wmh = WeightedMinHash(K, seed=9)
    for i in range(n):
        np.testing.assert_array_equal(S[i], wmh.signature(dict(zip(ids, X[i]))))
    assert (S[0] == EMPTY).all()


def test_lsh_accepts_int64_signatures_and_rejects_bad_inputs() -> None:
    """Regresión: firmas int64 se trataban como 'conjuntos de enteros' en silencio."""
    base = [{"A", "B"}, {"C"}, {"A", "B", "D"}]
    S = MinHash(16, seed=0).signatures(base)
    ref = MinHashLSH(8, 2, seed=0).fit(S).query(S, k=3)
    lsh = MinHashLSH(8, 2, seed=0).fit(S.view(np.int64))
    assert lsh.sets_ is None
    got = lsh.query(S.view(np.int64), k=3)
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])
    with pytest.raises(TypeError):
        MinHashLSH(8, 2).fit(np.zeros((2, 16)))
    with pytest.raises(TypeError):
        lsh.query("AB", k=1)
    with pytest.raises(ValueError):
        lsh.query(S, k=-1)
    idx, sc, _ = lsh.query(S, k=0)
    assert idx.shape == (3, 0) and sc.shape == (3, 0)


def test_lsh_k_larger_than_n_and_empty_base() -> None:
    lsh = MinHashLSH(4, 2, seed=1).fit([{"x"}])
    idx, sc, st = lsh.query([{"x"}, {"y"}], k=4)
    assert idx.tolist() == [[0, -1, -1, -1], [-1, -1, -1, -1]] and sc[0, 0] == 1.0
    assert np.isnan(sc[0, 1:]).all() and st["n_candidates"].tolist() == [1, 0]
    idx, _, st = MinHashLSH(4, 2).fit([]).query([{"x"}], k=2)
    assert idx.tolist() == [[-1, -1]] and st["n_candidates"].tolist() == [0]
