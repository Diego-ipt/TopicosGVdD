from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from tcgaknn.algoritmos.sketches import (
    CountMinSketch,
    HyperLogLog,
    KLLSketch,
    MisraGries,
    ReservoirSampler,
    hash64,
    hash64_many,
)


def _zipf_stream(n: int, universe: int, a: float, seed: int) -> np.ndarray:
    """Stream con frecuencias tipo Zipf (pocos ítems muy frecuentes y cola larga)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, universe + 1) ** a
    return rng.choice(universe, size=n, p=p / p.sum())


# --------------------------------------------------------------------------
# Hash
# --------------------------------------------------------------------------


def test_hash64_stable_and_vectorized() -> None:
    xs = np.array([0, 1, -5, 2**40, 123456789], dtype=np.int64)
    np.testing.assert_array_equal(hash64_many(xs, seed=7), [hash64(int(x), seed=7) for x in xs])
    assert hash64("TP53", 0) == hash64("TP53", 0)
    assert hash64("TP53", 0) != hash64("TP53", 1)
    assert hash64(5, 0) != hash64(5, 1)
    # bits balanceados: cada bit vale 1 en ~la mitad de 20000 hashes de enteros consecutivos
    h = hash64_many(np.arange(20000), seed=3)
    bits = ((h[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(float)
    assert np.abs(bits.mean(axis=0) - 0.5).max() < 0.02  # 4,4·σ con σ = 0,5/√20000


# --------------------------------------------------------------------------
# Count-Min
# --------------------------------------------------------------------------


def test_countmin_dimensions_from_error() -> None:
    cm = CountMinSketch.from_error(eps=0.001, delta=0.01)
    assert cm.width == math.ceil(math.e / 0.001) == 2719
    assert cm.depth == math.ceil(math.log(100)) == 5


@pytest.mark.parametrize("conservative", [False, True])
def test_countmin_never_underestimates_and_error_bound(conservative: bool) -> None:
    eps, delta = 0.002, 0.01
    stream = _zipf_stream(100_000, 20_000, 1.1, seed=1)
    true = np.bincount(stream, minlength=20_000)
    cm = CountMinSketch.from_error(eps, delta, seed=5, conservative=conservative)
    cm.add_many(stream)
    assert cm.total == stream.size
    est = cm.estimate_many(np.arange(20_000))
    assert (est >= true).all()  # nunca subestima
    over = est - true
    # P[error > ε·N] ≤ δ por ítem: a lo más δ de los ítems (con holgura 2δ)
    assert np.mean(over > eps * stream.size) <= 2 * delta
    # pero el sketch sí comprime: hay colisiones (error > 0 en algún ítem)
    assert over.max() > 0
    assert cm.estimate(int(stream[0])) == est[stream[0]]


def test_countmin_conservative_is_tighter_and_merge() -> None:
    stream = _zipf_stream(30_000, 5_000, 1.0, seed=2)
    std = CountMinSketch(200, 4, seed=1)
    cu = CountMinSketch(200, 4, seed=1, conservative=True)
    std.add_many(stream)
    cu.add_many(stream)
    true = np.bincount(stream, minlength=5_000)
    e_std, e_cu = std.estimate_many(np.arange(5_000)), cu.estimate_many(np.arange(5_000))
    assert (true <= e_cu).all() and (e_cu <= e_std).all()
    assert (e_cu - true).sum() < 0.8 * (e_std - true).sum()
    # merge = sketch del stream concatenado (exacto en el modo estándar)
    a, b = CountMinSketch(200, 4, seed=1), CountMinSketch(200, 4, seed=1)
    a.add_many(stream[:12_000])
    for x in stream[12_000:13_000]:  # también por la vía escalar
        b.add(int(x))
    b.add_many(stream[13_000:])
    np.testing.assert_array_equal(a.merge(b).table, std.table)
    with pytest.raises(ValueError):
        a.merge(CountMinSketch(200, 4, seed=2))


def test_countmin_heavy_hitters_strings() -> None:
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(2000)]
    stream = [genes[i] for i in rng.integers(0, 2000, 20_000)] + ["TP53"] * 900 + ["PIK3CA"] * 500
    cm = CountMinSketch.from_error(0.005, 0.01, seed=0)
    cm.add_many(stream)
    hh = cm.heavy_hitters(genes + ["TP53", "PIK3CA"], phi=0.02)
    names = [g for g, _ in hh]
    assert names[:2] == ["TP53", "PIK3CA"]  # sin falsos negativos y ordenados
    assert len(names) == 2  # los demás tienen f ≈ 10 ≪ 0,02·N ≈ 428
    assert hh[0][1] >= 900


# --------------------------------------------------------------------------
# Misra-Gries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [10, 50])
def test_misra_gries_bound(k: int) -> None:
    stream = _zipf_stream(50_000, 3_000, 1.2, seed=4)
    true = Counter(stream.tolist())
    mg = MisraGries(k)
    mg.extend(stream.tolist())
    assert len(mg.counters) <= k and mg.n == stream.size
    bound = stream.size / (k + 1)
    for item in range(3_000):
        est, f = mg.estimate(item), true.get(item, 0)
        assert f - bound <= est <= f
    # la cota no es vacía: el ítem más frecuente queda con estimado positivo y cerca
    top = true.most_common(1)[0][0]
    assert mg.frequent()[0][0] == top
    # todo ítem con f > φN aparece en frequent(φ)
    phi = 2.0 / k
    fr = {key for key, _ in mg.frequent(phi)}
    assert {key for key, f in true.items() if f > phi * stream.size} <= fr


def test_misra_gries_weighted_and_merge() -> None:
    stream = _zipf_stream(40_000, 2_000, 1.1, seed=5)
    k = 20
    a, b = MisraGries(k), MisraGries(k)
    a.extend(stream[:25_000].tolist())
    # b con pesos: los mismos ítems agrupados en corridas
    vals, cnt = np.unique(stream[25_000:], return_counts=True)
    for v, c in zip(vals.tolist(), cnt.tolist()):
        b.update(v, c)
    m = a.merge(b)
    assert len(m.counters) <= k and m.n == stream.size
    true = Counter(stream.tolist())
    bound = stream.size / (k + 1)
    for item in range(2_000):
        assert true.get(item, 0) - bound <= m.estimate(item) <= true.get(item, 0)


# --------------------------------------------------------------------------
# HyperLogLog
# --------------------------------------------------------------------------


def test_hll_alpha_and_registers() -> None:
    assert HyperLogLog(p=4).alpha == 0.673
    h = HyperLogLog(p=14)
    assert h.m == 16384 and abs(h.alpha - 0.7213 / (1 + 1.079 / 16384)) < 1e-12
    assert abs(h.std_error - 1.04 / 128) < 1e-12
    h.add_many(np.arange(1000))
    assert h.registers.max() <= 64 - 14 + 1


@pytest.mark.parametrize("n", [10, 300, 5_000, 30_000, 45_000, 100_000, 1_000_000])
def test_hll_relative_error(n: int) -> None:
    h = HyperLogLog(p=14, seed=11)
    h.add_many(np.arange(n, dtype=np.int64) * 7919 + 13)
    tol = 3 * 1.04 / math.sqrt(h.m)  # 3 errores estándar (2,4 %)
    assert abs(h.estimate() - n) / n < max(tol, 1.5 / n)  # n = 10: exacto o a 1


def test_hll_duplicates_strings_scalar_and_merge() -> None:
    names = [f"TCGA-{i:05d}" for i in range(20_000)]
    a, b, ab = HyperLogLog(12, seed=1), HyperLogLog(12, seed=1), HyperLogLog(12, seed=1)
    for s in names[:12_000] * 3:  # duplicados no cuentan
        a.add(s)
    b.add_many(names[8_000:])
    ab.add_many(names)
    a.merge(b)
    np.testing.assert_array_equal(a.registers, ab.registers)  # merge sin pérdida
    assert abs(a.estimate() - 20_000) / 20_000 < 3 * 1.04 / 64


def test_hll_error_distribution_small_p() -> None:
    """Con p = 8 (σ ≈ 6,5 %) el error cuadrático medio sobre 40 semillas ≈ 1,04/√m."""
    n = 20_000
    rel = []
    for s in range(40):
        h = HyperLogLog(8, seed=s)
        h.add_many(np.arange(n))
        rel.append(h.estimate() / n - 1)
    rel = np.array(rel)
    se = 1.04 / 16
    assert abs(rel.mean()) < 3 * se / math.sqrt(40) + 0.01  # casi insesgado
    assert 0.5 * se < rel.std() < 1.6 * se


# --------------------------------------------------------------------------
# KLL
# --------------------------------------------------------------------------


def _max_rank_error(sk: KLLSketch, data: np.ndarray) -> float:
    s = np.sort(data)
    grid = np.quantile(s, np.linspace(0.01, 0.99, 99))
    true_rank = np.searchsorted(s, grid, side="right") / s.size
    return float(np.abs(sk.rank(grid) - true_rank).max())


@pytest.mark.parametrize("dist", ["normal", "lognormal", "sorted"])
def test_kll_rank_error(dist: str) -> None:
    rng = np.random.default_rng(6)
    data = {"normal": rng.standard_normal(100_000),
            "lognormal": rng.lognormal(0, 2, 100_000),
            "sorted": np.arange(100_000, dtype=float)}[dist]  # orden adversario para muestreo ingenuo
    sk = KLLSketch(k=200, seed=0)
    sk.update_many(data)
    assert sk.n == 100_000
    assert sk.total_weight() == 100_000  # la compactación conserva el peso
    assert sk.size < 3 * 200 + 50  # memoria O(k), no O(N)
    assert _max_rank_error(sk, data) < 0.02
    # quantile ↔ rank y extremos exactos
    assert sk.quantile(0) == data.min() and sk.quantile(1) == data.max()
    for q in (0.1, 0.5, 0.9):
        true_q_rank = np.mean(data <= sk.quantile(q))
        assert abs(true_q_rank - q) < 0.02


def test_kll_update_equals_update_many_and_error_shrinks_with_k() -> None:
    rng = np.random.default_rng(7)
    data = rng.standard_normal(20_000)
    a, b = KLLSketch(50, seed=3), KLLSketch(50, seed=3)
    for x in data:
        a.update(x)
    b.update_many(np.concatenate([data[:5], [np.nan], data[5:]]))  # NaN ignorado
    assert a.levels == b.levels and a.n == b.n == 20_000
    # el error medio (sobre semillas) baja al subir k: O(1/k)
    errs = {k: np.mean([_max_rank_error(_fill(KLLSketch(k, seed=s), data), data) for s in range(6)]) for k in (16, 256)}
    assert errs[256] < errs[16] / 3


def _fill(sk: KLLSketch, data: np.ndarray) -> KLLSketch:
    sk.update_many(data)
    return sk


def test_kll_merge() -> None:
    rng = np.random.default_rng(8)
    parts = [rng.normal(loc, 1, 30_000) for loc in (0, 3, -2, 5)]
    sk = KLLSketch(200, seed=0)
    for i, p in enumerate(parts):
        sk.merge(_fill(KLLSketch(200, seed=i + 1), p))
    data = np.concatenate(parts)
    assert sk.n == data.size and sk.total_weight() == data.size
    assert sk.size < sk.max_size
    assert _max_rank_error(sk, data) < 0.025  # merges suman algo de error
    assert sk.quantile(0) == data.min()


# --------------------------------------------------------------------------
# Reservoir
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["R", "L"])
def test_reservoir_subsets_uniform_chi2(method: str) -> None:
    """Las C(6, 2) = 15 muestras posibles deben ser equiprobables (multinomial exacta)."""
    n, k, reps = 6, 2, 15_000
    counts: Counter = Counter()
    for r in range(reps):
        rs = ReservoirSampler(k, seed=r, method=method)
        for x in range(n):
            rs.update(x)
        counts[tuple(sorted(rs.sample()))] += 1
    obs = np.array([counts[c] for c in combinations(range(n), k)])
    assert obs.sum() == reps
    p = stats.chisquare(obs).pvalue
    assert p > 1e-3


@pytest.mark.parametrize("method", ["R", "L"])
def test_reservoir_inclusion_long_stream(method: str) -> None:
    """Con N = 2000 y k = 20, cada posición del stream entra con prob. k/N."""
    n, k, reps = 2_000, 20, 1_000
    hits = np.zeros(n, dtype=np.int64)
    data = np.arange(n)
    for r in range(reps):
        rs = ReservoirSampler(k, seed=1000 + r, method=method)
        rs.extend(data)  # en L usa los saltos
        s = rs.sample()
        assert len(s) == k and len(set(s)) == k and rs.n_seen == n
        hits[s] += 1
    # agrupar en 20 bloques de 100 posiciones: E = reps·k/20 = 1000 por bloque
    blocks = hits.reshape(20, -1).sum(axis=1)
    assert stats.chisquare(blocks).pvalue > 1e-3
    # el final del stream no está sobre-representado (error típico de un mal reservoir)
    assert abs(blocks[-1] - 1000) < 150


def test_reservoir_L_extend_matches_update_and_short_stream() -> None:
    a, b = ReservoirSampler(5, seed=9, method="L"), ReservoirSampler(5, seed=9, method="L")
    data = list(range(10_000))
    a.extend(data[:3])
    a.extend(data[3:4000])
    a.extend(np.array(data[4000:]))
    for x in data:
        b.update(x)
    assert a.sample() == [int(x) for x in b.sample()] and a.n_seen == b.n_seen == 10_000
    short = ReservoirSampler(10, seed=0)
    short.extend(iter(range(4)))
    assert short.sample() == [0, 1, 2, 3]


def test_hash_numpy_scalars_match_python_and_countmin_consistent() -> None:
    """Regresión: np.float64 hasheaba distinto que float (repr distinto)."""
    assert hash64(np.float64(1.5), 3) == hash64(1.5, 3)
    assert hash64(np.str_("TP53"), 3) == hash64("TP53", 3)
    cm = CountMinSketch(64, 3, seed=1)
    cm.add_many(np.array([1.5, 1.5, 2.5]))
    assert cm.estimate(1.5) >= 2


def test_sketches_edge_cases_empty_and_small() -> None:
    assert HyperLogLog(p=8).estimate() == 0.0
    kll = KLLSketch(k=8)
    kll.update_many([np.nan, np.nan])
    with pytest.raises(ValueError):
        kll.quantile(0.5)
    kll.update(3.0)
    assert kll.quantile(0.0) == kll.quantile(1.0) == 3.0
    mg = MisraGries(2)
    assert mg.frequent(0.1) == [] and mg.estimate("x") == 0
    rs = ReservoirSampler(10, method="L")
    rs.extend([1, 2, 3])
    assert rs.sample() == [1, 2, 3]
    assert CountMinSketch(4, 2).heavy_hitters([], phi=0.1) == []
