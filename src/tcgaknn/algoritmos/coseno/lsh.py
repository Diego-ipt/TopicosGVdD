"""LSH por bandas sobre bits SimHash, con re-ranking (plan §3.4).

Cada perfil x se resume en b_LSH = L·r bits h_i(x) = [x·Ω[:, i] ≥ 0], con Ω la
matriz compartida por ID de rasgo (``base.omega_matrix``). Los bits se cortan en
L bandas disjuntas de r bits; cada banda es una tabla hash (técnica de bandas de
Leskovec, Rajaraman y Ullman, *Mining of Massive Datasets*, cap. 3, §3.4). Un
perfil es **candidato** de la consulta q si coincide con q en al menos una banda
completa; los candidatos se re-ordenan con una métrica más fina.

Probabilidades (Charikar, 2002, "Similarity estimation techniques from rounding
algorithms", STOC; MMDS §3.7.2):

- Un hiperplano aleatorio separa u y v con probabilidad θ/π, así que
  P[h_i(u) = h_i(v)] = p = 1 − θ/π.
- Una banda de r bits independientes coincide con probabilidad p^r, y con L
  bandas independientes P_cand(θ) = 1 − (1 − p^r)^L (``s_curve``).
- El umbral de la curva S está cerca de p* ≈ (1/L)^(1/r) (MMDS §3.4.2), es decir
  θ* ≈ π·(1 − (1/L)^(1/r)) (``threshold_angle``).

Anidamiento con SimHash y RP (plan §3.11): ``omega_row`` genera cada fila con
un generador propio, de modo que las primeras c columnas de una Ω con más
columnas son idénticas a una Ω de c columnas. Así los bits de LSH son los
primeros L·r bits de SimHash y la proyección del re-ranking "m" son las
primeras m columnas de RP, con la misma semilla.

Estructura de las tablas: en vez de L diccionarios de listas de Python, cada
tabla se guarda en formato CSR, ordenando los códigos de banda una sola vez
(``argsort``): ``keys`` (códigos únicos ordenados, uint64), ``offsets`` (int64) e
``ids`` (int32). La búsqueda del bucket de todas las consultas en una tabla es
un ``searchsorted`` vectorizado. Las inserciones nuevas van a un búfer
pendiente que se compara por igualdad de códigos y se funde en las tablas
cuando crece (costo amortizado; ver ``insert``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..base import QueryResult, as_float32_2d, default_feature_ids, l2_normalize, omega_matrix, topk

RERANK_MODES = ("d", "m", None)


# --------------------------------------------------------------------------
# Curva S
# --------------------------------------------------------------------------


def s_curve(theta: np.ndarray | float, L: int, r: int) -> np.ndarray | float:
    """P[ser candidato] para un par a ángulo θ (radianes): 1 − (1 − (1 − θ/π)^r)^L."""
    p = 1.0 - np.asarray(theta, dtype=np.float64) / np.pi
    out = 1.0 - (1.0 - p**r) ** L
    return float(out) if np.ndim(out) == 0 else out


def threshold_angle(L: int, r: int) -> float:
    """Ángulo umbral aproximado de la curva S: θ* ≈ π·(1 − (1/L)^(1/r)) (radianes)."""
    return float(np.pi * (1.0 - (1.0 / L) ** (1.0 / r)))


def band_codes(bits: np.ndarray, L: int, r: int) -> np.ndarray:
    """Codifica cada banda de r bits como entero: uint64 de forma (n, L).

    La banda t son los bits [t·r, (t+1)·r); el bit j de la banda pesa 2^j.
    """
    B = np.asarray(bits, dtype=bool)
    if B.ndim == 1:
        B = B[None, :]
    if B.shape[1] < L * r:
        raise ValueError(f"se necesitan al menos L·r = {L * r} bits, llegaron {B.shape[1]}")
    B = B[:, : L * r].reshape(B.shape[0], L, r)
    codes = np.zeros((B.shape[0], L), dtype=np.uint64)
    for j in range(r):
        codes |= B[:, :, j].astype(np.uint64) << np.uint64(j)
    return codes


# --------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------


class BandedLSH:
    """LSH por bandas (L tablas de r bits) sobre SimHash, con re-ranking.

    Parámetros:

    - ``L``, ``r``: número de bandas y bits por banda (1 ≤ r ≤ 64).
    - ``seed``: semilla de Ω (la misma que usan SimHash y RP).
    - ``rerank``: cómo se ordenan los candidatos.
        * ``"d"``: coseno exacto en el espacio original (guarda la base
          normalizada; ``score_kind = "similarity"``).
        * ``"m"``: distancia euclidiana entre proyecciones RP de dimensión m,
          x̂·Ω[:, :m]/√m con x̂ = x/‖x‖ (guarda solo n × m;
          ``score_kind = "distance"``).
        * ``None``: número de bandas coincidentes (desempate por Hamming sobre
          los L·r bits y luego por índice; ``score_kind = "similarity"``).
    - ``fallback``: si es False (escenario principal), una consulta con |C| < k
      devuelve los |C| disponibles y rellena con -1. Si es True, une a C los k
      más cercanos en Hamming sobre los L·r bits (búsqueda lineal) y re-ordena.
    - ``feature_ids``: IDs de rasgo que generan Ω (por defecto ``f0, f1, …``).

    ``query`` devuelve en ``stats``: ``n_candidates`` (|C| antes del respaldo),
    ``empty`` (|C| = 0), ``short`` (|C| < k), ``n_fallback`` (agregados por el
    respaldo), ``time_s`` (desglose: hash, tablas, unión, respaldo, re-ranking),
    ``build_time_s``, ``index_bytes`` y ``rerank_space``.
    """

    def __init__(
        self,
        L: int = 16,
        r: int = 12,
        seed: int = 0,
        *,
        rerank: str | None = "d",
        m: int = 256,
        fallback: bool = False,
        feature_ids: Sequence[str] | None = None,
    ):
        if L < 1 or not 1 <= r <= 64:
            raise ValueError("se requiere L ≥ 1 y 1 ≤ r ≤ 64")
        if rerank not in RERANK_MODES:
            raise ValueError(f"rerank debe ser uno de {RERANK_MODES}")
        self.L, self.r, self.seed = int(L), int(r), int(seed)
        self.rerank = rerank
        self.m = int(m)
        self.fallback = bool(fallback)
        self.feature_ids = None if feature_ids is None else list(feature_ids)
        self.name = f"LSH(L={L},r={r},rerank={rerank}{',fb' if fallback else ''})"
        self.score_kind = "distance" if rerank == "m" else "similarity"
        self.Omega: np.ndarray | None = None
        self._reset()

    # ----------------------------------------------------------------- estado

    def _reset(self) -> None:
        self.n = 0
        self.codes = np.zeros((0, self.L), dtype=np.uint64)
        self.data: np.ndarray | None = None  # base normalizada ("d") o proyecciones ("m")
        self._keys: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        self._ids: list[np.ndarray] = []
        self.n_built = 0  # filas [0, n_built) están en las tablas; el resto, pendientes
        self.build_time_s = 0.0

    @property
    def n_bits(self) -> int:
        return self.L * self.r

    @property
    def n_cols(self) -> int:
        """Columnas de Ω que usa el índice: L·r para los bits y m para el re-ranking "m"."""
        return max(self.n_bits, self.m if self.rerank == "m" else 0)

    def _omega(self, d: int, *, new: bool = False) -> np.ndarray:
        if self.Omega is not None and self.Omega.shape[0] != d and not new:
            raise ValueError(f"el índice usa d = {self.Omega.shape[0]} rasgos y llegaron {d} columnas")
        if self.Omega is None or self.Omega.shape[0] != d:
            ids = self.feature_ids if self.feature_ids is not None else default_feature_ids(d)
            if len(ids) != d:
                raise ValueError(f"feature_ids tiene {len(ids)} IDs y los datos {d} columnas")
            self.Omega = omega_matrix(ids, self.n_cols, self.seed)
        return self.Omega

    def hash(self, X: np.ndarray, *, _new: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """Bits (bool n × L·r) y datos de re-ranking de X (None si ``rerank=None``).

        Una vez creada Ω (por ``fit``), X debe tener el mismo número de columnas;
        solo ``fit`` puede cambiar d (antes se regeneraba Ω en silencio y una
        consulta con otra dimensión devolvía basura en vez de un error).
        """
        Xn = l2_normalize(X)
        Om = self._omega(Xn.shape[1], new=_new)
        Y = Xn @ Om
        bits = Y[:, : self.n_bits] >= 0
        if self.rerank == "d":
            extra = Xn
        elif self.rerank == "m":
            extra = np.ascontiguousarray(Y[:, : self.m] / np.float32(math.sqrt(self.m)), dtype=np.float32)
        else:
            extra = None
        return bits, extra

    def _check_extra(self, extra: np.ndarray | None, n: int) -> np.ndarray | None:
        if self.rerank is None:
            return None
        if extra is None:
            raise ValueError(f'rerank="{self.rerank}" necesita rerank_data (base en d o proyecciones en m)')
        E = as_float32_2d(extra)
        if E.shape[0] != n:
            raise ValueError(f"rerank_data tiene {E.shape[0]} filas y los bits {n}")
        if self.rerank == "d":
            E = l2_normalize(E)
        elif E.shape[1] != self.m:
            raise ValueError(f"las proyecciones deben tener m = {self.m} columnas")
        return E

    # -------------------------------------------------------------- construir

    def fit(self, X: np.ndarray) -> "BandedLSH":
        """Construye el índice desde la matriz en memoria (n × d)."""
        t0 = time.perf_counter()
        bits, extra = self.hash(as_float32_2d(X), _new=True)
        self._fit_codes(band_codes(bits, self.L, self.r), extra)
        self.build_time_s = time.perf_counter() - t0
        return self

    def fit_bits(self, bits: np.ndarray, rerank_data: np.ndarray | None = None) -> "BandedLSH":
        """Construye el índice desde bits ya calculados (p. ej. ``acc.sign_bits(L·r)``).

        ``rerank_data`` es la base en d (``rerank="d"``; se normaliza aquí) o las
        proyecciones n × m (``rerank="m"``; p. ej. ``acc.projections(m)``).
        """
        t0 = time.perf_counter()
        codes = band_codes(bits, self.L, self.r)
        self._fit_codes(codes, self._check_extra(rerank_data, codes.shape[0]))
        self.build_time_s = time.perf_counter() - t0
        return self

    def _fit_codes(self, codes: np.ndarray, extra: np.ndarray | None) -> None:
        self._reset()
        self.codes = np.ascontiguousarray(codes, dtype=np.uint64)
        self.n = self.codes.shape[0]
        self.data = extra
        self.rebuild()

    def rebuild(self) -> None:
        """(Re)construye las L tablas CSR con todas las filas, incluidas las pendientes."""
        id_dtype = np.int32 if self.n < 2**31 else np.int64
        self._keys, self._offsets, self._ids = [], [], []
        for t in range(self.L):
            col = self.codes[:, t]
            order = np.argsort(col, kind="stable")
            keys, starts = np.unique(col[order], return_index=True)
            self._keys.append(keys)
            self._offsets.append(np.append(starts, self.n).astype(np.int64))
            self._ids.append(order.astype(id_dtype))
        self.n_built = self.n

    def insert(self, X_new: np.ndarray) -> np.ndarray:
        """Agrega filas a la base; devuelve sus índices."""
        bits, extra = self.hash(as_float32_2d(X_new))
        return self._insert_codes(band_codes(bits, self.L, self.r), extra)

    def insert_bits(self, bits: np.ndarray, rerank_data: np.ndarray | None = None) -> np.ndarray:
        """Como ``insert``, pero con bits ya calculados (ver ``fit_bits``)."""
        codes = band_codes(bits, self.L, self.r)
        return self._insert_codes(codes, self._check_extra(rerank_data, codes.shape[0]))

    def _insert_codes(self, codes: np.ndarray, extra: np.ndarray | None) -> np.ndarray:
        """Inserción incremental: O(L) por fila más una refusión amortizada.

        Las filas nuevas quedan en un búfer pendiente que las consultas revisan
        por igualdad de códigos (vectorizado). Cuando el búfer supera
        max(1.024, n_built/4) filas se reconstruyen las tablas, así que el costo
        amortizado por inserción es O(L·log n).
        """
        new = np.arange(self.n, self.n + codes.shape[0], dtype=np.int64)
        self.codes = np.concatenate([self.codes, codes.astype(np.uint64)])
        if self.rerank is not None:
            self.data = extra if self.data is None else np.concatenate([self.data, extra])
        self.n = self.codes.shape[0]
        if self.n - self.n_built > max(1024, self.n_built // 4):
            self.rebuild()
        return new

    # -------------------------------------------------------------- consultar

    def _lookup(self, qcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rangos [ini, fin) del bucket de cada consulta en cada tabla: (nq, L) int64."""
        nq = qcodes.shape[0]
        lo = np.zeros((nq, self.L), dtype=np.int64)
        hi = np.zeros((nq, self.L), dtype=np.int64)
        for t in range(self.L):
            keys = self._keys[t]
            if keys.size == 0:
                continue
            pos = np.searchsorted(keys, qcodes[:, t])
            posc = np.minimum(pos, keys.size - 1)
            hit = keys[posc] == qcodes[:, t]
            off = self._offsets[t]
            lo[:, t] = np.where(hit, off[posc], 0)
            hi[:, t] = np.where(hit, off[posc + 1], 0)
        return lo, hi

    def _candidates_one(self, qcode: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Candidatos (ordenados, sin duplicados) y bandas coincidentes de una consulta."""
        parts = [self._ids[t][lo[t] : hi[t]] for t in range(self.L) if hi[t] > lo[t]]
        tot = sum(p.size for p in parts)
        if tot == 0:
            cand = np.zeros(0, dtype=np.int64)
            cnt = np.zeros(0, dtype=np.int64)
        elif tot * 8 > self.n_built:  # muchos IDs: conteo denso O(n)
            bc = np.bincount(np.concatenate(parts), minlength=self.n_built)
            cand = np.flatnonzero(bc).astype(np.int64)
            cnt = bc[cand].astype(np.int64)
        else:
            cand, cnt = np.unique(np.concatenate(parts), return_counts=True)
            cand, cnt = cand.astype(np.int64), cnt.astype(np.int64)
        if self.n > self.n_built:  # búfer pendiente
            pc = (self.codes[self.n_built :] == qcode[None, :]).sum(axis=1)
            sel = np.flatnonzero(pc)
            cand = np.concatenate([cand, self.n_built + sel])
            cnt = np.concatenate([cnt, pc[sel].astype(np.int64)])
        return cand, cnt

    def _hamming(self, qcode: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        C = self.codes if rows is None else self.codes[rows]
        return np.bitwise_count(C ^ qcode[None, :]).sum(axis=1, dtype=np.int64)

    def candidates(self, Q: np.ndarray) -> list[np.ndarray]:
        """Conjunto de candidatos C(q) (int64 ordenado) de cada consulta, sin respaldo."""
        bits, _ = self.hash(as_float32_2d(Q))
        return self.candidates_bits(bits)

    def candidates_bits(self, Qbits: np.ndarray) -> list[np.ndarray]:
        qcodes = band_codes(Qbits, self.L, self.r)
        lo, hi = self._lookup(qcodes)
        return [self._candidates_one(qcodes[i], lo[i], hi[i])[0] for i in range(qcodes.shape[0])]

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        t0 = time.perf_counter()
        bits, extra = self.hash(as_float32_2d(Q))
        t_hash = time.perf_counter() - t0
        idx, scores, stats = self._query_codes(band_codes(bits, self.L, self.r), extra, k)
        stats["time_s"]["hash"] = t_hash + stats["time_s"]["hash"]
        return idx, scores, stats

    def query_bits(self, Qbits: np.ndarray, k: int, rerank_data: np.ndarray | None = None) -> QueryResult:
        """Consulta con bits ya calculados; ``rerank_data`` como en ``fit_bits``."""
        t0 = time.perf_counter()
        qcodes = band_codes(Qbits, self.L, self.r)
        extra = self._check_extra(rerank_data, qcodes.shape[0])
        t_hash = time.perf_counter() - t0
        idx, scores, stats = self._query_codes(qcodes, extra, k)
        stats["time_s"]["hash"] = t_hash + stats["time_s"]["hash"]
        return idx, scores, stats

    def _query_codes(self, qcodes: np.ndarray, qextra: np.ndarray | None, k: int) -> QueryResult:
        nq = qcodes.shape[0]
        times = dict(hash=0.0, lookup=0.0, union=0.0, fallback=0.0, rerank=0.0)
        idx_out = np.full((nq, k), -1, dtype=np.int64)
        val_out = np.full((nq, k), np.nan, dtype=np.float32)
        n_cand = np.zeros(nq, dtype=np.int64)
        n_fb = np.zeros(nq, dtype=np.int64)

        t0 = time.perf_counter()
        lo, hi = self._lookup(qcodes)
        times["lookup"] = time.perf_counter() - t0

        for i in range(nq):
            t0 = time.perf_counter()
            cand, cnt = self._candidates_one(qcodes[i], lo[i], hi[i])
            n_cand[i] = cand.size
            t1 = time.perf_counter()
            times["union"] += t1 - t0

            if self.fallback and cand.size < k and self.n > cand.size:
                ham = self._hamming(qcodes[i])
                near, _ = topk(ham[None, :], k, largest=False)
                near = near[0][near[0] >= 0]
                extra_ids = np.setdiff1d(near, cand, assume_unique=True)
                n_fb[i] = extra_ids.size
                order = np.argsort(np.concatenate([cand, extra_ids]), kind="stable")
                cand = np.concatenate([cand, extra_ids])[order]
                cnt = np.concatenate([cnt, np.zeros(extra_ids.size, dtype=np.int64)])[order]
            t2 = time.perf_counter()
            times["fallback"] += t2 - t1

            if cand.size:
                pos, val = self._rerank(cand, cnt, qcodes[i], None if qextra is None else qextra[i], k)
                pos, val = pos[0], val[0]
                ok = pos >= 0
                idx_out[i, ok] = cand[pos[ok]]
                val_out[i] = val
            times["rerank"] += time.perf_counter() - t2

        stats: dict[str, Any] = {
            "n_candidates": n_cand,
            "empty": n_cand == 0,
            "short": n_cand < k,
            "n_fallback": n_fb,
            "time_s": times,
            "build_time_s": self.build_time_s,
            "index_bytes": self.index_bytes(),
            "rerank_space": {"d": "d", "m": f"m={self.m}", None: "bandas"}[self.rerank],
        }
        return idx_out, val_out, stats

    def _rerank(
        self, cand: np.ndarray, cnt: np.ndarray, qcode: np.ndarray, qx: np.ndarray | None, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k entre los candidatos; devuelve posiciones dentro de ``cand`` y puntajes."""
        if self.rerank == "d":
            if cand.size * 4 > self.n:  # C grande: un producto completo evita copiar |C| × d
                s = (self.data @ qx)[cand]
            else:
                s = self.data[cand] @ qx
            return topk(s[None, :], k, largest=True)
        if self.rerank == "m":
            diff = self.data[cand] - qx[None, :]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            return topk(dist[None, :], k, largest=False)
        # Sin re-ranking: más bandas coincidentes primero; desempate por Hamming.
        ham = self._hamming(qcode, cand)
        key = cnt.astype(np.float64) * (self.n_bits + 1) + (self.n_bits - ham)
        pos, _ = topk(key[None, :], k, largest=True)
        val = np.full((1, k), np.nan, dtype=np.float32)
        ok = pos[0] >= 0
        val[0, ok] = cnt[pos[0][ok]]
        return pos, val

    # -------------------------------------------------------------- reportes

    def bucket_sizes(self, table: int | None = None) -> np.ndarray:
        """Tamaños de los buckets (int64) de una tabla o de todas (concatenados).

        Considera solo las filas ya fundidas en las tablas (llamar ``rebuild``
        antes si hay inserciones pendientes).
        """
        tabs = range(self.L) if table is None else [table]
        sizes = [np.diff(self._offsets[t]) for t in tabs]
        return np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)

    def bucket_summary(self) -> dict[str, float]:
        """Resumen de la distribución de tamaños de bucket (plan §3.4, riesgo)."""
        s = self.bucket_sizes()
        if s.size == 0:
            return {"n_buckets": 0}
        return {
            "n_buckets": int(s.size),
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "p95": float(np.percentile(s, 95)),
            "max": int(s.max()),
            "frac_singletons": float(np.mean(s == 1)),
            # E|C| aproximado de una consulta de la base: Σ_t Σ_b s_b²/n (con repeticiones)
            "expected_hits_per_query": float(np.sum(s.astype(np.float64) ** 2) / max(self.n_built, 1)),
        }

    def index_bytes(self) -> dict[str, int]:
        """Memoria del índice: tablas (claves + offsets + IDs), códigos y datos de re-ranking."""
        tables = sum(a.nbytes for a in self._keys) + sum(a.nbytes for a in self._offsets) + sum(a.nbytes for a in self._ids)
        rer = 0 if self.data is None else int(self.data.nbytes)
        return {
            "tables": int(tables),
            "codes": int(self.codes.nbytes),
            "rerank_data": rer,
            "omega": 0 if self.Omega is None else int(self.Omega.nbytes),
            "total_sin_omega": int(tables + self.codes.nbytes + rer),
        }
