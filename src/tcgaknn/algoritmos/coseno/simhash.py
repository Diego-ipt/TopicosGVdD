"""SH: firmas SimHash por hiperplanos aleatorios y búsqueda lineal en Hamming.

Referencia: M. S. Charikar, "Similarity estimation techniques from rounding
algorithms", STOC 2002, pp. 380–388 (DOI 10.1145/509907.509965). Para un
hiperplano aleatorio de normal a ~ N(0, I), h_a(x) = [a·x ≥ 0] cumple

    P[h_a(u) = h_a(v)] = 1 − θ(u, v)/π,

(es el argumento de Goemans–Williamson: el hiperplano separa u y v con
probabilidad θ/π). Con b hiperplanos independientes, Ham(s(u), s(v))/b es una
proporción binomial de media p = θ/π y desviación √(p(1−p)/b), así que

    θ̂ = π·Ham/b      y      cos θ̂  estiman el ángulo y el coseno.

Piezas del módulo:

- ``sign_bits`` / ``signatures``: bits (bool) y firmas empaquetadas (uint64).
- ``pack_bits`` / ``unpack_bits``: empaquetado little-endian en palabras de 64
  bits (bit j de la firma = bit j % 64 de la palabra j // 64).
- ``hamming`` / ``hamming_matrix``: popcount(XOR) con ``np.bitwise_count``
  (NumPy ≥ 2.0), por bloques para acotar la memoria.
- ``angle_estimate`` / ``cosine_estimate``.
- ``SimHashKNN``: índice ``KNNIndex`` con búsqueda lineal en Hamming y
  re-ranking opcional por coseno exacto de los f·k mejores candidatos.

Los hiperplanos son las primeras b columnas de Ω = ``omega_matrix(ids, b, seed)``,
generada por ID de rasgo (ver ``base``). Como ``omega_row`` saca las columnas de
un mismo generador en orden, Ω con b columnas es el prefijo de Ω con más
columnas: los primeros b bits de SH coinciden con ``StreamingProjectionAccumulator
.sign_bits(b)`` y con los bits que usa LSH (misma semilla e IDs).

Advertencia (plan §3.3): el ángulo es respecto del origen. Sin centrar por
rasgo (z-score) casi todos los bits coinciden entre todos los pacientes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..base import (
    QueryResult,
    as_float32_2d,
    default_feature_ids,
    l2_normalize,
    omega_matrix,
    topk,
)

WORD_BITS = 64
_SIGN_BLOCK_ROWS = 1024  # filas por bloque al calcular X·Ω en float64
_HAMMING_BLOCK_ELEMS = 1 << 22  # elementos (consultas × base) por bloque de Hamming

OmegaSpec = np.ndarray | tuple[Sequence[str], int]


def _check_b(b: int) -> int:
    b = int(b)
    if b <= 0 or b % WORD_BITS:
        raise ValueError(f"b debe ser un múltiplo positivo de {WORD_BITS}, llegó {b}")
    return b


def _resolve_omega(Omega: OmegaSpec, d: int, b: int) -> np.ndarray:
    """Devuelve Ω[:, :b] (d × b) a partir de una matriz o de (feature_ids, seed)."""
    if isinstance(Omega, tuple):
        ids, seed = Omega
        if len(ids) != d:
            raise ValueError(f"hay {len(ids)} IDs de rasgo para {d} columnas")
        return omega_matrix(list(ids), b, int(seed))
    W = np.asarray(Omega, dtype=np.float32)
    if W.ndim != 2 or W.shape[0] != d or W.shape[1] < b:
        raise ValueError(f"Ω debe ser ({d} × ≥{b}), llegó {W.shape}")
    return W[:, :b]


# --------------------------------------------------------------------------
# Firmas
# --------------------------------------------------------------------------


def sign_bits(X: np.ndarray, Omega: OmegaSpec, b: int) -> np.ndarray:
    """Bits SimHash (bool, n × b): bit i de x = [x·Ω[:, i] ≥ 0].

    ``Omega`` es una matriz d × (≥ b) o el par ``(feature_ids, seed)`` (se usa
    ``omega_matrix``). El producto se hace en float64 y por bloques de filas,
    igual que el acumulador en streaming, para que los signos coincidan con
    ``StreamingProjectionAccumulator.sign_bits``. El signo no cambia al escalar
    x por un positivo, así que normalizar las filas antes no altera los bits.
    Una fila nula da todos los bits en 1 (0 ≥ 0), como en el acumulador.
    No se admiten NaN ni Inf (hay que imputar antes).
    """
    b = _check_b(b)
    A = as_float32_2d(X)
    n, d = A.shape
    W = _resolve_omega(Omega, d, b).astype(np.float64)
    out = np.empty((n, b), dtype=bool)
    for i in range(0, n, _SIGN_BLOCK_ROWS):
        blk = A[i : i + _SIGN_BLOCK_ROWS]
        if not np.isfinite(blk).all():
            raise ValueError("X contiene NaN o Inf; imputar antes de calcular firmas")
        np.greater_equal(blk.astype(np.float64) @ W, 0.0, out=out[i : i + _SIGN_BLOCK_ROWS])
    return out


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Empaqueta bits (n × b, b múltiplo de 64) en uint64 (n × b/64), little-endian.

    El bit j de la firma queda en el bit (j mod 64) de la palabra j // 64, sin
    depender del orden de bytes de la máquina.
    """
    B = np.atleast_2d(np.asarray(bits, dtype=bool))
    _check_b(B.shape[1])
    by = np.packbits(B, axis=1, bitorder="little")  # byte k = bits 8k..8k+7
    return np.ascontiguousarray(by).view("<u8").astype(np.uint64, copy=False)


def unpack_bits(packed: np.ndarray, b: int | None = None) -> np.ndarray:
    """Inversa de ``pack_bits``: uint64 (n × w) → bool (n × b), b ≤ 64·w."""
    P = np.atleast_2d(np.asarray(packed, dtype=np.uint64))
    total = P.shape[1] * WORD_BITS
    b = total if b is None else int(b)
    if not 0 <= b <= total:
        raise ValueError(f"b = {b} fuera de rango para {P.shape[1]} palabras")
    by = np.ascontiguousarray(P.astype("<u8", copy=False)).view(np.uint8)
    return np.unpackbits(by, axis=1, count=b, bitorder="little").astype(bool)


def signatures(X: np.ndarray, Omega: OmegaSpec, b: int) -> np.ndarray:
    """Firmas SimHash empaquetadas: ``pack_bits(sign_bits(X, Omega, b))``."""
    return pack_bits(sign_bits(X, Omega, b))


# --------------------------------------------------------------------------
# Distancia de Hamming y estimadores
# --------------------------------------------------------------------------


def hamming_matrix(Qp: np.ndarray, Dp: np.ndarray, *, block_elems: int = _HAMMING_BLOCK_ELEMS) -> np.ndarray:
    """Distancias de Hamming (int32, nq × n) entre firmas empaquetadas.

    Ham = Σ_palabras popcount(q_w XOR d_w), con ``np.bitwise_count``. Se recorre
    por bloques de consultas y palabra por palabra, así la memoria temporal es
    O(bloque·n) y no O(bloque·n·w). Costo: O(nq·n·w) XOR + popcount.
    """
    Q = np.atleast_2d(np.asarray(Qp, dtype=np.uint64))
    D = np.atleast_2d(np.asarray(Dp, dtype=np.uint64))
    if Q.shape[1] != D.shape[1]:
        raise ValueError(f"las firmas tienen distinto largo: {Q.shape[1]} vs {D.shape[1]} palabras")
    nq, n = Q.shape[0], D.shape[0]
    H = np.zeros((nq, n), dtype=np.int32)
    if nq == 0 or n == 0:
        return H
    bq = max(1, block_elems // n)
    DT = np.ascontiguousarray(D.T)  # (w, n): cada palabra contigua
    for i in range(0, nq, bq):
        Hb = H[i : i + bq]
        for w in range(Q.shape[1]):
            Hb += np.bitwise_count(Q[i : i + bq, w, None] ^ DT[w][None, :])
    return H


def hamming(q_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
    """Hamming de una firma (w,) contra la base (n × w) → (n,).

    Si ``q_packed`` es 2D (nq × w) equivale a ``hamming_matrix``.
    """
    q = np.asarray(q_packed, dtype=np.uint64)
    if q.ndim == 1:
        return hamming_matrix(q[None, :], db_packed)[0]
    return hamming_matrix(q, db_packed)


def angle_estimate(hd: np.ndarray | int, b: int) -> np.ndarray:
    """θ̂ = π·Ham/b (radianes), estimador insesgado de θ (Charikar 2002)."""
    return np.pi * np.asarray(hd, dtype=np.float64) / b


def cosine_estimate(hd: np.ndarray | int, b: int) -> np.ndarray:
    """cos θ̂: estimación del coseno a partir de la distancia de Hamming."""
    return np.cos(angle_estimate(hd, b))


# --------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------


class SimHashKNN:
    """k-NN aproximado con firmas SimHash de b bits y búsqueda lineal en Hamming.

    - ``fit(X)``: normaliza las filas (L2), calcula Ω = ``omega_matrix(ids, b,
      seed)`` y las firmas empaquetadas (n × b/64 uint64, n·b/8 bytes).
    - ``query(Q, k)``: firma de cada consulta, Hamming contra toda la base,
      top-c por Hamming con c = f·k (empates por índice menor).

      * f = 0: devuelve esos k con score = cos θ̂ = cos(π·Ham/b).
      * f > 0: re-ranking de los c candidatos con el coseno exacto y score =
        coseno exacto. Requiere guardar la base normalizada en float32, es
        decir, n·d·4 bytes extra (≈ 0,9 GB con n = 11.000 y d = 20.500), contra
        n·b/8 bytes de las firmas (352 KB con b = 256). Ese costo se reporta en
        ``stats['base_bytes']``.

    ``score_kind`` = "similarity" en ambos casos. ``n_candidates`` = n porque
    la búsqueda en Hamming es lineal (todas las firmas se comparan).
    """

    name = "SH"
    score_kind = "similarity"

    def __init__(
        self,
        b: int = 256,
        rerank_factor: int = 0,
        seed: int = 0,
        feature_ids: Sequence[str] | None = None,
    ):
        self.b = _check_b(b)
        if rerank_factor < 0:
            raise ValueError("rerank_factor debe ser ≥ 0")
        self.rerank_factor = int(rerank_factor)
        self.seed = int(seed)
        self.feature_ids = None if feature_ids is None else list(feature_ids)
        self.Omega: np.ndarray | None = None
        self.packed: np.ndarray | None = None
        self.base: np.ndarray | None = None
        self.d: int | None = None

    # -- construcción -------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SimHashKNN":
        A = l2_normalize(X)
        n, d = A.shape
        if self.feature_ids is None:
            self.feature_ids = default_feature_ids(d)
        if len(self.feature_ids) != d:
            raise ValueError(f"hay {len(self.feature_ids)} IDs de rasgo para d = {d}")
        self.d = d
        self.Omega = omega_matrix(self.feature_ids, self.b, self.seed)
        # Bits desde X sin normalizar: mismos signos que el acumulador en streaming.
        self.packed = signatures(X, self.Omega, self.b)
        self.base = A if self.rerank_factor > 0 else None
        return self

    @classmethod
    def from_signatures(
        cls,
        packed: np.ndarray,
        b: int,
        base: np.ndarray | None = None,
        *,
        rerank_factor: int = 0,
        seed: int = 0,
        feature_ids: Sequence[str] | None = None,
    ) -> "SimHashKNN":
        """Índice a partir de firmas ya calculadas (p. ej. del acumulador en streaming).

        ``packed``: uint64 (n × ≥ b/64); se usan las primeras b/64 palabras.
        ``base``: vectores (n × m) para el re-ranking; puede ser Z completo o
        las proyecciones RP (re-ranking en el espacio reducido, plan §3.4). Si
        se pasan ``feature_ids`` (y la misma ``seed``), ``query`` puede firmar
        consultas crudas; si no, se consulta con ``query_packed``.
        """
        self = cls(b=b, rerank_factor=rerank_factor, seed=seed, feature_ids=feature_ids)
        P = np.atleast_2d(np.asarray(packed, dtype=np.uint64))
        w = self.b // WORD_BITS
        if P.shape[1] < w:
            raise ValueError(f"las firmas tienen {P.shape[1]} palabras; b = {self.b} pide {w}")
        self.packed = np.ascontiguousarray(P[:, :w])
        if self.rerank_factor > 0:
            if base is None:
                raise ValueError("rerank_factor > 0 requiere la base para el coseno exacto")
            B = l2_normalize(base)
            if B.shape[0] != self.packed.shape[0]:
                raise ValueError("base y firmas tienen distinto número de filas")
            self.base = B
        if self.feature_ids is not None:
            self.d = len(self.feature_ids)
        return self

    # -- consulta -----------------------------------------------------------

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        if self.packed is None:
            raise RuntimeError("índice sin ajustar: llamar fit o from_signatures")
        A = as_float32_2d(Q)
        if self.Omega is None:
            if self.feature_ids is None:
                raise RuntimeError("sin IDs de rasgo no se pueden firmar consultas; usar query_packed")
            self.Omega = omega_matrix(self.feature_ids, self.b, self.seed)
        if A.shape[1] != self.Omega.shape[0]:
            raise ValueError(f"la consulta tiene d = {A.shape[1]}, el índice d = {self.Omega.shape[0]}")
        Qp = signatures(A, self.Omega, self.b)
        return self.query_packed(Qp, k, Q_base=A if self.rerank_factor > 0 else None)

    def query_packed(self, Qp: np.ndarray, k: int, Q_base: np.ndarray | None = None) -> QueryResult:
        """Consulta con firmas ya empaquetadas (nq × ≥ b/64).

        ``Q_base`` (vectores de las consultas en el mismo espacio que ``base``)
        solo se necesita si hay re-ranking.
        """
        if self.packed is None:
            raise RuntimeError("índice sin ajustar: llamar fit o from_signatures")
        w = self.b // WORD_BITS
        Qw = np.atleast_2d(np.asarray(Qp, dtype=np.uint64))[:, :w]
        n, nq = self.packed.shape[0], Qw.shape[0]
        H = hamming_matrix(Qw, self.packed)
        f = self.rerank_factor
        stats: dict[str, Any] = {
            "n_candidates": np.full(nq, n, dtype=np.int64),
            "index_bytes": int(self.packed.nbytes),  # = n·b/8
            "rerank": f > 0,
            "rerank_factor": f,
            "b": self.b,
        }
        if f == 0:
            idx, hd = topk(H, k, largest=False)
            scores = np.full(idx.shape, np.nan, dtype=np.float32)
            ok = idx >= 0
            scores[ok] = cosine_estimate(hd[ok], self.b)
            stats["n_reranked"] = np.zeros(nq, dtype=np.int64)
            stats["base_bytes"] = 0
            return idx, scores, stats

        if Q_base is None:
            raise ValueError("el re-ranking necesita los vectores de las consultas (Q_base)")
        Qn = l2_normalize(Q_base)
        if self.base is None or Qn.shape[1] != self.base.shape[1]:
            raise ValueError("Q_base no coincide con la base guardada para el re-ranking")
        c = min(f * k, n)
        cand, _ = topk(H, c, largest=False)  # c ≤ n: sin relleno
        cand = np.sort(cand, axis=1)  # empates de coseno → índice global menor
        # Coseno exacto consulta a consulta: memoria O(c·d), no O(nq·c·d).
        cos = np.empty((nq, c), dtype=np.float32)
        for i in range(nq):
            cos[i] = self.base[cand[i]] @ Qn[i]
        pos, scores = topk(cos, k, largest=True)
        if c == 0:  # base vacía: todo relleno (take_along_axis fallaría con eje 0)
            idx = np.full(pos.shape, -1, dtype=np.int64)
        else:
            idx = np.where(pos >= 0, np.take_along_axis(cand, np.maximum(pos, 0), axis=1), -1).astype(np.int64)
        stats["n_reranked"] = np.full(nq, c, dtype=np.int64)
        stats["base_bytes"] = int(self.base.nbytes)
        return idx, scores.astype(np.float32), stats
