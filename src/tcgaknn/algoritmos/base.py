"""Piezas comunes de los algoritmos.

- ``KNNIndex``: interfaz que comparten todos los índices de búsqueda.
- ``l2_normalize`` y ``topk``: utilidades numéricas deterministas.
- Matriz aleatoria Ω **generada por ID de rasgo**: la fila de Ω que corresponde
  a un gen o sonda depende solo de (semilla, ID). Así la versión en memoria
  (``omega_matrix``) y la versión en streaming (``omega_row`` por cada fila que
  llega) usan exactamente la misma Ω, sin guardarla.
- ``StreamingProjectionAccumulator``: acumula Y = Z·Ω fila de rasgo a fila de
  rasgo (el stream llega con un rasgo para todas las muestras), con
  estandarización por rasgo y normas por muestra, en lotes (GEMM).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

QueryResult = tuple[np.ndarray, np.ndarray, dict[str, Any]]


@runtime_checkable
class KNNIndex(Protocol):
    """Interfaz común de los índices de búsqueda.

    ``query`` devuelve ``(idx, scores, stats)``:

    - ``idx``: int64 de forma (nq, k); se rellena con -1 si hay menos de k resultados.
    - ``scores``: float32 (nq, k); similitud (mayor es mejor) o distancia (menor
      es mejor) según el índice, documentado en ``score_kind``; NaN en el relleno.
    - ``stats``: al menos ``n_candidates`` (int64 de largo nq).
    """

    name: str
    score_kind: str  # "similarity" | "distance"

    def fit(self, X: np.ndarray) -> "KNNIndex": ...

    def query(self, Q: np.ndarray, k: int) -> QueryResult: ...


def as_float32_2d(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2:
        raise ValueError(f"se esperaba una matriz 2D, llegó forma {A.shape}")
    return np.ascontiguousarray(A)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza filas a norma 1; las filas nulas quedan en cero."""
    A = as_float32_2d(X)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, norms, out=np.zeros_like(A), where=norms > eps)


def topk(scores: np.ndarray, k: int, *, largest: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Top-k por fila, determinista.

    Orden: por valor (descendente si ``largest``) y, en empates, por índice
    menor, incluso en la frontera del k-ésimo. NaN se trata como el peor valor.
    Si k > n, se rellena con -1 y NaN.
    """
    S = np.atleast_2d(np.asarray(scores))
    nq, n = S.shape
    idx_out = np.full((nq, k), -1, dtype=np.int64)
    val_out = np.full((nq, k), np.nan, dtype=np.float32)
    kk = min(k, n)
    if kk == 0:
        return idx_out, val_out
    key = (-S if largest else S).astype(np.float64, copy=True)
    key[np.isnan(key)] = np.inf
    part = np.argpartition(key, kk - 1, axis=1)[:, :kk]
    thr = np.take_along_axis(key, part, axis=1).max(axis=1)
    for i in range(nq):
        row = key[i]
        less = np.flatnonzero(row < thr[i])
        eq = np.flatnonzero(row == thr[i])[: kk - less.size]
        sel = np.concatenate([less, eq])
        sel = sel[np.lexsort((sel, row[sel]))]
        idx_out[i, :kk] = sel
        val_out[i, :kk] = S[i, sel]
    return idx_out, val_out


# --------------------------------------------------------------------------
# Matriz aleatoria Ω derivada de (semilla, ID de rasgo)
# --------------------------------------------------------------------------


def feature_seed(feature_id: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}\x00{feature_id}".encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "little")


def omega_row(feature_id: str, cols: int, seed: int) -> np.ndarray:
    """Fila de Ω (N(0, 1), float32) para un rasgo; reproducible sin guardar Ω."""
    return np.random.default_rng(feature_seed(feature_id, seed)).standard_normal(cols, dtype=np.float32)


def omega_matrix(feature_ids: Sequence[str], cols: int, seed: int) -> np.ndarray:
    """Ω completa (d × cols) para la versión en memoria; fila j = ``omega_row(ids[j])``."""
    return np.stack([omega_row(f, cols, seed) for f in feature_ids]) if len(feature_ids) else np.zeros((0, cols), np.float32)


def default_feature_ids(d: int) -> list[str]:
    """IDs sintéticos ("f0", "f1", …) para datos sin nombres de rasgo."""
    return [f"f{j}" for j in range(d)]


# --------------------------------------------------------------------------
# Acumulador de proyecciones en streaming
# --------------------------------------------------------------------------


class StreamingProjectionAccumulator:
    """Acumula Y = Z · Ω[:, :cols] a medida que llegan filas de rasgo.

    Cada llamada a ``update(feature_id, x)`` recibe un rasgo con un valor por
    muestra (x de largo n). Por rasgo:

    1. ``transform`` opcional (p. ej. ``lambda x: np.log2(np.maximum(x, 0) + 1)``).
    2. Si ``standardize``: z = (x − μ)/σ con μ y σ calculados solo en las
       muestras de ``fit_mask`` (regla anti-fuga); si σ = 0 o no hay datos, el
       rasgo se descarta y se cuenta en ``n_skipped``.
    3. NA → 0 (imputación por la media en el espacio estandarizado).
    4. Y += z ⊗ ω_j y ‖z_i‖² += z_i².

    Los rasgos se agrupan en lotes de ``batch_size`` y se suman con una
    multiplicación de matrices (BLAS). Memoria: O(n·cols + batch_size·(n + cols)).

    Por linealidad, ``projections(m)`` equivale a proyectar las filas
    L2-normalizadas de Z con R = Ω[:, :m]ᵀ/√m, y ``sign_bits(b)`` a SimHash.
    """

    def __init__(
        self,
        n_samples: int,
        cols: int,
        seed: int,
        *,
        standardize: bool = True,
        fit_mask: np.ndarray | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        batch_size: int = 256,
    ):
        self.n_samples = n_samples
        self.cols = cols
        self.seed = seed
        self.standardize = standardize
        self.fit_mask = None if fit_mask is None else np.asarray(fit_mask, dtype=bool)
        if self.fit_mask is not None and self.fit_mask.shape != (n_samples,):
            raise ValueError("fit_mask debe tener largo n_samples")
        self.transform = transform
        self.batch_size = batch_size
        self.Y = np.zeros((n_samples, cols), dtype=np.float64)
        self.sqnorm = np.zeros(n_samples, dtype=np.float64)
        self.n_features = 0
        self.n_skipped = 0
        self.feature_ids: list[str] = []
        self._zbuf = np.empty((batch_size, n_samples), dtype=np.float32)
        self._wbuf = np.empty((batch_size, cols), dtype=np.float32)
        self._fill = 0

    def update(self, feature_id: str, x: np.ndarray) -> bool:
        """Agrega un rasgo; devuelve False si se descartó."""
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.n_samples,):
            raise ValueError(f"{feature_id}: largo {v.shape}, se esperaba ({self.n_samples},)")
        if self.transform is not None:
            v = np.asarray(self.transform(v), dtype=np.float64)
        if self.standardize:
            ref = v if self.fit_mask is None else v[self.fit_mask]
            ref = ref[~np.isnan(ref)]
            if ref.size < 2:
                self.n_skipped += 1
                return False
            mu, sd = ref.mean(), ref.std()
            if not np.isfinite(sd) or sd <= 1e-12:
                self.n_skipped += 1
                return False
            v = (v - mu) / sd
        v = np.where(np.isnan(v), 0.0, v)
        self._zbuf[self._fill] = v
        self._wbuf[self._fill] = omega_row(feature_id, self.cols, self.seed)
        self._fill += 1
        self.feature_ids.append(feature_id)
        self.n_features += 1
        if self._fill == self.batch_size:
            self._flush()
        return True

    def _flush(self) -> None:
        if self._fill:
            Z = self._zbuf[: self._fill].astype(np.float64)
            self.Y += Z.T @ self._wbuf[: self._fill].astype(np.float64)
            self.sqnorm += np.einsum("ij,ij->j", Z, Z)
            self._fill = 0

    def raw(self) -> np.ndarray:
        """Y = Z · Ω sin normalizar (n × cols)."""
        self._flush()
        return self.Y

    def projections(self, m: int | None = None, *, normalize: bool = True) -> np.ndarray:
        """Proyección RP de dimensión m (por defecto, todas las columnas), float32."""
        self._flush()
        m = self.cols if m is None else m
        P = self.Y[:, :m] / np.sqrt(m)
        if normalize:
            nrm = np.sqrt(self.sqnorm)[:, None]
            P = np.divide(P, nrm, out=np.zeros_like(P), where=nrm > 1e-12)
        return P.astype(np.float32)

    def sign_bits(self, b: int | None = None) -> np.ndarray:
        """Bits SimHash (bool, n × b): signo de las primeras b columnas de Y."""
        self._flush()
        b = self.cols if b is None else b
        return self.Y[:, :b] >= 0
