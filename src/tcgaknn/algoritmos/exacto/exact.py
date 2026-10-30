"""k-NN exacto por fuerza bruta (EX): la verdad de terreno del proyecto.

- ``ExactKNN``: índice en memoria (interfaz ``KNNIndex``) para cuatro métricas:

  ========== ============================================== ============
  métrica    cálculo                                        ``score_kind``
  ========== ============================================== ============
  cosine     filas L2-normalizadas y producto punto (BLAS)  similarity
  correlation filas centradas por muestra y luego coseno     similarity
             (= correlación de Pearson entre muestras)
  l2         ‖q‖² + ‖x‖² − 2·q·x (BLAS), con refinamiento   distance
             exacto en float64 de los mejores candidatos
  l1         Σ |q_t − x_t| con un kernel numba en paralelo  distance
             (respaldo: ``scipy.spatial.distance.cdist``)
  ========== ============================================== ============

- ``StreamingExactKNN``: el mismo k-NN exacto para un conjunto FIJO de
  consultas, acumulado fila de rasgo a fila de rasgo (stream feature-major,
  como lo entrega ``tcgaknn.stream``). Acumula Q·Bᵀ, normas, sumas por
  muestra y Σ|q − b|, con la misma regla de estandarización que
  ``StreamingProjectionAccumulator``; nunca materializa la matriz n × d.

Las consultas se procesan por bloques (``block_size`` consultas a la vez) para
acotar la memoria a O(block_size·n) además de la base. El top-k usa
``base.topk``: determinista, empates por índice menor.

Referencias de las fórmulas:

- Pearson como coseno de vectores centrados: r(u, v) = (u − ū)·(v − v̄) /
  (‖u − ū‖·‖v − v̄‖); es la ``correlation`` de SciPy
  (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html),
  que se usa como prueba cruzada.
- Expansión ‖q − x‖² = ‖q‖² + ‖x‖² − 2 q·x: la misma que usa
  ``sklearn.metrics.pairwise.euclidean_distances``, cuya documentación advierte
  la pérdida de precisión por cancelación; aquí se mitiga centrando los datos
  en la media de la base (la distancia es invariante a traslaciones) y
  recalculando en float64 la distancia de los mejores candidatos.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..base import QueryResult, as_float32_2d, l2_normalize, topk

METRICS = ("cosine", "correlation", "l2", "l1")
SIMILARITY_METRICS = frozenset({"cosine", "correlation"})


# --------------------------------------------------------------------------
# Kernel L1 (numba, con respaldo en SciPy)
# --------------------------------------------------------------------------

_L1_KERNEL: Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None = None
_L1_KERNEL_TRIED = False


def _get_l1_kernel() -> Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None:
    """Compila (una vez, en el primer uso) el kernel numba ``out[i, j] += Σ_t |A[i,t] − B[j,t]|``.

    Paralelo sobre las filas de A; acumula en float64. ``reassoc`` permite a
    LLVM vectorizar la suma (cambia el orden de suma, no la precisión de
    float64). Si numba no está disponible devuelve None y se usa SciPy.
    """
    global _L1_KERNEL, _L1_KERNEL_TRIED
    if _L1_KERNEL_TRIED:
        return _L1_KERNEL
    _L1_KERNEL_TRIED = True
    try:
        import numba
    except ImportError:  # pragma: no cover - numba está en el entorno del proyecto
        return None

    @numba.njit(parallel=True, fastmath={"reassoc", "nsz"}, cache=False)
    def _l1_accumulate(A, B, out):  # pragma: no cover - código compilado
        na, d = A.shape
        nb = B.shape[0]
        for i in numba.prange(na):
            for j in range(nb):
                s = 0.0
                for t in range(d):
                    s += abs(np.float64(A[i, t]) - np.float64(B[j, t]))
                out[i, j] += s

    _L1_KERNEL = _l1_accumulate
    return _L1_KERNEL


def l1_accumulate(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Suma en ``out`` (float64, na × nb) las distancias L1 entre filas de A y B."""
    A = as_float32_2d(A)
    B = as_float32_2d(B)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimensiones distintas: {A.shape[1]} y {B.shape[1]}")
    if out.shape != (A.shape[0], B.shape[0]) or out.dtype != np.float64:
        raise ValueError("out debe ser float64 de forma (na, nb)")
    kernel = _get_l1_kernel()
    if kernel is not None:
        kernel(A, B, out)
    else:  # pragma: no cover
        from scipy.spatial.distance import cdist

        out += cdist(A.astype(np.float64), B.astype(np.float64), metric="cityblock")
    return out


def _center_rows(X: np.ndarray) -> np.ndarray:
    """Resta a cada fila su media (calculada en float64)."""
    A = as_float32_2d(X)
    return (A - A.mean(axis=1, dtype=np.float64, keepdims=True)).astype(np.float32)


def _check_finite(A: np.ndarray, what: str) -> np.ndarray:
    """Rechaza NaN/inf: EX es la verdad de terreno y no imputa en silencio.

    (Sin esta validación, un NaN dejaba en cero la fila en coseno/correlación
    pero envenenaba la media de la base y todas las distancias en ``l2``.)
    Los NA del stream se tratan en ``standardize_feature`` (NA → 0).
    """
    if A.size and not np.isfinite(A).all():
        raise ValueError(f"{what} contiene NaN o inf; imputar antes de ExactKNN")
    return A


def _check_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"métrica desconocida {metric!r}; opciones: {METRICS}")
    return metric


# --------------------------------------------------------------------------
# Índice exacto en memoria
# --------------------------------------------------------------------------


class ExactKNN:
    """k-NN exacto por fuerza bruta (línea base EX, plan §3.1).

    ``fit(X)`` guarda la base preparada según la métrica (normalizada,
    centrada o trasladada a la media de la base); ``query(Q, k)`` compara cada
    bloque de consultas contra toda la base y toma el top-k.

    Para ``l2`` el producto matricial en float32 puede perder precisión por
    cancelación cuando ‖q − x‖² ≪ ‖q‖² + ‖x‖². Se mitiga de dos formas:
    (1) se resta la media de la base a la base y a las consultas, y
    (2) se toman los ``min(n, 2k + refine_extra)`` mejores candidatos de esa
    aproximación y se recalcula su distancia exacta en float64 antes del
    top-k final. Las distancias devueltas son, por tanto, las exactas.

    ``stats``: ``n_candidates`` (= n para cada consulta), ``metric``,
    ``build_time_s``, ``index_bytes`` y ``rerank_space``.
    """

    def __init__(self, metric: str = "cosine", block_size: int = 1024, *, refine_extra: int = 16):
        self.metric = _check_metric(metric)
        if block_size < 1:
            raise ValueError("block_size debe ser ≥ 1")
        self.block_size = int(block_size)
        self.refine_extra = int(refine_extra)
        self.name = "EX" if metric == "cosine" else f"EX-{metric}"
        self.score_kind = "similarity" if metric in SIMILARITY_METRICS else "distance"
        self._X: np.ndarray | None = None
        self._sq: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self.build_time_s = 0.0

    # -- construcción -------------------------------------------------------

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """Transformación por fila que comparten la base y las consultas."""
        if self.metric == "cosine":
            return l2_normalize(X)
        if self.metric == "correlation":
            return l2_normalize(_center_rows(X))
        if self.metric == "l2":
            return (as_float32_2d(X) - self._mu).astype(np.float32)
        return as_float32_2d(X)

    def fit(self, X: np.ndarray) -> "ExactKNN":
        t0 = time.perf_counter()
        A = _check_finite(as_float32_2d(X), "X")
        if self.metric == "l2":
            mu = A.mean(axis=0, dtype=np.float64) if A.shape[0] else np.zeros(A.shape[1])
            self._mu = mu.astype(np.float32)
        self._X = np.ascontiguousarray(self._prepare(A))
        if self.metric == "l2":
            self._sq = np.einsum("ij,ij->i", self._X, self._X, dtype=np.float64)
        self.build_time_s = time.perf_counter() - t0
        return self

    @property
    def n(self) -> int:
        return 0 if self._X is None else self._X.shape[0]

    @property
    def index_bytes(self) -> int:
        if self._X is None:
            return 0
        extra = sum(a.nbytes for a in (self._sq, self._mu) if a is not None)
        return int(self._X.nbytes + extra)

    # -- consulta ------------------------------------------------------------

    def _block_scores(self, Qb: np.ndarray) -> np.ndarray:
        """Matriz (nb × n) de similitudes o distancias del bloque contra la base."""
        X = self._X
        if self.metric in SIMILARITY_METRICS:
            return Qb @ X.T
        if self.metric == "l2":
            qsq = np.einsum("ij,ij->i", Qb, Qb, dtype=np.float64)
            D2 = qsq[:, None] + self._sq[None, :] - 2.0 * (Qb @ X.T).astype(np.float64)
            return np.maximum(D2, 0.0)
        out = np.zeros((Qb.shape[0], X.shape[0]), dtype=np.float64)
        return l1_accumulate(Qb, X, out)

    def _refine_l2(self, Qb: np.ndarray, D2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k L2 con distancia exacta en float64 sobre los mejores candidatos."""
        kk = min(self.n, 2 * k + self.refine_extra)
        if kk == 0:  # base vacía o k = 0: solo relleno
            return topk(D2, k, largest=False)
        cand, _ = topk(D2, kk, largest=False)
        exact = np.empty(cand.shape, dtype=np.float64)
        for i in range(Qb.shape[0]):
            diff = self._X[cand[i]].astype(np.float64) - Qb[i].astype(np.float64)
            exact[i] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        # Orden final por (distancia exacta, índice): se reordena la lista de
        # candidatos para que topk desempate por el índice de la base.
        order = np.argsort(cand, axis=1, kind="stable")
        cand_s = np.take_along_axis(cand, order, axis=1)
        exact_s = np.take_along_axis(exact, order, axis=1)
        pos, val = topk(exact_s, k, largest=False)
        idx = np.where(pos >= 0, np.take_along_axis(cand_s, np.maximum(pos, 0), axis=1), -1)
        return idx.astype(np.int64), val

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        if self._X is None:
            raise RuntimeError("llamar a fit antes de query")
        if k < 0:
            raise ValueError("k debe ser ≥ 0")
        Qp = self._prepare(_check_finite(as_float32_2d(Q), "Q"))
        if Qp.shape[1] != self._X.shape[1]:
            raise ValueError(f"las consultas tienen d = {Qp.shape[1]}, la base d = {self._X.shape[1]}")
        nq = Qp.shape[0]
        idx = np.full((nq, k), -1, dtype=np.int64)
        scores = np.full((nq, k), np.nan, dtype=np.float32)
        largest = self.score_kind == "similarity"
        for s in range(0, nq, self.block_size):
            Qb = np.ascontiguousarray(Qp[s : s + self.block_size])
            S = self._block_scores(Qb)
            if self.metric == "l2":
                bi, bv = self._refine_l2(Qb, S, k)
            else:
                bi, bv = topk(S, k, largest=largest)
            idx[s : s + Qb.shape[0]] = bi
            scores[s : s + Qb.shape[0]] = bv
        stats: dict[str, Any] = {
            "n_candidates": np.full(nq, self.n, dtype=np.int64),
            "metric": self.metric,
            "build_time_s": self.build_time_s,
            "index_bytes": self.index_bytes,
            "rerank_space": "exact",
        }
        return idx, scores, stats


# --------------------------------------------------------------------------
# k-NN exacto en streaming (consultas fijas, stream por rasgo)
# --------------------------------------------------------------------------


def standardize_feature(
    v: np.ndarray,
    *,
    standardize: bool,
    fit_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Regla por rasgo idéntica a ``StreamingProjectionAccumulator.update``.

    z = (x − μ)/σ con μ, σ (poblacional) de las muestras de ``fit_mask`` sin
    NA; si hay < 2 datos o σ ≤ 1e-12 devuelve None (rasgo descartado). Luego
    NA → 0. Opera en float64.
    """
    v = np.asarray(v, dtype=np.float64)
    if standardize:
        ref = v if fit_mask is None else v[fit_mask]
        ref = ref[~np.isnan(ref)]
        if ref.size < 2:
            return None
        mu, sd = ref.mean(), ref.std()
        if not np.isfinite(sd) or sd <= 1e-12:
            return None
        v = (v - mu) / sd
    return np.where(np.isnan(v), 0.0, v)


class StreamingExactKNN:
    """k-NN exacto para consultas fijas, acumulado fila de rasgo a fila de rasgo.

    El stream feature-major entrega, por rasgo j, un valor x_j(s) para cada
    muestra s. Con z_j = regla de ``standardize_feature`` (en float32, como el
    acumulador de proyecciones), para cada consulta q ∈ Q y cada b ∈ B se
    acumula:

    - G[q, b] = Σ_j z_j(q)·z_j(b)       (producto por lotes, BLAS en float64)
    - ‖q‖², ‖b‖², Σ_j z_j(q), Σ_j z_j(b) y el número d de rasgos aceptados
    - L1[q, b] = Σ_j |z_j(q) − z_j(b)|  (solo si se pide ``"l1"``)

    y al final (``result``):

    - coseno: G / (‖q‖·‖b‖)
    - l2: √max(‖q‖² + ‖b‖² − 2G, 0)
    - correlación (Pearson entre muestras sobre sus d rasgos, es decir, el
      coseno tras centrar cada muestra):
      (G − d·q̄·b̄) / √((‖q‖² − d·q̄²)(‖b‖² − d·b̄²)), con q̄ = Σ_j z_j(q)/d.
    - l1: L1

    Coincide con ``ExactKNN`` sobre la matriz materializada Z (float32) con la
    misma transformación. Memoria: O(|Q|·|B|) por matriz acumulada, más el
    lote de ``batch_size``·(|Q| + |B|).

    Las filas de una muestra con norma nula (todo NA o todo en la media)
    tienen similitud 0 con todas, como ``l2_normalize`` en ``ExactKNN``.
    """

    def __init__(
        self,
        n_samples: int,
        query_idx: Sequence[int] | np.ndarray,
        base_idx: Sequence[int] | np.ndarray,
        metrics: Sequence[str] = METRICS,
        *,
        standardize: bool = True,
        fit_mask: np.ndarray | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        batch_size: int = 256,
    ):
        self.n_samples = int(n_samples)
        self.query_idx = np.asarray(query_idx, dtype=np.int64)
        self.base_idx = np.asarray(base_idx, dtype=np.int64)
        for name, ix in (("query_idx", self.query_idx), ("base_idx", self.base_idx)):
            if ix.ndim != 1 or (ix.size and (ix.min() < 0 or ix.max() >= n_samples)):
                raise ValueError(f"{name} debe ser un vector de índices en [0, n_samples)")
        self.metrics = tuple(_check_metric(m) for m in metrics)
        self.standardize = standardize
        self.fit_mask = None if fit_mask is None else np.asarray(fit_mask, dtype=bool)
        if self.fit_mask is not None and self.fit_mask.shape != (n_samples,):
            raise ValueError("fit_mask debe tener largo n_samples")
        self.transform = transform
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser ≥ 1")
        nq, nb = self.query_idx.size, self.base_idx.size
        self._need_dot = any(m != "l1" for m in self.metrics)
        self._need_l1 = "l1" in self.metrics
        self.G = np.zeros((nq, nb), dtype=np.float64) if self._need_dot else None
        self.L1 = np.zeros((nq, nb), dtype=np.float64) if self._need_l1 else None
        self.sq_q = np.zeros(nq, dtype=np.float64)
        self.sq_b = np.zeros(nb, dtype=np.float64)
        self.sum_q = np.zeros(nq, dtype=np.float64)
        self.sum_b = np.zeros(nb, dtype=np.float64)
        self.n_features = 0
        self.n_skipped = 0
        self._qbuf = np.empty((self.batch_size, nq), dtype=np.float32)
        self._bbuf = np.empty((self.batch_size, nb), dtype=np.float32)
        self._fill = 0

    def update(self, feature_id: str, x: np.ndarray) -> bool:
        """Agrega un rasgo (un valor por muestra); devuelve False si se descartó."""
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.n_samples,):
            raise ValueError(f"{feature_id}: largo {v.shape}, se esperaba ({self.n_samples},)")
        if self.transform is not None:
            v = np.asarray(self.transform(v), dtype=np.float64)
        z = standardize_feature(v, standardize=self.standardize, fit_mask=self.fit_mask)
        if z is None:
            self.n_skipped += 1
            return False
        z32 = z.astype(np.float32)
        self._qbuf[self._fill] = z32[self.query_idx]
        self._bbuf[self._fill] = z32[self.base_idx]
        self._fill += 1
        self.n_features += 1
        if self._fill == self.batch_size:
            self._flush()
        return True

    def _flush(self) -> None:
        if not self._fill:
            return
        Zq = self._qbuf[: self._fill].astype(np.float64)  # (lote × |Q|)
        Zb = self._bbuf[: self._fill].astype(np.float64)  # (lote × |B|)
        if self._need_dot:
            self.G += Zq.T @ Zb
        self.sq_q += np.einsum("ij,ij->j", Zq, Zq)
        self.sq_b += np.einsum("ij,ij->j", Zb, Zb)
        self.sum_q += Zq.sum(axis=0)
        self.sum_b += Zb.sum(axis=0)
        if self._need_l1:
            l1_accumulate(
                np.ascontiguousarray(self._qbuf[: self._fill].T),
                np.ascontiguousarray(self._bbuf[: self._fill].T),
                self.L1,
            )
        self._fill = 0

    def scores(self, metric: str) -> np.ndarray:
        """Matriz completa (|Q| × |B|) de similitudes o distancias, float64."""
        metric = _check_metric(metric)
        if metric not in self.metrics:
            raise ValueError(f"la métrica {metric!r} no se acumuló (metrics={self.metrics})")
        self._flush()
        if metric == "l1":
            return self.L1.copy()
        G = self.G
        if metric == "l2":
            return np.sqrt(np.maximum(self.sq_q[:, None] + self.sq_b[None, :] - 2.0 * G, 0.0))
        if metric == "cosine":
            num, vq, vb = G, self.sq_q, self.sq_b
        else:  # correlation
            d = max(self.n_features, 1)
            mq, mb = self.sum_q / d, self.sum_b / d
            num = G - d * np.outer(mq, mb)
            vq = np.maximum(self.sq_q - d * mq * mq, 0.0)
            vb = np.maximum(self.sq_b - d * mb * mb, 0.0)
        den = np.sqrt(np.outer(vq, vb))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-24)

    def result(
        self,
        metric: str,
        k: int,
        *,
        exclude_self: bool = False,
        as_sample_idx: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k exacto: ``(idx int64 [|Q|, k], scores float32 [|Q|, k])``.

        ``idx`` son posiciones dentro de ``base_idx`` (como si se hubiera hecho
        ``ExactKNN.fit(Z[base_idx])``); con ``as_sample_idx`` se devuelven los
        índices de muestra originales. ``exclude_self`` descarta los pares en
        que la consulta y el candidato son la misma muestra (útil cuando Q ⊂ B).
        Relleno: -1 y NaN.
        """
        S = self.scores(metric)
        largest = metric in SIMILARITY_METRICS
        if exclude_self:
            same = self.query_idx[:, None] == self.base_idx[None, :]
            S = np.where(same, np.nan, S)  # topk trata NaN como el peor valor
        idx, val = topk(S, k, largest=largest)
        if exclude_self:
            # Si el único candidato que quedaría es la propia muestra, se rellena.
            bad = np.isnan(val)
            idx[bad] = -1
        if as_sample_idx:
            idx = np.where(idx >= 0, self.base_idx[np.maximum(idx, 0)], -1)
        return idx.astype(np.int64), val.astype(np.float32)
