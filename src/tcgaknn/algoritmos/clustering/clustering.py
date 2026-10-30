"""Mini-batch k-means (MBK) sobre representaciones compactas (plan §3.8).

- ``kmeans_plusplus_init``: siembra k-means++ por muestreo D² (Arthur y
  Vassilvitskii, 2007, "k-means++: the advantages of careful seeding", SODA,
  pp. 1027–1035), con la variante "greedy" de scikit-learn (varios candidatos
  por paso y se queda con el que más baja el potencial).
- ``MiniBatchKMeans``: implementación propia del Algoritmo 1 de Sculley (2010),
  "Web-scale k-means clustering", WWW '10, pp. 1177–1178,
  doi:10.1145/1772690.1772862. Cada centro c lleva un conteo v[c] y se mueve
  hacia cada punto asignado con tasa η = 1/v[c]; ``partial_fit`` procesa un
  lote, así que sirve para un stream de pacientes (filas de proyecciones RP o
  de firmas SH que llegan de a poco). Memoria O(K·m) más el lote.
- ``hamming_to_pm1``: firmas SimHash → {−1, +1}^b, donde ‖x − y‖² = 4·Ham(x, y),
  de modo que k-means euclidiano agrupa por distancia de Hamming.
- ``ari``: índice de Rand ajustado (Hubert y Arabie, 1985) vía scikit-learn.

Actualización vectorizada (equivalente exacta al bucle de Sculley). En el
algoritmo, las asignaciones del lote se calculan primero (con los centros
viejos) y luego cada punto x asignado a c hace v[c] += 1; c ← (1 − 1/v[c])·c + x/v[c].
Esa recurrencia es una media corriente, así que tras los n_c puntos del lote:

    c_nuevo = (v_viejo·c_viejo + Σ x) / (v_viejo + n_c)

que se calcula para todos los centros a la vez con ``np.add.at``/bincount.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..base import as_float32_2d

_BLOCK = 4096  # filas por bloque al calcular distancias a los centros


def _sq_dists(X: np.ndarray, C: np.ndarray, c_sq: np.ndarray | None = None) -> np.ndarray:
    """‖x − c‖² (float64, recortado a ≥ 0) entre filas de X y de C."""
    Xd = X.astype(np.float64, copy=False)
    Cd = C.astype(np.float64, copy=False)
    if c_sq is None:
        c_sq = (Cd * Cd).sum(axis=1)
    D = (Xd * Xd).sum(axis=1)[:, None] - 2.0 * (Xd @ Cd.T) + c_sq[None, :]
    np.maximum(D, 0.0, out=D)
    return D


def _check_finite(A: np.ndarray) -> np.ndarray:
    """Rechaza NaN/inf: romperían el muestreo D² y las medias de los centros."""
    if not np.isfinite(A).all():
        raise ValueError("X contiene NaN o infinitos")
    return A


def _assign(X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centro más cercano (empates → índice menor) y distancia² por fila, en bloques."""
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int64)
    dmin = np.empty(n, dtype=np.float64)
    Cd = C.astype(np.float64, copy=False)
    c_sq = (Cd * Cd).sum(axis=1)
    for s in range(0, n, _BLOCK):
        D = _sq_dists(X[s : s + _BLOCK], Cd, c_sq)
        lab = D.argmin(axis=1)
        labels[s : s + _BLOCK] = lab
        dmin[s : s + _BLOCK] = D[np.arange(D.shape[0]), lab]
    return labels, dmin


def kmeans_plusplus_init(
    X: np.ndarray,
    K: int,
    rng: np.random.Generator,
    *,
    n_local_trials: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Siembra k-means++; devuelve ``(centros float32 [K, d], índices int64 [K])``.

    El primer centro es uniforme; cada siguiente se muestrea con probabilidad
    proporcional a D(x)², la distancia² al centro elegido más cercano. Con
    ``n_local_trials`` > 1 se muestrean varios candidatos y se elige el que
    minimiza el potencial Σ D²; por defecto 2 + ⌊ln K⌋, como scikit-learn
    (``n_local_trials=1`` es el k-means++ original). Si todas las D² son 0
    (menos puntos distintos que K), se completa con puntos uniformes.
    Costo O(n·d·K·n_local_trials).
    """
    A = _check_finite(as_float32_2d(X))
    n = A.shape[0]
    if not 1 <= K <= n:
        raise ValueError(f"K={K} debe estar entre 1 y n={n}")
    trials = 2 + int(np.log(K)) if n_local_trials is None else int(n_local_trials)
    if trials < 1:
        raise ValueError("n_local_trials debe ser ≥ 1")
    idx = np.empty(K, dtype=np.int64)
    idx[0] = rng.integers(n)
    closest = _sq_dists(A, A[idx[:1]])[:, 0]
    for c in range(1, K):
        pot = closest.sum()
        if pot <= 0.0:
            cand = rng.integers(n, size=trials)
        else:
            cum = np.cumsum(closest)
            cand = np.searchsorted(cum, rng.random(trials) * cum[-1], side="right")
            cand = np.minimum(cand, n - 1)
        D_cand = np.minimum(closest[None, :], _sq_dists(A[cand], A).astype(np.float64))
        best = int(np.argmin(D_cand.sum(axis=1)))
        idx[c] = cand[best]
        closest = D_cand[best]
    return A[idx].copy(), idx


class MiniBatchKMeans:
    """Mini-batch k-means de Sculley (2010) con tasa por centro 1/v[c].

    Parámetros: ``K`` clusters, ``batch_size`` B, ``n_iter`` iteraciones de
    ``fit`` (cada una con un lote de B pacientes muestreado sin reemplazo),
    ``seed`` y ``tol``. Con ``tol`` > 0, ``fit`` se detiene cuando el
    desplazamiento cuadrático total de los centros en una iteración es
    ≤ tol·(varianza media por coordenada de X), el mismo criterio relativo que
    usa scikit-learn. ``init`` puede ser "k-means++" o una matriz K × m de
    centros iniciales. ``n_init`` (solo ``fit``) repite siembra + iteraciones
    y se queda con la corrida de menor inercia sobre X: k-means converge a
    óptimos locales (p. ej. dos nubes fusionadas y otra partida en dos) aun con
    k-means++, y reiniciar es el remedio estándar (``n_init`` de scikit-learn).

    Costo por iteración O(B·K·m); memoria O(K·m). Atributos tras ajustar:
    ``centers_`` (float32), ``counts_`` (v[c]), ``n_iter_``, ``n_seen_``,
    ``fit_time_s``; tras ``fit``, además ``labels_`` e ``inertia_``.
    """

    name = "MBK"

    def __init__(
        self,
        K: int,
        batch_size: int = 256,
        n_iter: int = 100,
        seed: int = 0,
        tol: float = 0.0,
        *,
        init: str | np.ndarray = "k-means++",
        n_init: int = 1,
        n_local_trials: int | None = None,
    ):
        if K < 1 or batch_size < 1 or n_iter < 0 or n_init < 1:
            raise ValueError("K, batch_size y n_init deben ser ≥ 1, n_iter ≥ 0")
        self.K = int(K)
        self.batch_size = int(batch_size)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.tol = float(tol)
        self.init = init
        self.n_init = int(n_init)
        self.n_local_trials = n_local_trials
        self.rng = np.random.default_rng(seed)
        self._C: np.ndarray | None = None  # centros en float64
        self.counts_: np.ndarray | None = None
        self.n_iter_ = 0
        self.n_seen_ = 0
        self.fit_time_s: float | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    # ------------------------------------------------------------------
    @property
    def centers_(self) -> np.ndarray:
        if self._C is None:
            raise RuntimeError("el modelo no está ajustado")
        return self._C.astype(np.float32)

    def _initialize(self, X: np.ndarray) -> None:
        if isinstance(self.init, str):
            if self.init != "k-means++":
                raise ValueError(f"init desconocido: {self.init!r}")
            if X.shape[0] < self.K:
                raise ValueError(f"se necesitan ≥ K={self.K} filas para inicializar; llegaron {X.shape[0]}")
            C, _ = kmeans_plusplus_init(X, self.K, self.rng, n_local_trials=self.n_local_trials)
        else:
            C = as_float32_2d(self.init)
            if C.shape != (self.K, X.shape[1]):
                raise ValueError(f"init debe tener forma ({self.K}, {X.shape[1]})")
        self._C = C.astype(np.float64)
        self.counts_ = np.zeros(self.K, dtype=np.int64)

    def _step(self, B: np.ndarray) -> float:
        """Una iteración de Sculley sobre el lote B; devuelve Σ‖Δc‖²."""
        assert self._C is not None and self.counts_ is not None
        labels, _ = _assign(B, self._C)
        n_c = np.bincount(labels, minlength=self.K)
        sums = np.zeros_like(self._C)
        np.add.at(sums, labels, B.astype(np.float64))
        new_counts = self.counts_ + n_c
        hit = n_c > 0
        old = self._C[hit].copy()
        self._C[hit] = (self.counts_[hit, None] * old + sums[hit]) / new_counts[hit, None]
        self.counts_ = new_counts
        self.n_seen_ += B.shape[0]
        self.n_iter_ += 1
        return float(((self._C[hit] - old) ** 2).sum())

    def partial_fit(self, X_batch: np.ndarray) -> "MiniBatchKMeans":
        """Procesa un lote del stream (todas sus filas, en una iteración).

        En la primera llamada inicializa los centros con k-means++ sobre ese
        lote (necesita ≥ K filas) o con ``init`` si es una matriz.
        """
        B = _check_finite(as_float32_2d(X_batch))
        if self._C is None:
            self._initialize(B)
        elif B.shape[1] != self._C.shape[1]:
            raise ValueError("el lote no tiene la dimensión de los centros")
        t0 = time.perf_counter()
        self._step(B)
        self.fit_time_s = (self.fit_time_s or 0.0) + time.perf_counter() - t0
        return self

    def fit(self, X: np.ndarray) -> "MiniBatchKMeans":
        """Ajuste en memoria: k-means++ sobre X y ``n_iter`` lotes aleatorios.

        Con ``n_init`` > 1 repite todo y conserva la corrida de menor inercia.
        """
        A = _check_finite(as_float32_2d(X))
        n = A.shape[0]
        t0 = time.perf_counter()
        thr = self.tol * float(A.astype(np.float64).var(axis=0).mean()) if self.tol > 0 else -1.0
        bs = min(self.batch_size, n)
        best: tuple | None = None
        n_init = 1 if not isinstance(self.init, str) else self.n_init  # centros dados: una corrida
        for _ in range(n_init):
            self._C = None
            self.n_iter_ = self.n_seen_ = 0
            self._initialize(A)
            for _ in range(self.n_iter):
                sel = self.rng.choice(n, size=bs, replace=False)
                if self._step(A[sel]) <= thr:
                    break
            labels, dmin = _assign(A, self._C)
            inertia = float(dmin.sum())
            if best is None or inertia < best[0]:
                best = (inertia, labels, self._C.copy(), self.counts_.copy(), self.n_iter_, self.n_seen_)
        assert best is not None
        self.inertia_, self.labels_, self._C, self.counts_, self.n_iter_, self.n_seen_ = best
        self.fit_time_s = time.perf_counter() - t0
        return self

    def fit_stream(self, batches: Iterable[np.ndarray]) -> "MiniBatchKMeans":
        """Atajo: ``partial_fit`` sobre cada lote de un iterable."""
        for B in batches:
            self.partial_fit(B)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Cluster más cercano de cada fila (int64)."""
        if self._C is None:
            raise RuntimeError("el modelo no está ajustado")
        return _assign(as_float32_2d(X), self._C)[0]

    def inertia(self, X: np.ndarray) -> float:
        """Σ ‖x − c(x)‖² sobre las filas de X con los centros actuales."""
        if self._C is None:
            raise RuntimeError("el modelo no está ajustado")
        return float(_assign(as_float32_2d(X), self._C)[1].sum())


def hamming_to_pm1(bits: np.ndarray, b: int | None = None, *, packed: bool | None = None) -> np.ndarray:
    """Firmas SimHash → matriz float32 en {−1, +1} (bit 1 → +1, bit 0 → −1).

    Acepta bits como bool/0-1 (n × b) o empaquetados en uint64 (n × b/64), con
    el empaquetado de ``simhash.signatures`` (``np.packbits(..., bitorder="little")``
    visto como uint64). ``packed=None`` decide por el dtype (uint64 ⇒ empaquetado);
    ``b`` recorta los bits útiles al desempaquetar.

    Para x, y ∈ {−1, +1}^b cada bit distinto aporta (±2)² = 4, así que
    ‖x − y‖² = 4·Ham(x, y) y ⟨x, y⟩ = b − 2·Ham(x, y).
    """
    A = np.asarray(bits)
    if A.ndim == 1:
        A = A[None, :]
    if packed is None:
        packed = A.dtype == np.uint64
    if packed:
        if A.dtype != np.uint64:
            raise ValueError("las firmas empaquetadas deben ser uint64")
        # Orden de bytes little-endian explícito, igual que simhash (independiente de la plataforma).
        U = np.ascontiguousarray(A.astype("<u8", copy=False))
        B = np.unpackbits(U.view(np.uint8), axis=1, bitorder="little").astype(bool)
        if b is not None:
            if b > B.shape[1]:
                raise ValueError(f"b={b} supera los {B.shape[1]} bits empaquetados")
            B = B[:, :b]
    else:
        if A.dtype != bool:
            if np.any((A != 0) & (A != 1)):
                raise ValueError("bits no empaquetados deben ser 0/1 o bool")
            A = A.astype(bool)
        if b is not None and b > A.shape[1]:
            raise ValueError(f"b={b} supera los {A.shape[1]} bits disponibles")
        B = A if b is None else A[:, :b]
    return np.where(B, np.float32(1.0), np.float32(-1.0))


def ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Índice de Rand ajustado (Hubert y Arabie, 1985): 1 = particiones iguales, ≈ 0 = azar."""
    t = np.asarray(labels_true).ravel()
    p = np.asarray(labels_pred).ravel()
    if t.shape != p.shape:
        raise ValueError("labels_true y labels_pred deben tener el mismo largo")
    return float(adjusted_rand_score(t, p))
