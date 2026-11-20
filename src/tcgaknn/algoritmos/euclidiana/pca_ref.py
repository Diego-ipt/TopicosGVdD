"""PCA + k-NN: referencia dependiente de los datos (plan §3.6, experimento E10).

Responde empíricamente "¿por qué no PCA?". PCA es la reducción lineal óptima
en varianza (Eckart–Young), pero, a diferencia de RP/SH/LSH:

- **Depende de los datos:** las componentes se calculan sobre la base; un
  paciente nuevo cambia (en principio) las componentes y obliga a reajustar.
  ``refit_time`` mide ese costo; RP/SH insertan un paciente en O(d·m) sin
  recalcular nada.
- **No es de una pasada sobre el stream de rasgos:** la SVD necesita la matriz
  n × d (o varias pasadas sobre ella), mientras que Y = Z·Ω se acumula fila de
  rasgo a fila de rasgo (``StreamingProjectionAccumulator``).

Costo (con n < d): SVD exacta O(n²·d); aleatorizada ≈ O(n·d·m) más algunas
iteraciones de potencia (Halko, Martinsson y Tropp, 2011, "Finding structure
with randomness", SIAM Review 53(2):217–288; es lo que implementa
``sklearn.decomposition.PCA(svd_solver="randomized")``).

Geometría: las filas se normalizan a norma 1 (como en la cadena canónica, §2.5)
y se busca con distancia euclidiana en el espacio de las m componentes. Como
PCA centra y luego rota, la distancia entre proyecciones es la distancia entre
las proyecciones ortogonales de los vectores unitarios sobre el subespacio
principal: con m = rango completo coincide exactamente con ‖u − v‖ = √(2 − 2·cos),
y por eso el ranking aproxima el del coseno exacto (verdad de terreno de EX).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ..base import QueryResult, as_float32_2d, l2_normalize, topk


def _sq_euclidean(Q: np.ndarray, Y: np.ndarray, y_sq: np.ndarray) -> np.ndarray:
    """‖q − y‖² para todos los pares, en float64 y recortado a ≥ 0."""
    Qd = Q.astype(np.float64)
    D = (Qd * Qd).sum(axis=1)[:, None] - 2.0 * (Qd @ Y.T) + y_sq[None, :]
    np.maximum(D, 0.0, out=D)
    return D


class PCAKNN:
    """k-NN euclidiano en el espacio de las m componentes principales.

    Parámetros:

    - ``m``: número de componentes; si supera min(n, d) se recorta y el valor
      usado queda en ``m_effective``.
    - ``seed``: ``random_state`` del solver aleatorizado (determinismo).
    - ``svd_solver``: "randomized" (por defecto, plan §3.6), "full", "arpack"…
    - ``keep_data``: guarda la base normalizada para poder llamar a
      ``refit_time`` (cuesta 4·n·d bytes adicionales; no cuenta en ``index_bytes``).
    - ``block``: consultas por bloque al calcular distancias (memoria O(block·n)).

    ``score_kind = "distance"``: ``scores`` son distancias euclidianas en ℝ^m
    (menor es mejor). La búsqueda es lineal sobre los n pacientes, así que
    ``stats['n_candidates'] = n`` por consulta.
    """

    name = "PCA"
    score_kind = "distance"

    def __init__(
        self,
        m: int = 256,
        seed: int = 0,
        svd_solver: str = "randomized",
        *,
        keep_data: bool = True,
        block: int = 1024,
    ):
        if m < 1:
            raise ValueError("m debe ser ≥ 1")
        self.m = int(m)
        self.seed = int(seed)
        self.svd_solver = svd_solver
        self.keep_data = keep_data
        self.block = int(block)
        self.m_effective: int | None = None
        self.fit_time_s: float | None = None
        self.last_refit_time_s: float | None = None
        self.pca_: PCA | None = None
        self.Y_: np.ndarray | None = None
        self._y_sq: np.ndarray | None = None
        self._Xn: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _new_pca(self, n: int, d: int) -> PCA:
        m_eff = min(self.m, n, d)
        if self.svd_solver == "arpack":  # ARPACK exige m < min(n, d)
            m_eff = min(m_eff, min(n, d) - 1)
        if m_eff < 1:
            raise ValueError(f"no se puede ajustar PCA con n={n}, d={d}")
        self.m_effective = m_eff
        return PCA(n_components=m_eff, svd_solver=self.svd_solver, random_state=self.seed)

    def _set_base(self, pca: PCA, Xn: np.ndarray, Y: np.ndarray | None = None) -> None:
        self.pca_ = pca
        self.Y_ = np.ascontiguousarray(pca.transform(Xn) if Y is None else Y, dtype=np.float32)
        Yd = self.Y_.astype(np.float64)
        self._y_sq = (Yd * Yd).sum(axis=1)
        self._Xn = Xn if self.keep_data else None

    def fit(self, X: np.ndarray) -> "PCAKNN":
        """Normaliza las filas, ajusta PCA y proyecta la base; registra ``fit_time_s``."""
        Xn = l2_normalize(X)
        t0 = time.perf_counter()
        pca = self._new_pca(*Xn.shape).fit(Xn)
        self._set_base(pca, Xn)
        self.fit_time_s = time.perf_counter() - t0
        return self

    def transform(self, Q: np.ndarray) -> np.ndarray:
        """Proyección de perfiles nuevos (normalizados) sobre las componentes ya ajustadas: O(d·m)."""
        if self.pca_ is None:
            raise RuntimeError("llama a fit antes de transform")
        return np.asarray(self.pca_.transform(l2_normalize(Q)), dtype=np.float32)

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        if self.Y_ is None or self._y_sq is None:
            raise RuntimeError("llama a fit antes de query")
        t0 = time.perf_counter()
        P = self.transform(as_float32_2d(Q))
        nq, n = P.shape[0], self.Y_.shape[0]
        Yd = self.Y_.astype(np.float64)
        idx = np.full((nq, k), -1, dtype=np.int64)
        scores = np.full((nq, k), np.nan, dtype=np.float32)
        for s in range(0, nq, self.block):
            D = _sq_euclidean(P[s : s + self.block], Yd, self._y_sq)
            i_b, d_b = topk(D, k, largest=False)
            idx[s : s + self.block] = i_b
            scores[s : s + self.block] = np.sqrt(d_b)
        stats: dict[str, Any] = {
            "n_candidates": np.full(nq, n, dtype=np.int64),
            "build_time_s": self.fit_time_s,
            "query_time_s": time.perf_counter() - t0,
            "index_bytes": self.index_bytes(),
            "m_effective": self.m_effective,
        }
        return idx, scores, stats

    def index_bytes(self) -> int:
        """Memoria del índice: componentes (m·d) + media (d) + base proyectada (n·m)."""
        if self.pca_ is None or self.Y_ is None:
            return 0
        return int(self.pca_.components_.nbytes + self.pca_.mean_.nbytes + self.Y_.nbytes)

    def resident_bytes(self) -> int:
        """RAM principal retenida: índice + arreglo de distancias + base para refit.

        `index_bytes()` mantiene su contrato (memoria del índice comprimido),
        y esta propiedad captura el coste de `keep_data=True` que antes faltaba
        al comparar consumo real de memoria entre algoritmos.
        """
        return (self.index_bytes()
                + (0 if self._y_sq is None else self._y_sq.nbytes)
                + (0 if self._Xn is None else self._Xn.nbytes))

    def refit_time(self, X_nuevo: np.ndarray, *, update: bool = False) -> float:
        """Segundos que cuesta reajustar PCA tras agregar ``X_nuevo`` a la base.

        Ajusta un PCA nuevo (mismos m, solver y semilla) sobre [base; X_nuevo] y
        proyecta la base ampliada, que es lo que exige un método dependiente de
        los datos para mantener sus componentes óptimas. Con ``update=True`` el
        índice queda reemplazado por el reajustado; si no, el índice no cambia.
        Requiere ``keep_data=True``. El valor también queda en ``last_refit_time_s``.
        """
        if self._Xn is None:
            raise RuntimeError("refit_time requiere fit previo con keep_data=True")
        Xn_new = l2_normalize(X_nuevo)
        if Xn_new.shape[1] != self._Xn.shape[1]:
            raise ValueError("X_nuevo debe tener las mismas columnas que la base")
        Xall = np.concatenate([self._Xn, Xn_new], axis=0)
        m_prev = self.m_effective
        t0 = time.perf_counter()
        pca = self._new_pca(*Xall.shape).fit(Xall)
        Y_all = pca.transform(Xall)
        elapsed = time.perf_counter() - t0
        if update:
            self._set_base(pca, Xall, Y_all)
        else:
            self.m_effective = m_prev
        self.last_refit_time_s = elapsed
        return elapsed
