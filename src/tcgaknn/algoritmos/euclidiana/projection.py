"""RP: proyección aleatoria de Johnson–Lindenstrauss y k-NN en el espacio proyectado.

- ``jl_min_dim`` / ``jl_eps_for``: la cota de Dasgupta–Gupta y su inversa.
- ``RandomProjection``: x ↦ x·R con R de d × m, en dos variantes:

  * ``gaussian``: R = Ω[:, :m]/√m, entradas N(0, 1/m). Ω es la de ``base``
    (generada por ID de rasgo), así que la proyección en memoria coincide con
    ``StreamingProjectionAccumulator.projections(m)`` y comparte Ω con SimHash
    y LSH.
  * ``achlioptas``: entradas √(3/m)·{+1 con p = 1/6, 0 con p = 2/3, −1 con
    p = 1/6}. Matriz propia, también generada fila a fila por ID de rasgo (con
    una sal distinta de la de Ω) para poder usarla en streaming.

- ``ProjectedKNN``: índice ``KNNIndex`` que proyecta la base (filas
  L2-normalizadas) y busca de forma exhaustiva con distancia euclidiana en ℝ^m.
- ``distortion_stats``: distribución de ρ(u, v) = ‖Ru − Rv‖²/‖u − v‖² (plan §4.2.2).

Garantía (lema JL, versión de Dasgupta y Gupta 2003, Teorema 2.1): si
m ≥ 4·ln n/(ε²/2 − ε³/3) (= 24 ln n/(3ε² − 2ε³)), cada par falla con
probabilidad ≤ 2/n², y por la cota de la unión una R al azar cumple a la vez
para todos los pares (1 − ε)‖u − v‖² ≤ ‖Ru − Rv‖² ≤ (1 + ε)‖u − v‖² con
probabilidad ≥ 1/n (repetir O(n) veces da probabilidad constante). Es una cota
de peor caso: en la práctica la fracción de pares fuera de 1 ± ε es mínima.
Con R gaussiana N(0, 1/m), para un par fijo m·ρ ~ χ²_m, así que E[ρ] = 1 y
Var[ρ] = 2/m. Con Achlioptas (s = 3 en la notación de Li et al.) el cuarto
momento de las entradas es 3, igual que el gaussiano, y la varianza también es
2/m (Li, Hastie y Church 2006, Lema 1: Var = (2 + (s − 3)·Σu⁴/‖u‖⁴)/m).

Referencias:

- S. Dasgupta, A. Gupta. An elementary proof of a theorem of Johnson and
  Lindenstrauss. Random Structures & Algorithms 22(1):60–65, 2003.
- D. Achlioptas. Database-friendly random projections: Johnson–Lindenstrauss
  with binary coins. J. Comput. Syst. Sci. 66(4):671–687, 2003.
- P. Li, T. Hastie, K. Church. Very sparse random projections. KDD 2006.

Nota: ``sklearn.random_projection.johnson_lindenstrauss_min_dim`` usa la misma
expresión pero trunca (``int``) en vez de redondear hacia arriba: da 446, 1033
y 2147 para n = 11.000; aquí se usa ``ceil`` (447, 1034, 2148), que es lo que
exige la desigualdad m ≥ ….
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..base import (
    QueryResult,
    as_float32_2d,
    default_feature_ids,
    feature_seed,
    l2_normalize,
    omega_matrix,
    omega_row,
    topk,
)

KINDS = ("gaussian", "achlioptas")
_ACHLIOPTAS_SALT = "achlioptas"


# --------------------------------------------------------------------------
# Cota de Johnson–Lindenstrauss (Dasgupta–Gupta)
# --------------------------------------------------------------------------


def _jl_denominator(eps: float) -> float:
    return eps**2 / 2 - eps**3 / 3


def jl_min_dim(n: int, eps: float) -> int:
    """m mínimo de Dasgupta–Gupta: ⌈4 ln n / (ε²/2 − ε³/3)⌉, con 0 < ε < 1.

    No depende de d. Ejemplos con n = 11.000: ε = 0,5 → 447; 0,3 → 1.034;
    0,2 → 2.148.
    """
    if not 0 < eps < 1:
        raise ValueError(f"eps debe estar en (0, 1), llegó {eps}")
    if n < 1:
        raise ValueError(f"n debe ser ≥ 1, llegó {n}")
    if n == 1:
        return 1
    return max(1, math.ceil(4 * math.log(n) / _jl_denominator(eps)))


def jl_eps_for(n: int, m: int, *, tol: float = 1e-10) -> float | None:
    """Menor ε ∈ (0, 1) tal que ``jl_min_dim(n, ε) ≤ m`` (inversa de la cota).

    ε²/2 − ε³/3 crece en (0, 1) y vale a lo más 1/6 en ε = 1, así que si
    4 ln n / m > 1/6 no hay ε < 1 que sirva y se devuelve None. Se resuelve por
    bisección; el resultado queda del lado que cumple la cota.
    """
    if m < 1:
        raise ValueError(f"m debe ser ≥ 1, llegó {m}")
    if n <= 1:
        return 0.0
    target = 4 * math.log(n) / m  # se busca ε con ε²/2 − ε³/3 ≥ target
    if target >= 1 / 6:
        return None
    lo, hi = 0.0, 1.0  # f(lo) < target ≤ f(hi)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if _jl_denominator(mid) >= target:
            hi = mid
        else:
            lo = mid
    # Por redondeo de ceil, asegura que la cota se cumple con el ε devuelto.
    while hi < 1 and jl_min_dim(n, hi) > m:
        hi = min(1 - 1e-12, hi + tol)
    return hi if hi < 1 else None


# --------------------------------------------------------------------------
# Matrices de proyección (por ID de rasgo, reproducibles en streaming)
# --------------------------------------------------------------------------


def random_matrix(d: int, cols: int, seed: int, feature_ids: Sequence[str] | None = None) -> np.ndarray:
    """Ω gaussiana N(0, 1) de d × cols (alias de ``omega_matrix`` con IDs por defecto)."""
    ids = default_feature_ids(d) if feature_ids is None else list(feature_ids)
    if len(ids) != d:
        raise ValueError(f"se esperaban {d} IDs de rasgo, llegaron {len(ids)}")
    return omega_matrix(ids, cols, seed)


def project(X: np.ndarray, Omega: np.ndarray, m: int) -> np.ndarray:
    """X · Ω[:, :m]/√m (proyección gaussiana JL), float32."""
    A = as_float32_2d(X)
    if Omega.shape[0] != A.shape[1] or m > Omega.shape[1]:
        raise ValueError(f"Ω {Omega.shape} no sirve para X {A.shape} con m = {m}")
    return (A @ Omega[:, :m]) / np.float32(math.sqrt(m))


def achlioptas_row(feature_id: str, cols: int, seed: int) -> np.ndarray:
    """Fila de signos de Achlioptas (int8 en {−1, 0, +1}, p = 1/6, 2/3, 1/6).

    Se genera con un uniforme por entrada y umbrales 1/6 y 5/6, a partir de una
    semilla derivada de (seed, sal, ID). Las primeras c columnas no dependen de
    ``cols`` (``Generator.random`` consume un entero de 64 bits por valor), igual
    que en Ω. La escala √(3/m) se aplica al proyectar.
    """
    u = np.random.default_rng(feature_seed(f"{_ACHLIOPTAS_SALT}\x00{feature_id}", seed)).random(cols)
    s = np.zeros(cols, dtype=np.int8)
    s[u < 1 / 6] = 1
    s[u >= 5 / 6] = -1
    return s


def achlioptas_matrix(feature_ids: Sequence[str], cols: int, seed: int) -> np.ndarray:
    """Signos de Achlioptas (d × cols, int8); fila j = ``achlioptas_row(ids[j])``."""
    if not len(feature_ids):
        return np.zeros((0, cols), np.int8)
    return np.stack([achlioptas_row(f, cols, seed) for f in feature_ids])


class RandomProjection:
    """Proyección aleatoria x ↦ x·R ∈ ℝ^m, con R de d × m independiente de los datos.

    - ``kind="gaussian"``: R = Ω[:, :m]/√m con Ω = ``omega_matrix(ids, cols, seed)``.
    - ``kind="achlioptas"``: R = S[:, :m]·√(3/m) con S de signos {−1, 0, +1}.

    ``feature_ids`` (largo d) fija qué fila de la matriz corresponde a cada
    rasgo; por defecto ``default_feature_ids(d)``, como el acumulador en
    streaming. ``cols`` (≥ m) es el ancho con que se genera la matriz; como las
    primeras m columnas no dependen de ``cols``, solo afecta a la memoria.

    ``fit`` acepta d (entero) o una matriz X (usa X.shape[1]); no mira los datos.
    ``row(feature_id)`` devuelve la fila de R de un rasgo para acumular
    y += z_j ⊗ R_j en streaming sin guardar R.
    """

    def __init__(
        self,
        m: int,
        kind: str = "gaussian",
        seed: int = 0,
        feature_ids: Sequence[str] | None = None,
        cols: int | None = None,
    ):
        if m < 1:
            raise ValueError(f"m debe ser ≥ 1, llegó {m}")
        if kind not in KINDS:
            raise ValueError(f"kind debe ser uno de {KINDS}, llegó {kind!r}")
        self.m = int(m)
        self.kind = kind
        self.seed = seed
        self.feature_ids = None if feature_ids is None else list(feature_ids)
        self.cols = self.m if cols is None else int(cols)
        if self.cols < self.m:
            raise ValueError(f"cols ({self.cols}) debe ser ≥ m ({self.m})")
        self.d: int | None = None
        self.components_: np.ndarray | None = None  # R (d × m), float32
        self.fit_time_s = 0.0

    @property
    def scale(self) -> float:
        return 1 / math.sqrt(self.m) if self.kind == "gaussian" else math.sqrt(3 / self.m)

    def row(self, feature_id: str) -> np.ndarray:
        """Fila de R (largo m, float32) para un rasgo, sin construir R completa."""
        if self.kind == "gaussian":
            base = omega_row(feature_id, self.cols, self.seed)[: self.m]
        else:
            base = achlioptas_row(feature_id, self.cols, self.seed)[: self.m].astype(np.float32)
        return base * np.float32(self.scale)

    def fit(self, X_or_d: np.ndarray | int) -> "RandomProjection":
        t0 = time.perf_counter()
        d = int(X_or_d) if np.isscalar(X_or_d) else as_float32_2d(X_or_d).shape[1]
        ids = default_feature_ids(d) if self.feature_ids is None else self.feature_ids
        if len(ids) != d:
            raise ValueError(f"feature_ids tiene {len(ids)} IDs pero d = {d}")
        if self.kind == "gaussian":
            base = omega_matrix(ids, self.cols, self.seed)[:, : self.m]
        else:
            base = achlioptas_matrix(ids, self.cols, self.seed)[:, : self.m].astype(np.float32)
        self.components_ = np.ascontiguousarray(base * np.float32(self.scale), dtype=np.float32)
        self.d = d
        self.fit_time_s = time.perf_counter() - t0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X (n × d) ↦ X·R (n × m), float32. X no debe tener NaN (imputar antes)."""
        if self.components_ is None:
            raise RuntimeError("RandomProjection sin fit")
        A = as_float32_2d(X)
        if A.shape[1] != self.d:
            raise ValueError(f"X tiene {A.shape[1]} columnas, se esperaban {self.d}")
        return A @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def nbytes(self) -> int:
        return 0 if self.components_ is None else int(self.components_.nbytes)


# --------------------------------------------------------------------------
# k-NN en el espacio proyectado
# --------------------------------------------------------------------------


class ProjectedKNN:
    """k-NN exhaustivo con distancia euclidiana sobre proyecciones aleatorias.

    ``fit(X)``: normaliza las filas a norma 1 (si ``normalize``) y guarda
    P = X̂·R (n × m, float32). ``query(Q, k)``: proyecta Q igual y devuelve los
    k más cercanos por ‖q' − p'‖, que aproxima √(2 − 2·cos(q, x)) dentro de
    √(1 ± ε). ``score_kind = "distance"`` (menor es mejor).

    Las distancias se calculan como ‖q'‖² + ‖p'‖² − 2·q'·p' (GEMM, por bloques
    de ``query_block`` consultas), recortadas a ≥ 0. ``stats`` trae
    ``n_candidates`` (= n: búsqueda lineal), tiempos de proyección y búsqueda, y
    los bytes de la base proyectada y de R.
    """

    score_kind = "distance"

    def __init__(
        self,
        m: int = 256,
        kind: str = "gaussian",
        seed: int = 0,
        normalize: bool = True,
        feature_ids: Sequence[str] | None = None,
        cols: int | None = None,
        query_block: int = 1024,
    ):
        self.rp = RandomProjection(m, kind=kind, seed=seed, feature_ids=feature_ids, cols=cols)
        self.normalize = normalize
        self.query_block = query_block
        self.name = f"RP-{kind}-m{m}"
        self.P_: np.ndarray | None = None
        self._sq: np.ndarray | None = None
        self.fit_stats: dict[str, Any] = {}

    def _project(self, X: np.ndarray) -> np.ndarray:
        A = as_float32_2d(X)
        # ``l2_normalize`` convierte en cero una fila con NaN (su norma es NaN y
        # no pasa el umbral): el paciente quedaría como vector nulo sin aviso.
        if not np.isfinite(A).all():
            raise ValueError("X tiene NaN o ±inf: imputar antes de proyectar")
        if self.normalize:
            A = l2_normalize(A)
        return self.rp.transform(A)

    def fit(self, X: np.ndarray) -> "ProjectedKNN":
        A = as_float32_2d(X)
        self.rp.fit(A.shape[1])
        t0 = time.perf_counter()
        self.P_ = np.ascontiguousarray(self._project(A))
        self._sq = np.einsum("ij,ij->i", self.P_, self.P_)
        self.fit_stats = {
            "t_matrix_s": self.rp.fit_time_s,
            "t_project_base_s": time.perf_counter() - t0,
            "index_bytes": int(self.P_.nbytes),
            "matrix_bytes": self.rp.nbytes,
        }
        return self

    def transform(self, Q: np.ndarray) -> np.ndarray:
        """Proyección de consultas (con la misma normalización que la base)."""
        return self._project(Q)

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        if self.P_ is None:
            raise RuntimeError("ProjectedKNN sin fit")
        if k < 0:
            raise ValueError(f"k debe ser ≥ 0, llegó {k}")
        t0 = time.perf_counter()
        Qp = self._project(Q)
        t_proj = time.perf_counter() - t0
        nq, n = Qp.shape[0], self.P_.shape[0]
        idx = np.empty((nq, k), dtype=np.int64)
        dist = np.empty((nq, k), dtype=np.float32)
        t0 = time.perf_counter()
        for a in range(0, nq, self.query_block):
            B = Qp[a : a + self.query_block]
            D2 = np.einsum("ij,ij->i", B, B)[:, None] + self._sq[None, :] - 2 * (B @ self.P_.T)
            np.maximum(D2, 0, out=D2)
            ib, vb = topk(D2, k, largest=False)
            idx[a : a + len(B)] = ib
            dist[a : a + len(B)] = np.sqrt(vb)  # NaN del relleno se mantiene
        t_search = time.perf_counter() - t0
        stats = {
            "n_candidates": np.full(nq, n, dtype=np.int64),
            "t_project_s": t_proj,
            "t_search_s": t_search,
            "m": self.rp.m,
            **self.fit_stats,
        }
        return idx, dist, stats


# --------------------------------------------------------------------------
# Distorsión de distancias (validación de JL, plan §4.2.2)
# --------------------------------------------------------------------------


def distortion_stats(
    X: np.ndarray,
    Xp: np.ndarray,
    pairs: np.ndarray | None = None,
    eps_list: Sequence[float] = (0.2, 0.3, 0.5),
    rng: np.random.Generator | int | None = 0,
    *,
    max_pairs: int = 1_000_000,
    chunk: int = 8192,
    return_rho: bool = False,
) -> dict[str, Any]:
    """Estadísticas de ρ(u, v) = ‖Ru − Rv‖²/‖u − v‖² sobre pares de filas.

    - ``X``: puntos originales (n × d); ``Xp``: sus proyecciones (n × m), misma fila.
    - ``pairs``: arreglo (p × 2) de índices; si es None, todos los pares i < j
      cuando n(n − 1)/2 ≤ ``max_pairs``, o ``max_pairs`` pares i ≠ j al azar
      (con ``rng``) en otro caso.
    - Las diferencias se calculan directo (no con ‖u‖² + ‖v‖² − 2u·v, que pierde
      precisión en pares cercanos), en float64 y por bloques de ``chunk`` pares.
      Los pares con ‖u − v‖ = 0 se excluyen y se cuentan en ``n_degenerate``;
      los pares con NaN/inf en X o Xp se excluyen y se cuentan en ``n_nonfinite``.

    Devuelve ``n_pairs``, ``n_degenerate``, ``n_nonfinite``, ``mean``, ``std``, ``p1``, ``p99``,
    ``min``, ``max``, ``eps_p99`` (p99 de |ρ − 1|, el "ε empírico") y
    ``pct_outside`` = {ε: % de pares con ρ fuera de [1 − ε, 1 + ε]}; con
    ``return_rho`` también ``rho`` (para histogramas).
    """
    A = np.asarray(X)
    B = np.asarray(Xp)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError(f"X {A.shape} y Xp {B.shape} deben ser 2D con el mismo n")
    n = A.shape[0]
    if pairs is None:
        if n < 2:
            raise ValueError("se necesitan al menos 2 puntos")
        if n * (n - 1) // 2 <= max_pairs:
            I, J = np.triu_indices(n, 1)
        else:
            g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
            I = g.integers(0, n, size=max_pairs)
            J = g.integers(0, n - 1, size=max_pairs)
            J = J + (J >= I)  # uniforme sobre j ≠ i
    else:
        P = np.asarray(pairs, dtype=np.int64)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("pairs debe tener forma (p, 2)")
        if P.size and (P.min() < 0 or P.max() >= n):  # índices negativos darían la vuelta sin aviso
            raise ValueError(f"pairs tiene índices fuera de [0, {n})")
        I, J = P[:, 0], P[:, 1]
    num = np.empty(len(I), dtype=np.float64)
    den = np.empty(len(I), dtype=np.float64)
    for a in range(0, len(I), chunk):
        i, j = I[a : a + chunk], J[a : a + chunk]
        du = A[i].astype(np.float64) - A[j]
        dp = B[i].astype(np.float64) - B[j]
        den[a : a + chunk] = np.einsum("ij,ij->i", du, du)
        num[a : a + chunk] = np.einsum("ij,ij->i", dp, dp)
    finite = np.isfinite(den) & np.isfinite(num)
    ok = finite & (den > 0)
    rho = num[ok] / den[ok]
    out: dict[str, Any] = {
        "n_pairs": int(rho.size),
        "n_degenerate": int((finite & (den == 0)).sum()),
        "n_nonfinite": int((~finite).sum()),
    }
    if rho.size == 0:
        out.update(mean=np.nan, std=np.nan, p1=np.nan, p99=np.nan, min=np.nan, max=np.nan, eps_p99=np.nan,
                   pct_outside={float(e): np.nan for e in eps_list})
    else:
        p1, p99 = np.quantile(rho, [0.01, 0.99])
        dev = np.abs(rho - 1)
        out.update(
            mean=float(rho.mean()),
            std=float(rho.std()),
            p1=float(p1),
            p99=float(p99),
            min=float(rho.min()),
            max=float(rho.max()),
            eps_p99=float(np.quantile(dev, 0.99)),
            pct_outside={float(e): float(100 * (dev > e).mean()) for e in eps_list},
        )
    if return_rho:
        out["rho"] = rho
    return out
