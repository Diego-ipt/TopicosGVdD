"""E2LSH: LSH p-estable para distancias ℓ2 y ℓ1 (Datar, Immorlica, Indyk y Mirrokni, 2004).

Función hash (plan §3.5; el paper llama r al ancho, aquí es **w**, y b al
desplazamiento, aquí es **β**):

    h_{a,β}(v) = ⌊(a·v + β)/w⌋,   β ~ U(0, w)

- p = 2: a ~ N(0, I_d) (la gaussiana es 2-estable) → hash para ‖·‖₂.
- p = 1: a con entradas Cauchy estándar (1-estable) → hash para ‖·‖₁.

Como a es p-estable, a·u − a·v se distribuye como ‖u − v‖ₚ·X, con X p-estable.
Por eso la probabilidad de colisión depende solo de c = ‖u − v‖ₚ:

    p(c) = ∫₀^w (1/c)·f_p(t/c)·(1 − t/w) dt,   f_p = densidad de |X|

y es decreciente en c. Formas cerradas (paper, §4.1, con u = w/c):

- Gaussiana: p(c) = 1 − 2Φ(−u) − (2/(√(2π)·u))·(1 − e^{−u²/2})
- Cauchy:    p(c) = (2/π)·atan(u) − (1/(π·u))·ln(1 + u²)

Índice: r funciones concatenadas por tabla (la clave de la tabla es la tupla de
r enteros, análogo exacto de los r bits por banda de LSH) y L tablas. Un punto
es **candidato** si comparte bucket con la consulta en al menos una tabla:

    P_cand(c) = 1 − (1 − p(c)^r)^L

Los candidatos se re-ordenan con la distancia ℓp exacta (``rerank="d"``), con la
distancia estimada desde las proyecciones (``rerank="proj"``) o por número de
tablas en que colisionan (``rerank="none"``). Política de pocos candidatos
idéntica a LSH (§3.4): sin respaldo; si |C| < k se devuelven los |C|
disponibles y el resto se rellena con -1 / NaN.

Streaming: la fila de la matriz A = [a_1 … a_{r·L}] que corresponde a un rasgo
se genera desde (semilla, ID de rasgo), igual que Ω en ``base``. Como a·v es
lineal, Y = X·A se acumula fila de rasgo a fila de rasgo (Y += x_j ⊗ A_j) sin
guardar A ni X, y luego ``fit_projections(Y)`` arma las tablas. Para p = 2,
``A_j = omega_row(id_j, r·L, a_seed)``, así que sirve directamente
``StreamingProjectionAccumulator(n, r·L, seed=index.a_seed)`` (``raw()`` = Z·A).

Referencias:

- M. Datar, N. Immorlica, P. Indyk, V. S. Mirrokni. "Locality-Sensitive Hashing
  Scheme Based on p-Stable Distributions". SCG 2004, pp. 253–262.
  DOI 10.1145/997817.997857. Fórmulas cerradas de p(c) en §4.1; experimentos
  con R = 1 y w = 4 en §5; ρ poco sensible a w pasado cierto punto (Fig. 2).
- A. Andoni, P. Indyk. "E2LSH 0.1 User Manual" (2005): w = 4 por defecto con R = 1.
- P. Indyk. "Stable distributions, pseudorandom generators, embeddings, and data
  stream computation". J. ACM 53(3), 2006: estimador de ‖·‖₁ por la mediana de
  |proyecciones Cauchy| (usado en ``rerank="proj"`` con p = 1).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..base import QueryResult, as_float32_2d, default_feature_ids, feature_seed, omega_row, topk

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_CODE_CLIP = float(2**62)


# --------------------------------------------------------------------------
# Teoría: probabilidad de colisión, ρ y curva de candidatos
# --------------------------------------------------------------------------


def collision_probability(c: np.ndarray | float, w: float, p: int = 2) -> np.ndarray | float:
    """Probabilidad de que una función h_{a,β} haga colisionar dos puntos a distancia ℓp = c.

    Formas cerradas de Datar et al. (2004), §4.1, con u = w/c. Vectorizada en c;
    c ≤ 0 (puntos iguales) da 1. Devuelve float si c es escalar.
    """
    if w <= 0:
        raise ValueError("w debe ser positivo")
    cc = np.asarray(c, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(cc > 0, w / np.where(cc > 0, cc, 1.0), np.inf)
        if p == 2:
            # 1 − 2Φ(−u) − 2/(√(2π)·u)·(1 − e^{−u²/2}); −expm1 evita cancelación con u chico
            prob = 1.0 - 2.0 * ndtr(-u) - 2.0 / (np.sqrt(2.0 * np.pi) * u) * (-np.expm1(-0.5 * u * u))
        elif p == 1:
            # (2/π)·atan(u) − ln(1 + u²)/(π·u)
            prob = (2.0 / np.pi) * np.arctan(u) - np.log1p(u * u) / (np.pi * u)
        else:
            raise ValueError("p debe ser 1 (Cauchy) o 2 (gaussiana)")
    prob = np.where(np.isinf(u), 1.0, np.clip(prob, 0.0, 1.0))
    return float(prob) if prob.ndim == 0 else prob


def rho(c: np.ndarray | float, w: float, p: int = 2) -> np.ndarray | float:
    """Exponente ρ = ln(1/p(1)) / ln(1/p(c)) (con R = 1; c es el factor de aproximación).

    El índice LSH usa O(n^ρ) evaluaciones de distancia; el paper muestra que para
    ℓ2 el mínimo en w de ρ queda estrictamente bajo 1/c, y para ℓ1 cerca (y por
    encima) de 1/c, acercándose a medida que w crece.
    """
    p1 = collision_probability(1.0, w, p)
    pc = np.asarray(collision_probability(c, w, p), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(p1) / np.log(pc)
    return float(out) if np.ndim(out) == 0 else out


def candidate_probability(c: np.ndarray | float, w: float, p: int, r: int, L: int) -> np.ndarray | float:
    """Curva S del índice: P(ser candidato) = 1 − (1 − p(c)^r)^L."""
    pc = np.asarray(collision_probability(c, w, p), dtype=np.float64)
    out = 1.0 - (1.0 - pc**r) ** L
    return float(out) if out.ndim == 0 else out


def suggest_w(distances_sample: np.ndarray, *, quantile: float = 0.5, factor: float = 4.0) -> float:
    """Heurística para el ancho w a partir de una muestra de distancias "cercanas".

    ``distances_sample`` debe ser una muestra de distancias ℓp entre consultas y
    sus vecinos verdaderos (p. ej. la distancia al k-ésimo vecino exacto en E0).
    Se toma la escala R = cuantil ``quantile`` de esa muestra (mediana por
    defecto) y se devuelve w = ``factor``·R.

    Justificación: con R normalizado a 1, Datar et al. (2004, §5) y el manual de
    E2LSH usan w = 4; el paper muestra (Fig. 2) que ρ es poco sensible a w una
    vez que w está "suficientemente" lejos de 0, pero un w demasiado grande acerca
    p(1) y p(c) a 1 y obliga a usar más funciones por tabla. Con w = 4R, un
    vecino a distancia R colisiona por función con p ≈ 0,80 (ℓ2) o ≈ 0,62 (ℓ1).
    Se ignoran valores no finitos y no positivos.
    """
    D = np.asarray(distances_sample, dtype=np.float64).ravel()
    D = D[np.isfinite(D) & (D > 0)]
    if D.size == 0:
        raise ValueError("la muestra de distancias no tiene valores positivos finitos")
    if not 0.0 <= quantile <= 1.0 or factor <= 0:
        raise ValueError("quantile debe estar en [0, 1] y factor ser positivo")
    return float(factor * np.quantile(D, quantile))


# --------------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------------


def _mix64(x: np.ndarray) -> np.ndarray:
    """Finalizador de splitmix64 (biyección de uint64; la aritmética da la vuelta en 2⁶⁴)."""
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def _bucket_keys(codes: np.ndarray) -> np.ndarray:
    """Clave uint64 de cada (punto, tabla) a partir de las r coordenadas enteras.

    codes: int64 (n, L, r). Se encadena h ← mix(h ⊕ código_j). Con r = 1 la clave
    es biyectiva; con r > 1 dos tuplas distintas chocan con probabilidad ~2⁻⁶⁴.
    """
    n, L, r = codes.shape
    h = np.full((n, L), _GOLDEN, dtype=np.uint64)
    u = np.ascontiguousarray(codes).view(np.uint64)
    for j in range(r):
        h = _mix64(h ^ u[:, :, j])
    return h


def _lp_dist(q: np.ndarray, Xc: np.ndarray, p: int, chunk_elems: int = 1 << 22) -> np.ndarray:
    """Distancia ℓp exacta (float64) de q a cada fila de Xc, en bloques de ~``chunk_elems``."""
    m, d = Xc.shape
    out = np.empty(m, dtype=np.float64)
    step = max(1, chunk_elems // max(d, 1))
    q64 = q.astype(np.float64)
    for s in range(0, m, step):
        D = Xc[s : s + step].astype(np.float64) - q64
        if p == 2:
            out[s : s + step] = np.sqrt(np.einsum("ij,ij->i", D, D))
        else:
            out[s : s + step] = np.abs(D).sum(axis=1)
    return out


# --------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------


class E2LSH:
    """Índice LSH p-estable con L tablas de r funciones h_{a,β} concatenadas.

    Parámetros:

    - ``p``: 2 (ℓ2, a gaussiana) o 1 (ℓ1, a Cauchy). ``score_kind = "distance"``.
    - ``w``: ancho del bucket, en las mismas unidades que la distancia (ver ``suggest_w``).
    - ``r``: funciones por tabla; ``L``: número de tablas.
    - ``seed``: semilla de A (por ID de rasgo) y de β.
    - ``rerank``: ``True``/``"d"`` (ℓp exacta en el espacio original; guarda X),
      ``"proj"`` (ℓp estimada desde Y = X·A: ‖ΔY‖₂/√(rL) para p = 2 y
      mediana |ΔY| para p = 1; no guarda X) o ``False``/``"none"`` (orden por
      número de tablas en que coincide; score = 1 − coincidencias/L).
    - ``by_feature_id``: si True (por defecto), la fila j de A depende solo de
      (semilla, ID del rasgo j), como Ω en ``base``; así la versión en memoria y
      la acumulada en streaming coinciden. Si False, A sale de un solo generador
      (más rápido con d grande, pero no acumulable por ID).

    Costos (plan §3.5): construcción O(n·d·r·L); consulta O(d·r·L + L·log n + |C|·d);
    memoria L·n IDs (int32) + L·n claves (uint64) (+ X o Y según ``rerank``).
    """

    score_kind = "distance"

    def __init__(
        self,
        p: int = 2,
        w: float = 4.0,
        r: int = 4,
        L: int = 16,
        seed: int = 0,
        rerank: bool | str = True,
        *,
        by_feature_id: bool = True,
        salt: str = "e2lsh",
    ):
        if p not in (1, 2):
            raise ValueError("p debe ser 1 (Cauchy) o 2 (gaussiana)")
        if not w > 0 or r < 1 or L < 1:
            raise ValueError("se requiere w > 0, r ≥ 1 y L ≥ 1")
        modes = {True: "d", False: "none", "d": "d", "proj": "proj", "none": "none"}
        if rerank not in modes:
            raise ValueError("rerank debe ser True/'d', 'proj' o False/'none'")
        self.p, self.w, self.r, self.L, self.seed = p, float(w), int(r), int(L), int(seed)
        self.rerank = modes[rerank]
        self.by_feature_id = by_feature_id
        self.name = f"E2LSH-L{p}"
        # Semilla de A (derivada con un salt para no repetir la Ω de RP/SH) y β
        self.a_seed = feature_seed(f"{salt}|a", self.seed)
        self.beta = np.random.default_rng(feature_seed(f"{salt}|beta", self.seed)).uniform(0.0, self.w, self.n_funcs)
        self.feature_ids: list[str] | None = None
        self._A: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._keys_sorted: np.ndarray | None = None  # (L, n) uint64
        self._order: np.ndarray | None = None  # (L, n) int32
        self.n = 0
        self.build_time_s = 0.0

    # ---------------- matriz A y proyecciones ----------------

    @property
    def n_funcs(self) -> int:
        return self.r * self.L

    def a_row(self, feature_id: str) -> np.ndarray:
        """Fila de A (float32, largo r·L) para un rasgo; reproducible sin guardar A.

        p = 2 → ``omega_row(feature_id, r·L, a_seed)`` (N(0, 1)); p = 1 → Cauchy
        estándar con el mismo generador por ID.
        """
        if self.p == 2:
            return omega_row(feature_id, self.n_funcs, self.a_seed)
        rng = np.random.default_rng(feature_seed(feature_id, self.a_seed))
        return rng.standard_cauchy(self.n_funcs).astype(np.float32)

    def a_matrix(self, feature_ids: Sequence[str]) -> np.ndarray:
        """A completa (d × r·L, float32); fila j = ``a_row(feature_ids[j])``."""
        if not len(feature_ids):
            return np.zeros((0, self.n_funcs), np.float32)
        return np.stack([self.a_row(f) for f in feature_ids])

    def _ensure_A(self, d: int) -> np.ndarray:
        if self._A is not None:
            if self._A.shape[0] != d:
                raise ValueError(f"dimensión {d} distinta de la del índice ({self._A.shape[0]})")
            return self._A
        if self.by_feature_id:
            ids = self.feature_ids if self.feature_ids is not None else default_feature_ids(d)
            if len(ids) != d:
                raise ValueError(f"se dieron {len(ids)} IDs de rasgo para d = {d}")
            self._A = self.a_matrix(ids)
        else:
            rng = np.random.default_rng(self.a_seed)
            if self.p == 2:
                self._A = rng.standard_normal((d, self.n_funcs), dtype=np.float32)
            else:
                self._A = rng.standard_cauchy((d, self.n_funcs)).astype(np.float32)
        return self._A

    def projections(self, X: np.ndarray, block: int = 1024) -> np.ndarray:
        """Y = X·A (float64, n × r·L), las proyecciones a·v antes de sumar β y dividir por w.

        Se calcula en bloques de filas con X y A en float32 promovidos a float64,
        igual que ``StreamingProjectionAccumulator`` (misma Y salvo el orden de suma).
        """
        Xf = as_float32_2d(X)
        A = self._ensure_A(Xf.shape[1]).astype(np.float64)
        Y = np.empty((Xf.shape[0], self.n_funcs), dtype=np.float64)
        for s in range(0, Xf.shape[0], block):
            Y[s : s + block] = Xf[s : s + block].astype(np.float64) @ A
        if not np.isfinite(Y).all():
            raise ValueError("las proyecciones tienen NaN/inf: imputa los NA antes de indexar")
        return Y

    def codes_from_projections(self, Y: np.ndarray) -> np.ndarray:
        """Códigos enteros ⌊(a·v + β)/w⌋ (int64, n × L × r) desde Y = X·A."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[1] != self.n_funcs:
            raise ValueError(f"Y debe tener forma (n, {self.n_funcs})")
        if not np.isfinite(Y).all():
            # NaN pasado a int64 da un código basura silencioso (mismo bucket para todos)
            raise ValueError("las proyecciones tienen NaN/inf: imputa los NA antes de indexar")
        G =np.floor((Y + self.beta) / self.w)
        np.clip(G, -_CODE_CLIP, _CODE_CLIP, out=G)
        return G.astype(np.int64).reshape(Y.shape[0], self.L, self.r)

    def hash_codes(self, X: np.ndarray) -> np.ndarray:
        """Códigos enteros de cada punto en cada función (int64, n × L × r)."""
        return self.codes_from_projections(self.projections(X))

    # ---------------- construcción ----------------

    def fit(self, X: np.ndarray, feature_ids: Sequence[str] | None = None) -> "E2LSH":
        """Indexa X (n × d). ``feature_ids`` (largo d) fija los IDs que generan A."""
        t0 = time.perf_counter()
        Xf = as_float32_2d(X)
        if feature_ids is not None:
            self.feature_ids = list(feature_ids)
        self._A = None
        Y = self.projections(Xf)
        self._build(Y, Xf)
        self.build_time_s = time.perf_counter() - t0
        return self

    def fit_projections(
        self,
        Y: np.ndarray,
        X: np.ndarray | None = None,
        feature_ids: Sequence[str] | None = None,
        *,
        rerank_projections: np.ndarray | None = None,
    ) -> "E2LSH":
        """Indexa desde proyecciones ya acumuladas Y = X·A (n × r·L), p. ej. en streaming.

        ``X`` solo hace falta con ``rerank="d"``. ``feature_ids`` permite luego
        consultar con vectores crudos (``query``); si no, usar ``query_projections``.
        ``rerank_projections`` permite utilizar un número mayor de columnas
        ya acumuladas (sin emplearlas para construir las tablas hash).
        """
        t0 = time.perf_counter()
        if feature_ids is not None:
            self.feature_ids = list(feature_ids)
            self._A = None
        Xf = None if X is None else as_float32_2d(X)
        if self.rerank == "d" and Xf is None:
            raise ValueError("rerank='d' necesita X para la distancia exacta")
        self._build(np.asarray(Y, dtype=np.float64), Xf,
                    rerank_projections=rerank_projections)
        self.build_time_s = time.perf_counter() - t0
        return self

    def _build(self, Y: np.ndarray, X: np.ndarray | None,
               rerank_projections: np.ndarray | None = None) -> None:
        if X is not None and X.shape[0] != Y.shape[0]:
            raise ValueError("X e Y deben tener el mismo número de filas")
        keys = _bucket_keys(self.codes_from_projections(Y))  # (n, L)
        self.n = Y.shape[0]
        order = np.argsort(keys, axis=0, kind="stable").T  # (L, n): estable → IDs crecientes por bucket
        self._order = np.ascontiguousarray(order, dtype=np.int32)
        self._keys_sorted = np.ascontiguousarray(np.take_along_axis(keys, order.T, axis=0).T)
        self._X = X if self.rerank == "d" else None
        if self.rerank == "proj":
            rerank_y = Y if rerank_projections is None else np.asarray(rerank_projections, dtype=np.float64)
            if (rerank_y.ndim != 2 or rerank_y.shape[0] != self.n or
                    rerank_y.shape[1] < self.n_funcs or not np.isfinite(rerank_y).all()):
                raise ValueError("rerank_projections requiere N filas, >= L*r columnas finitas")
            self._Y = np.ascontiguousarray(rerank_y, dtype=np.float32)
        else:
            if rerank_projections is not None:
                raise ValueError("rerank_projections solo es válido con rerank='proj'")
            self._Y = None

    # ---------------- consulta ----------------

    def query(self, Q: np.ndarray, k: int) -> QueryResult:
        """k vecinos aproximados de cada fila de Q (ver ``KNNIndex``)."""
        Qf = as_float32_2d(Q)
        return self._query(self.projections(Qf), Qf, k)

    def query_projections(self, Yq: np.ndarray, k: int, Q: np.ndarray | None = None,
                          *, rerank_query_projections: np.ndarray | None = None) -> QueryResult:
        """Consulta usando Yq para el hash y opcionalmente más columnas para re-ranking."""
        Qf = None if Q is None else as_float32_2d(Q)
        if self.rerank == "d" and Qf is None:
            raise ValueError("rerank='d' necesita Q para la distancia exacta")
        return self._query(np.atleast_2d(np.asarray(Yq, dtype=np.float64)), Qf, k,
                           rerank_Yq=rerank_query_projections)

    def candidates_from_projections(self, Yq: np.ndarray) -> list[np.ndarray]:
        """Lista (por consulta) de IDs candidatos con repetición: uno por tabla en que coincide."""
        if self._keys_sorted is None or self._order is None:
            raise RuntimeError("el índice no está ajustado (fit)")
        kq = _bucket_keys(self.codes_from_projections(Yq))  # (nq, L)
        lo = np.empty(kq.shape, dtype=np.int64)
        hi = np.empty(kq.shape, dtype=np.int64)
        for t in range(self.L):
            lo[:, t] = np.searchsorted(self._keys_sorted[t], kq[:, t], side="left")
            hi[:, t] = np.searchsorted(self._keys_sorted[t], kq[:, t], side="right")
        out = []
        for i in range(kq.shape[0]):
            segs = [self._order[t, lo[i, t] : hi[i, t]] for t in range(self.L) if hi[i, t] > lo[i, t]]
            out.append(np.concatenate(segs).astype(np.int64) if segs else np.zeros(0, np.int64))
        return out

    def _query(self, Yq: np.ndarray, Q: np.ndarray | None, k: int,
               *, rerank_Yq: np.ndarray | None = None) -> QueryResult:
        if self.rerank == "proj":
            assert self._Y is not None
            if rerank_Yq is None:
                if self._Y.shape[1] != self.n_funcs:
                    raise ValueError("debe proporcionar rerank_query_projections con todas las columnas")
                rerank_Yq = Yq
            else:
                rerank_Yq = np.atleast_2d(np.asarray(rerank_Yq, dtype=np.float64))
            if (rerank_Yq.shape != (Yq.shape[0], self._Y.shape[1]) or
                    not np.isfinite(rerank_Yq).all()):
                raise ValueError("rerank_query_projections: forma o valores inválidos")
        nq = Yq.shape[0]
        idx = np.full((nq, k), -1, dtype=np.int64)
        scores = np.full((nq, k), np.nan, dtype=np.float32)
        n_cand = np.zeros(nq, dtype=np.int64)
        for i, raw in enumerate(self.candidates_from_projections(Yq)):
            if raw.size == 0:
                continue
            cand, counts = np.unique(raw, return_counts=True)  # ordenados: empates → índice menor
            n_cand[i] = cand.size
            if self.rerank == "d":
                assert self._X is not None and Q is not None
                dist = _lp_dist(Q[i], self._X[cand], self.p)
            elif self.rerank == "proj":
                assert self._Y is not None
                dY = self._Y[cand].astype(np.float64) - rerank_Yq[i]
                if self.p == 2:
                    dist = np.sqrt(np.einsum("ij,ij->i", dY, dY) / self._Y.shape[1])
                else:
                    dist = np.median(np.abs(dY), axis=1)
            else:
                dist = 1.0 - counts / self.L
            loc, val = topk(dist[None, :], k, largest=False)
            ok = loc[0] >= 0
            idx[i, ok] = cand[loc[0, ok]]
            scores[i, ok] = val[0, ok]
        stats: dict[str, Any] = {
            "n_candidates": n_cand,
            "frac_below_k": float(np.mean(n_cand < k)) if nq else 0.0,
            "frac_empty": float(np.mean(n_cand == 0)) if nq else 0.0,
            "build_time_s": self.build_time_s,
            "index_bytes": self.index_bytes(),
            "rerank_space": self.rerank,
        }
        return idx, scores, stats

    # ---------------- diagnóstico ----------------

    def index_bytes(self) -> int:
        """Bytes de los arreglos del índice (tablas, A, β y X o Y si se guardan)."""
        arrs = [self._keys_sorted, self._order, self._A, self.beta, self._X, self._Y]
        return int(sum(a.nbytes for a in arrs if a is not None))

    def bucket_sizes(self, table: int = 0) -> np.ndarray:
        """Tamaños de los buckets no vacíos de una tabla (para reportar su distribución)."""
        if self._keys_sorted is None:
            raise RuntimeError("el índice no está ajustado (fit)")
        _, counts = np.unique(self._keys_sorted[table], return_counts=True)
        return counts.astype(np.int64)
