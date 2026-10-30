"""MinHash y LSH por bandas para similitud de Jaccard entre conjuntos.

Pensado para perfiles que son **conjuntos** (p. ej. los genes mutados de cada
muestra, que llegan como eventos del stream de mutaciones) y, con la variante
ponderada, para vectores no negativos (similitud de Jaccard generalizada).

- ``token_hash``: hash estable de 64 bits (blake2b), no ``hash()`` de Python
  (que cambia entre procesos por ``PYTHONHASHSEED``).
- ``MinHash``: firma de K mínimos con la familia universal
  h_i(x) = (a_i·x + b_i) mod p, p = 2^61 − 1 (Carter y Wegman, 1979), en NumPy
  vectorizado sin desborde (multiplicación modular por partes de 32 bits).
  P[min h_i(A) = min h_i(B)] ≈ J(A, B) (Broder, 1997), así que la fracción de
  posiciones iguales es un estimador de J con varianza J(1 − J)/K.
- ``StreamingMinHash``: la misma firma, actualizada evento a evento por
  muestra (mínimo acumulado: independiente del orden y de los repetidos).
- ``MinHashLSH``: bandas de r filas sobre la firma; P[candidato] =
  1 − (1 − s^r)^b (curva S) con umbral ≈ (1/b)^(1/r) (Leskovec, Rajaraman y
  Ullman, *Mining of Massive Datasets*, cap. 3, §3.4).
- ``WeightedMinHash`` / ``StreamingWeightedMinHash``: ICWS (Ioffe, 2010) para
  la similitud de Jaccard ponderada Σ min(x, y) / Σ max(x, y); la versión en
  streaming recibe filas de rasgo (un gen con su valor en todas las muestras).

Convención para conjuntos vacíos (explícita y la misma en todo el módulo):
J(∅, ∅) = 1 (como ``scipy.spatial.distance.jaccard``, que da distancia 0) y
J(∅, A) = 0 si A ≠ ∅. La firma de ∅ es el centinela ``EMPTY`` (2^64 − 1) en
todas las posiciones; ningún hash real (< p) lo alcanza, de modo que el
estimador reproduce la convención exactamente.

Referencias:

- A. Z. Broder (1997), "On the resemblance and containment of documents".
- J. Leskovec, A. Rajaraman, J. Ullman, *Mining of Massive Datasets*, cap. 3
  (§3.3 MinHash, §3.4 LSH por bandas). http://www.mmds.org/
- S. Ioffe (2010), "Improved Consistent Sampling, Weighted Minhash and L1
  Sketching", ICDM. Fórmulas de ICWS cotejadas con ``datasketch``
  (``WeightedMinHashGenerator.minhash``): https://github.com/ekzhu/datasketch
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..base import QueryResult, feature_seed, topk

MERSENNE_P = (1 << 61) - 1
_P = np.uint64(MERSENNE_P)
_M32 = np.uint64(0xFFFFFFFF)
_M29 = np.uint64((1 << 29) - 1)
EMPTY = np.uint64(np.iinfo(np.uint64).max)  # firma de ∅: mayor que cualquier hash real
_CHUNK = 4096  # tokens por bloque al hashear (memoria O(_CHUNK · K))


# --------------------------------------------------------------------------
# Hash de tokens y aritmética módulo 2^61 − 1
# --------------------------------------------------------------------------


def token_hash(token: Any) -> int:
    """Hash estable de 64 bits (blake2b) de un token; ``bytes`` se usan tal cual, lo demás vía ``str``."""
    data = token if isinstance(token, bytes) else str(token).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def token_hashes(tokens: Iterable[Any]) -> np.ndarray:
    """``token_hash`` de cada token, como uint64 (sin deduplicar)."""
    return np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64)


def _reduce(v: np.ndarray) -> np.ndarray:
    """v mod p para v < 2^64 usando 2^61 ≡ 1 (mod p); resultado en [0, p)."""
    s = (v & _P) + (v >> np.uint64(61))  # < 2^61 + 8
    return np.where(s >= _P, s - _P, s)


def mulmod_p(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(a · x) mod (2^61 − 1) elemento a elemento (con broadcasting), sin desborde.

    Requiere a, x < p. Con a = a1·2^32 + a0 y x = x1·2^32 + x0:
    a·x = a1x1·2^64 + (a1x0 + a0x1)·2^32 + a0x0, donde todos los productos
    caben en uint64 (a1, x1 < 2^29). Como 2^61 ≡ 1: 2^64 ≡ 8 y, partiendo
    mid = mh·2^29 + ml, mid·2^32 ≡ mh + ml·2^32.
    """
    a = np.asarray(a, dtype=np.uint64)
    return _mulmod_split(a & _M32, a >> np.uint64(32), np.asarray(x, dtype=np.uint64))


def _mulmod_split(a0: np.ndarray, a1: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Núcleo de ``mulmod_p`` con a ya partido en (a0, a1) (se reutiliza entre llamadas)."""
    x0, x1 = x & _M32, x >> np.uint64(32)
    lo = a0 * x0  # < 2^64
    mid = a1 * x0 + a0 * x1  # < 2^62
    hi = a1 * x1  # < 2^58
    s = (
        (hi << np.uint64(3))
        + (mid >> np.uint64(29))
        + ((mid & _M29) << np.uint64(32))
        + (lo & _P)
        + (lo >> np.uint64(61))
    )  # < 3·2^61 + 2^34 < 2^63
    return _reduce(s)


# --------------------------------------------------------------------------
# Jaccard exacto
# --------------------------------------------------------------------------


def exact_jaccard(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """|A ∩ B| / |A ∪ B|; J(∅, ∅) = 1 por convención (ver docstring del módulo)."""
    A, B = set(a), set(b)
    union = len(A | B)
    return 1.0 if union == 0 else len(A & B) / union


def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray | float:
    """Fracción de posiciones iguales entre firmas MinHash (último eje); admite broadcasting.

    Es insesgado para J con varianza J(1 − J)/K. Para firmas ponderadas (ICWS)
    se usa ``weighted_jaccard_estimate``.
    """
    s1, s2 = np.asarray(sig1), np.asarray(sig2)
    if s1.shape[-1:] != s2.shape[-1:]:
        raise ValueError(f"firmas de largo distinto: {s1.shape} vs {s2.shape}")
    out = (s1 == s2).mean(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------
# MinHash para conjuntos
# --------------------------------------------------------------------------


class MinHash:
    """Firmas MinHash de K = ``num_perm`` permutaciones simuladas.

    h_i(x) = (a_i·x + b_i) mod p con a_i ∈ [1, p), b_i ∈ [0, p) sorteados con
    ``np.random.default_rng(seed)`` y x = token_hash(token) mod p. La firma de
    un conjunto es sig_i = min_{x∈A} h_i(x) (uint64[K]); la de ∅ es ``EMPTY``.
    """

    def __init__(self, num_perm: int = 128, seed: int = 0):
        if num_perm < 1:
            raise ValueError("num_perm debe ser ≥ 1")
        self.num_perm = int(num_perm)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE_P, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE_P, size=num_perm, dtype=np.uint64)
        self._a0, self._a1 = self.a & _M32, self.a >> np.uint64(32)

    def hash_values(self, x: np.ndarray) -> np.ndarray:
        """Matriz uint64 (len(x), K) con h_i(x_j); x son hashes de 64 bits."""
        xr = _reduce(np.asarray(x, dtype=np.uint64).ravel())[:, None]
        s = _mulmod_split(self._a0[None, :], self._a1[None, :], xr) + self.b[None, :]  # < 2p < 2^63
        return np.where(s >= _P, s - _P, s)

    def signature_from_hashes(self, x: np.ndarray) -> np.ndarray:
        """Firma de un conjunto dado por sus hashes de 64 bits (los repetidos no importan)."""
        x = np.asarray(x, dtype=np.uint64).ravel()
        sig = np.full(self.num_perm, EMPTY, dtype=np.uint64)
        for start in range(0, x.size, _CHUNK):
            np.minimum(sig, self.hash_values(x[start : start + _CHUNK]).min(axis=0), out=sig)
        return sig

    def signature(self, tokens: Iterable[Any]) -> np.ndarray:
        """Firma uint64[K] de un conjunto de tokens (str, bytes u otros vía ``str``)."""
        return self.signature_from_hashes(token_hashes(set(tokens)))

    def signatures(self, sets: Iterable[Iterable[Any]]) -> np.ndarray:
        """Firmas apiladas (n, K) de una colección de conjuntos."""
        rows = [self.signature(s) for s in sets]
        return np.stack(rows) if rows else np.zeros((0, self.num_perm), dtype=np.uint64)

    @staticmethod
    def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray | float:
        return jaccard_estimate(sig1, sig2)


class StreamingMinHash:
    """Firmas MinHash por muestra construidas evento a evento.

    ``update(sample_key, token)`` hace sig[key] = min(sig[key], h(token)). El
    mínimo es conmutativo, asociativo e idempotente, así que el resultado no
    depende del orden de llegada ni de los repetidos, y coincide con
    ``MinHash(num_perm, seed).signature(conjunto completo)``.

    Memoria: O(n_muestras · K) para las firmas más un caché opcional de los
    hashes de cada token distinto (O(|vocabulario| · K); p. ej. 20.000 genes ·
    128 · 8 B ≈ 20 MB). Uso típico con el stream de mutaciones::

        smh = StreamingMinHash(128, seed=0)
        for rec in StreamService(raiz, data_types=["mutations"]):
            if rec.kind == "event":
                smh.update_record(rec, "Hugo_Symbol")
        keys, S = smh.signatures()
    """

    def __init__(self, num_perm: int = 128, seed: int = 0, *, cache_tokens: bool = True):
        self.minhash = MinHash(num_perm, seed)
        self.num_perm = self.minhash.num_perm
        self.cache_tokens = cache_tokens
        self._cache: dict[Any, np.ndarray] = {}
        self._index: dict[Any, int] = {}
        self._keys: list[Any] = []
        self._sig = np.full((16, self.num_perm), EMPTY, dtype=np.uint64)
        self.n_events = 0

    def _row(self, key: Any) -> int:
        i = self._index.get(key)
        if i is None:
            i = len(self._keys)
            if i == self._sig.shape[0]:
                grown = np.full((2 * i, self.num_perm), EMPTY, dtype=np.uint64)
                grown[:i] = self._sig
                self._sig = grown
            self._index[key] = i
            self._keys.append(key)
        return i

    def add_key(self, sample_key: Any) -> None:
        """Registra una muestra sin tokens (su firma queda en ``EMPTY``: conjunto vacío)."""
        self._row(sample_key)

    def _token_values(self, token: Any) -> np.ndarray:
        h = self._cache.get(token)
        if h is None:
            h = self.minhash.hash_values(np.array([token_hash(token)], dtype=np.uint64))[0]
            if self.cache_tokens:
                self._cache[token] = h
        return h

    def update(self, sample_key: Any, token: Any) -> None:
        i = self._row(sample_key)
        np.minimum(self._sig[i], self._token_values(token), out=self._sig[i])
        self.n_events += 1

    def update_many(self, sample_keys: Sequence[Any], tokens: Sequence[Any]) -> None:
        """Versión por lotes: hashea cada token distinto una vez y reduce con ``np.minimum.at``."""
        if len(sample_keys) != len(tokens):
            raise ValueError("sample_keys y tokens deben tener el mismo largo")
        if not len(tokens):
            return
        rows = np.fromiter((self._row(k) for k in sample_keys), dtype=np.int64, count=len(sample_keys))
        uniq: dict[Any, int] = {}
        tok_idx = np.fromiter((uniq.setdefault(t, len(uniq)) for t in tokens), dtype=np.int64, count=len(tokens))
        H = self.minhash.hash_values(token_hashes(uniq.keys()))
        np.minimum.at(self._sig, rows, H[tok_idx])
        self.n_events += len(tokens)

    def update_record(self, rec: Any, token_field: str = "Hugo_Symbol") -> bool:
        """Toma un ``Record`` de tipo ``event`` del stream; ignora tokens vacíos o NA."""
        if getattr(rec, "kind", None) != "event" or rec.key is None:
            return False
        token = rec.fields.get(token_field)
        if token is None or str(token).strip() in ("", "NA", "NaN", "nan", ".", "Unknown"):
            self.add_key(rec.key)
            return False
        self.update(rec.key, str(token).strip())
        return True

    def signatures(self) -> tuple[list[Any], np.ndarray]:
        """(claves en orden de primera aparición, firmas uint64 (n, K))."""
        return list(self._keys), self._sig[: len(self._keys)].copy()


# --------------------------------------------------------------------------
# LSH por bandas sobre firmas MinHash
# --------------------------------------------------------------------------


def s_curve(s: np.ndarray | float, bands: int, rows: int) -> np.ndarray | float:
    """P[ser candidato] = 1 − (1 − s^rows)^bands para un par con Jaccard s."""
    return 1.0 - (1.0 - np.asarray(s, dtype=np.float64) ** rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    """Umbral aproximado de la curva S: (1/bands)^(1/rows) (MMDS §3.4.2)."""
    return (1.0 / bands) ** (1.0 / rows)


class MinHashLSH:
    """LSH por bandas sobre firmas MinHash, con interfaz tipo ``KNNIndex``.

    La firma (K ≥ bands·rows columnas) se parte en ``bands`` bandas de ``rows``
    posiciones; dos elementos son candidatos si coinciden en todas las
    posiciones de al menos una banda. Los candidatos se ordenan por Jaccard:

    - ``rerank="exact"``: Jaccard exacto (requiere haber dado conjuntos en
      ``fit`` y en ``query``);
    - ``rerank="estimate"``: Jaccard estimado con la firma completa (K columnas);
    - ``rerank="auto"`` (por defecto): exacto si hay conjuntos, si no estimado.

    ``fit`` y ``query`` aceptan una matriz uint64 (n, K) de firmas o una
    secuencia de conjuntos de tokens (se firman con ``MinHash(bands·rows, seed)``).
    ``score_kind = "similarity"``; si hay menos de k candidatos se rellena con
    -1 / NaN. Si no hay candidatos el resultado queda vacío (no hay respaldo).
    """

    name = "MinHashLSH"
    score_kind = "similarity"

    def __init__(self, bands: int = 16, rows: int = 8, seed: int = 0, *, rerank: str = "auto"):
        if bands < 1 or rows < 1:
            raise ValueError("bands y rows deben ser ≥ 1")
        if rerank not in ("auto", "exact", "estimate"):
            raise ValueError("rerank debe ser 'auto', 'exact' o 'estimate'")
        self.bands, self.rows, self.seed, self.rerank = int(bands), int(rows), int(seed), rerank
        self.num_perm = self.bands * self.rows
        self.minhash = MinHash(self.num_perm, seed)
        self.sig_: np.ndarray | None = None
        self.sets_: list[frozenset] | None = None
        self.tables_: list[dict[bytes, np.ndarray]] = []
        self.build_time_s = 0.0

    def s_curve(self, s: np.ndarray | float) -> np.ndarray | float:
        return s_curve(s, self.bands, self.rows)

    @property
    def threshold(self) -> float:
        return lsh_threshold(self.bands, self.rows)

    def _prepare(self, X: Any) -> tuple[np.ndarray, list[frozenset] | None]:
        if isinstance(X, (str, bytes)):
            raise TypeError("se esperaba una colección de conjuntos o una matriz de firmas, no un str")
        if isinstance(X, np.ndarray):
            # Firmas guardadas como int64 (u otro entero) se reinterpretan como uint64;
            # antes se trataban en silencio como "conjuntos de enteros" (bug).
            if X.dtype.kind not in "ui":
                raise TypeError(f"matriz de firmas con dtype {X.dtype}: se esperaba entero (uint64)")
            S = X.astype(np.int64, copy=False).view(np.uint64) if X.dtype.kind == "i" else X.astype(np.uint64, copy=False)
            S = S[None, :] if S.ndim == 1 else S
            if S.ndim != 2 or S.shape[1] < self.num_perm:
                raise ValueError(f"se esperaban firmas (n, ≥{self.num_perm}), llegó {S.shape}")
            return np.ascontiguousarray(S), None
        sets = [frozenset(s) for s in X]
        return self.minhash.signatures(sets), sets

    def _band_keys(self, S: np.ndarray, t: int) -> list[bytes]:
        block = np.ascontiguousarray(S[:, t * self.rows : (t + 1) * self.rows])
        return [row.tobytes() for row in block]

    def fit(self, X: Any) -> "MinHashLSH":
        t0 = time.perf_counter()
        self.sig_, self.sets_ = self._prepare(X)
        self.tables_ = []
        for t in range(self.bands):
            block = np.ascontiguousarray(self.sig_[:, t * self.rows : (t + 1) * self.rows])
            keys = block.view(np.dtype((np.void, block.dtype.itemsize * self.rows))).ravel()
            uniq, inv = np.unique(keys, return_inverse=True)
            order = np.argsort(inv, kind="stable")
            bounds = np.searchsorted(inv[order], np.arange(uniq.size + 1))
            self.tables_.append(
                {uniq[g].tobytes(): order[bounds[g] : bounds[g + 1]].astype(np.int64) for g in range(uniq.size)}
            )
        self.build_time_s = time.perf_counter() - t0
        return self

    def candidates(self, qsig: np.ndarray) -> np.ndarray:
        """Índices (int64, ordenados) que comparten al menos una banda con la firma ``qsig``."""
        if self.sig_ is None:
            raise RuntimeError("llama a fit antes de consultar")
        q = np.ascontiguousarray(np.asarray(qsig, dtype=np.uint64)[: self.num_perm])
        hits = [
            self.tables_[t].get(q[t * self.rows : (t + 1) * self.rows].tobytes())
            for t in range(self.bands)
        ]
        hits = [h for h in hits if h is not None]
        return np.unique(np.concatenate(hits)) if hits else np.zeros(0, dtype=np.int64)

    def index_bytes(self) -> int:
        """Tamaño aproximado: firmas + índices de los buckets + claves."""
        if self.sig_ is None:
            return 0
        n_keys = sum(len(tb) for tb in self.tables_)
        return int(self.sig_.nbytes + self.bands * self.sig_.shape[0] * 8 + n_keys * 8 * self.rows)

    def query(self, Q: Any, k: int) -> QueryResult:
        if self.sig_ is None:
            raise RuntimeError("llama a fit antes de consultar")
        if int(k) < 0:
            raise ValueError("k debe ser ≥ 0")
        k = int(k)
        QS, qsets = self._prepare(Q)
        if QS.shape[1] != self.sig_.shape[1]:
            raise ValueError("las firmas de consulta y de la base tienen distinto largo")
        use_exact = self.rerank != "estimate" and qsets is not None and self.sets_ is not None
        if self.rerank == "exact" and not use_exact:
            raise ValueError("rerank='exact' requiere conjuntos en fit y en query")
        nq = QS.shape[0]
        idx = np.full((nq, k), -1, dtype=np.int64)
        scores = np.full((nq, k), np.nan, dtype=np.float32)
        n_cand = np.zeros(nq, dtype=np.int64)
        for i in range(nq):
            cand = self.candidates(QS[i])
            n_cand[i] = cand.size
            if cand.size == 0:
                continue
            if use_exact:
                s = np.array([exact_jaccard(qsets[i], self.sets_[c]) for c in cand], dtype=np.float64)
            else:
                s = (self.sig_[cand] == QS[i]).mean(axis=1)
            loc, val = topk(s[None, :], k)
            ok = loc[0] >= 0
            idx[i, ok] = cand[loc[0, ok]]
            scores[i, ok] = val[0, ok]
        stats = {
            "n_candidates": n_cand,
            "build_time_s": self.build_time_s,
            "index_bytes": self.index_bytes(),
            "rerank_space": "jaccard_exact" if use_exact else "jaccard_estimate",
        }
        return idx, scores, stats


# --------------------------------------------------------------------------
# MinHash ponderado: ICWS (Ioffe, 2010)
# --------------------------------------------------------------------------


def weighted_jaccard(x: Mapping[Any, float] | np.ndarray, y: Mapping[Any, float] | np.ndarray) -> float:
    """Jaccard generalizado (Ruzicka) Σ min / Σ max para pesos ≥ 0; 1 si ambos son nulos.

    Acepta diccionarios token → peso o vectores densos del mismo largo; los
    pesos negativos o NaN se tratan como 0 (fuera del soporte).
    """
    if isinstance(x, Mapping) or isinstance(y, Mapping):
        keys = set(x) | set(y)
        xv = np.array([x.get(t, 0.0) for t in keys], dtype=np.float64)
        yv = np.array([y.get(t, 0.0) for t in keys], dtype=np.float64)
    else:
        xv, yv = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xv = np.where(np.isnan(xv) | (xv < 0), 0.0, xv)
    yv = np.where(np.isnan(yv) | (yv < 0), 0.0, yv)
    den = np.maximum(xv, yv).sum()
    return 1.0 if den == 0 else float(np.minimum(xv, yv).sum() / den)


def _icws_params(feature_id: Any, num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, ln c, β) de un rasgo: r, c ~ Gamma(2, 1), β ~ U(0, 1); dependen solo de (semilla, ID)."""
    rng = np.random.default_rng(feature_seed(str(feature_id), seed))
    r = rng.gamma(2.0, 1.0, num_perm)
    ln_c = np.log(rng.gamma(2.0, 1.0, num_perm))
    beta = rng.uniform(0.0, 1.0, num_perm)
    return r, ln_c, beta


def _icws_step(ln_w: np.ndarray, r: np.ndarray, ln_c: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ln a y t de ICWS para pesos con logaritmo ``ln_w`` (m,) → matrices (m, K).

    t = ⌊ln w / r + β⌋, ln y = r (t − β), ln a = ln c − ln y − r.
    """
    t = np.floor(ln_w[:, None] / r[None, :] + beta[None, :])
    ln_a = ln_c[None, :] - r[None, :] * (t - beta[None, :]) - r[None, :]
    return ln_a, t


class WeightedMinHash:
    """ICWS: firma de K muestras consistentes (rasgo k*, entero t*) por vector de pesos ≥ 0.

    P[firma_i(x) = firma_i(y)] = Σ min(x, y) / Σ max(x, y) (Ioffe, 2010). Los
    parámetros (r, c, β) de cada rasgo se generan por ID de rasgo, como Ω en
    ``base``, así que la versión en memoria y la de streaming coinciden.

    La firma es uint64 (K, 2): columna 0 = ``token_hash`` del rasgo elegido,
    columna 1 = t* (int64 reinterpretado). La firma del vector nulo es
    ``EMPTY`` en ambas columnas. Compara con ``weighted_jaccard_estimate``.
    """

    def __init__(self, num_perm: int = 128, seed: int = 0):
        self.num_perm, self.seed = int(num_perm), int(seed)
        self._params: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def params(self, feature_id: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = str(feature_id)
        p = self._params.get(key)
        if p is None:
            p = self._params[key] = _icws_params(key, self.num_perm, self.seed)
        return p

    def signature(self, weights: Mapping[Any, float]) -> np.ndarray:
        """Firma de un vector disperso dado como diccionario rasgo → peso (≤ 0 o NaN se ignoran)."""
        items = [(str(f), float(w)) for f, w in weights.items() if np.isfinite(w) and w > 0]
        sig = np.full((self.num_perm, 2), EMPTY, dtype=np.uint64)
        if not items:
            return sig
        best = np.full(self.num_perm, np.inf)
        for f, w in items:
            r, ln_c, beta = self.params(f)
            ln_a, t = _icws_step(np.array([np.log(w)]), r, ln_c, beta)
            better = ln_a[0] < best
            best[better] = ln_a[0, better]
            sig[better, 0] = np.uint64(token_hash(f))
            sig[better, 1] = t[0, better].astype(np.int64).view(np.uint64)
        return sig


def weighted_jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray | float:
    """Fracción de muestras ICWS iguales (rasgo y t); firmas de forma (..., K, 2)."""
    eq = (np.asarray(sig1) == np.asarray(sig2)).all(axis=-1)
    out = eq.mean(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


class StreamingWeightedMinHash:
    """ICWS en streaming sobre filas de rasgo (un rasgo con su peso en las n muestras).

    ``update(feature_id, x)`` calcula ln a para las muestras con x > 0 y
    actualiza el mínimo y el (rasgo, t) asociado: el resultado no depende del
    orden de los rasgos y coincide con ``WeightedMinHash.signature`` de cada
    muestra. Los pesos deben ser ≥ 0 (p. ej. expresión log2(TPM + 1)); NaN y
    valores ≤ 0 quedan fuera del soporte. Costo: O(n · K) por rasgo.
    """

    def __init__(self, n_samples: int, num_perm: int = 128, seed: int = 0):
        self.n_samples, self.num_perm, self.seed = int(n_samples), int(num_perm), int(seed)
        self.best = np.full((n_samples, num_perm), np.inf)
        self.sig = np.full((n_samples, num_perm, 2), EMPTY, dtype=np.uint64)
        self.n_features = 0

    def update(self, feature_id: Any, x: np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.n_samples,):
            raise ValueError(f"{feature_id}: largo {v.shape}, se esperaba ({self.n_samples},)")
        self.n_features += 1
        rows = np.flatnonzero(np.isfinite(v) & (v > 0))
        if rows.size == 0:
            return
        r, ln_c, beta = _icws_params(str(feature_id), self.num_perm, self.seed)
        ln_a, t = _icws_step(np.log(v[rows]), r, ln_c, beta)
        cur = self.best[rows]
        better = ln_a < cur
        self.best[rows] = np.where(better, ln_a, cur)
        sub = self.sig[rows]
        sub[..., 0] = np.where(better, np.uint64(token_hash(str(feature_id))), sub[..., 0])
        sub[..., 1] = np.where(better, t.astype(np.int64).view(np.uint64), sub[..., 1])
        self.sig[rows] = sub

    def signatures(self) -> np.ndarray:
        """Firmas uint64 (n, K, 2)."""
        return self.sig.copy()
