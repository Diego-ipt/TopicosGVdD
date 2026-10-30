"""Perfilado de los datos para decidir qué medidas de similitud son aplicables.

Todo se calcula consumiendo ``StreamService`` (nunca se carga un archivo
completo). Por cada tipo de dato se producen dos niveles de información:

1. **Estadísticos exactos por rasgo** sobre todas las filas del stream (o las
   primeras ``max_rows``): media, varianza, mínimo, máximo, asimetría y las
   fracciones de NA, ceros, negativos, enteros y valores en [0, 1], en escala
   cruda y transformada. Son d números por estadístico; se resumen con
   cuantiles e histogramas y se guardan como vectores en un ``.npz``.
2. **Sketch del perfil**: una muestra de S muestras (columnas) estratificada
   por tipo de cáncer y un *reservoir sampling* (Algorithm R) de hasta F rasgos.
   La matriz S × F cabe en memoria y sobre ella se comparan las medidas
   (coseno, L2, L1, 1 − Pearson, 1 − Spearman, Jaccard y Hamming sobre una
   binarización) con cuatro variantes de preprocesamiento. Es un sketch: las
   conclusiones valen para un subconjunto uniforme de rasgos y una muestra
   balanceada de pacientes, no para la matriz completa.

Diagnósticos calculados sobre cada matriz de distancias:

- **Concentración**: contraste relativo (D_max − D_min)/D_min por consulta
  (Beyer et al., 1999, "When is nearest neighbor meaningful?"; Aggarwal et al.,
  2001) y coeficiente de variación de las distancias.
- **Hubness**: asimetría de la distribución de k-ocurrencias N_k (Radovanović,
  Nanopoulos e Ivanović, 2010, JMLR 11:2487-2531).
- **Pureza k-NN** leave-one-out y exactitud del voto por mayoría.
- **Acuerdo entre medidas**: Jaccard medio de los conjuntos de 10 vecinos.
- **Dimensión intrínseca**: componentes PCA para 50/80/90/95 % de la varianza y
  estimador TwoNN (Facco et al., 2017, Sci. Rep. 7:12140), d̂ = N / Σ log(r₂/r₁).
- **Normas por muestra**: si la magnitud lleva información del tipo (η² de la
  norma frente a la etiqueta y pureza k-NN usando solo la norma).

Mutaciones (MC3): se agrupa el stream de eventos por muestra (el MAF viene
ordenado por ``Tumor_Sample_Barcode``; si una clave reaparece se cuenta y se
marca), se miden los tamaños de los conjuntos de genes mutados de todos los
pacientes y se comparan Jaccard, coseno binario y Hamming sobre S pacientes.
"No silenciosa" sigue la lista ``vc_nonSyn`` por defecto de maftools
(Mayakonda et al., 2018, Genome Res. 28:1747).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps
from scipy.spatial.distance import pdist, squareform

from tcgaknn.algoritmos.base import topk
from tcgaknn.stream import StreamService

MATRIX_TYPES = ("expression", "methylation_450k", "methylation_27k_450k", "mirna", "paradigm", "rppa")
ALL_TYPES = MATRIX_TYPES + ("mutations",)

TUMOR_CODES = ("01", "03", "06")
#: Código de tipo de muestra preferido cuando un paciente tiene varias (plan §2.4).
CANONICAL_CODE = {"LAML": "03", "SKCM": "06"}

#: Clasificaciones no silenciosas (``vc_nonSyn`` por defecto de maftools).
NON_SILENT = frozenset({
    "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site", "Translation_Start_Site",
    "Nonsense_Mutation", "Nonstop_Mutation", "In_Frame_Del", "In_Frame_Ins", "Missense_Mutation",
})

STAT_NAMES = (
    "n_validos", "media", "varianza", "min", "max", "frac_na", "frac_cero", "frac_neg",
    "frac_entero", "frac_01", "asimetria", "media_t", "varianza_t", "asimetria_t",
)

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

MATRIX_METRICS = ("coseno", "L2", "L1", "pearson", "spearman", "jaccard", "hamming")
SET_METRICS = ("jaccard", "coseno_binario", "hamming")


# --------------------------------------------------------------------------
# Transformaciones y binarizaciones por tipo de dato
# --------------------------------------------------------------------------


def log2p1(x: np.ndarray) -> np.ndarray:
    """log2(max(x, 0) + 1); conserva NaN (plan §2.5.4)."""
    return np.log2(np.maximum(x, 0) + 1)


def m_value(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """M-value de metilación, M = log2(β/(1 − β)) con β recortado a [eps, 1 − eps].

    Du et al. (2010), BMC Bioinformatics 11:587: M es aproximadamente
    homocedástico, β no (su varianza se comprime cerca de 0 y 1).
    """
    b = np.clip(x, eps, 1 - eps)
    return np.log2(b / (1 - b))


TRANSFORMS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray] | None]] = {
    "expression": ("log2(max(x,0)+1)", log2p1),
    "mirna": ("log2(max(x,0)+1)", log2p1),
    "methylation_450k": ("M-value log2(b/(1-b)), b en [0.001, 0.999]", m_value),
    "methylation_27k_450k": ("M-value log2(b/(1-b)), b en [0.001, 0.999]", m_value),
    "rppa": ("ninguna", None),
    "paradigm": ("ninguna", None),
}

#: Binarización "natural" (variantes sin z-score); con z-score se usa z > 1.
NATURAL_BINARIZATION: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "expression": ("x > 1 (gen expresado: log2(x+1) > 1)", lambda X: X > 1),
    "mirna": ("x > 1 (miRNA expresado: log2(x+1) > 1)", lambda X: X > 1),
    "methylation_450k": ("beta > 0,5 (sonda metilada; equivale a M > 0)", lambda X: X > 0.5),
    "methylation_27k_450k": ("beta > 0,5 (sonda metilada; equivale a M > 0)", lambda X: X > 0.5),
    "rppa": ("x > 0 (proteína sobre la mediana normalizada)", lambda X: X > 0),
    "paradigm": ("x > 0 (vía/entidad activada)", lambda X: X > 0),
}
Z_BINARIZATION = "z > 1 por rasgo (sobreexpresado respecto de la muestra S; la L2 por muestra no lo cambia)"


# --------------------------------------------------------------------------
# Barcodes, etiquetas y selección estratificada
# --------------------------------------------------------------------------


def patient_barcode(barcode: str) -> str:
    """Barcode de paciente: primeros 12 caracteres (``TCGA-XX-XXXX``)."""
    return barcode[:12]


def sample_type_code(barcode: str) -> str | None:
    """Caracteres 14–15 del barcode (tipo de muestra) o None si no los trae (p. ej. PARADIGM)."""
    if len(barcode) >= 15 and barcode[12] == "-" and barcode[13:15].isdigit():
        return barcode[13:15]
    return None


def load_labels(root: Path | str) -> dict[str, str]:
    """Lee ``clinical_cdr`` en streaming: barcode de paciente → tipo de cáncer (``type``)."""
    labels: dict[str, str] = {}
    for rec in StreamService(root, data_types=["clinical_cdr"]):
        if rec.kind != "table_row" or not rec.key:
            continue
        t = rec.fields.get("type")
        if t is not None and str(t).strip():
            labels[str(rec.key).strip()] = str(t).strip()
    return labels


def stratified_allocation(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Reparte n entre estratos en partes iguales, con "llenado de agua".

    Los estratos con menos elementos que su cuota se toman completos y el
    sobrante se reparte entre los demás. Determinista (orden alfabético).
    """
    alloc = {t: 0 for t in counts}
    remaining = int(n)
    active = sorted(t for t, c in counts.items() if c > 0)
    while remaining > 0 and active:
        share = max(remaining // len(active), 1)
        nxt = []
        for t in active:
            if remaining == 0:
                break
            take = min(share, counts[t] - alloc[t], remaining)
            alloc[t] += take
            remaining -= take
            if alloc[t] < counts[t]:
                nxt.append(t)
        active = nxt
    return alloc


def _code_priority(code: str | None, label: str) -> int:
    if code is None:
        return 0
    return 0 if code == CANONICAL_CODE.get(label, "01") else 1


def eligible_by_patient(
    barcodes: Sequence[str],
    labels: Mapping[str, str],
    codes: Iterable[str] = TUMOR_CODES,
) -> tuple[dict[str, int], dict[str, int]]:
    """Una columna por paciente con etiqueta y tipo de muestra tumoral.

    Devuelve (paciente → índice de columna, conteos del embudo). Sin código de
    muestra (PARADIGM usa barcodes de paciente) se acepta la columna. Entre
    varias columnas de un paciente se prefiere el código canónico (01; 03 en
    LAML; 06 en SKCM) y luego el barcode menor (plan §2.4).
    """
    codes = frozenset(codes)
    funnel = Counter()
    best: dict[str, tuple[int, str, int]] = {}
    for i, b in enumerate(barcodes):
        funnel["columnas"] += 1
        p = patient_barcode(b)
        lab = labels.get(p)
        if lab is None:
            funnel["sin_etiqueta"] += 1
            continue
        code = sample_type_code(b)
        if code is None:
            funnel["sin_codigo_de_muestra"] += 1
        elif code not in codes:
            funnel[f"excluida_codigo_{code}"] += 1
            continue
        key = (_code_priority(code, lab), b, i)
        if p not in best or key < best[p]:
            if p in best:
                funnel["duplicada_mismo_paciente"] += 1
            best[p] = key
        else:
            funnel["duplicada_mismo_paciente"] += 1
    funnel["pacientes_elegibles"] = len(best)
    return {p: v[2] for p, v in best.items()}, dict(funnel)


def stratified_pick(
    by_type: Mapping[str, Sequence[Any]], n: int, seed: int
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Orden aleatorio reproducible dentro de cada tipo y cuota por tipo.

    Devuelve (tipo → lista completa permutada, tipo → cuota). Los elegidos son
    los primeros ``cuota`` de cada lista permutada.
    """
    rng = np.random.default_rng(seed)
    alloc = stratified_allocation({t: len(v) for t, v in by_type.items()}, n)
    ordered = {}
    for t in sorted(by_type):
        items = list(by_type[t])
        perm = rng.permutation(len(items))
        ordered[t] = [items[j] for j in perm]
    return ordered, alloc


def select_samples(
    barcodes: Sequence[str],
    labels: Mapping[str, str],
    n: int,
    seed: int,
    codes: Iterable[str] = TUMOR_CODES,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Elige hasta n columnas estratificadas por tipo de cáncer (una por paciente).

    Devuelve (índices int64 ordenados, etiquetas en ese orden, información del embudo).
    """
    pat_to_col, funnel = eligible_by_patient(barcodes, labels, codes)
    by_type: dict[str, list[int]] = defaultdict(list)
    for p in sorted(pat_to_col):
        by_type[labels[p]].append(pat_to_col[p])
    ordered, alloc = stratified_pick(by_type, n, seed)
    chosen = sorted(c for t in ordered for c in ordered[t][: alloc[t]])
    idx = np.asarray(chosen, dtype=np.int64)
    y = [labels[patient_barcode(barcodes[i])] for i in idx]
    info = {
        "embudo": funnel,
        "elegibles_por_tipo": {t: len(v) for t, v in sorted(by_type.items())},
        "elegidas_por_tipo": dict(sorted(Counter(y).items())),
    }
    return idx, y, info


# --------------------------------------------------------------------------
# Reservoir sampling (Algorithm R)
# --------------------------------------------------------------------------


class ReservoirSampler:
    """Algorithm R (Vitter, 1985, ACM TOMS 11(1):37-57) con almacenamiento externo.

    ``offer()`` se llama una vez por elemento del stream y devuelve la casilla
    (0 … capacity−1) donde guardarlo, o −1 si se descarta. Tras t elementos,
    cada uno está en el reservorio con probabilidad min(1, capacity/t).
    """

    def __init__(self, capacity: int, seed: int | np.random.Generator):
        if capacity < 1:
            raise ValueError("capacity debe ser >= 1")
        self.capacity = int(capacity)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.n_seen = 0

    def offer(self) -> int:
        i = self.n_seen
        self.n_seen += 1
        if i < self.capacity:
            return i
        j = int(self.rng.integers(0, i + 1))
        return j if j < self.capacity else -1

    @property
    def size(self) -> int:
        return min(self.n_seen, self.capacity)


def reservoir_sample(items: Iterable[Any], capacity: int, seed: int) -> list[Any]:
    """Muestra uniforme sin reemplazo de un iterable de largo desconocido."""
    rs = ReservoirSampler(capacity, seed)
    store: list[Any] = []
    for it in items:
        s = rs.offer()
        if s < 0:
            continue
        if s == len(store):
            store.append(it)
        else:
            store[s] = it
    return store


# --------------------------------------------------------------------------
# Estadísticos por rasgo (exactos, en streaming)
# --------------------------------------------------------------------------


def _moments(v: np.ndarray) -> tuple[float, float, float]:
    mu = float(v.mean())
    d = v - mu
    var = float(np.dot(d, d) / v.size)
    skew = float(np.dot(d * d, d) / v.size / var**1.5) if var > 1e-24 else math.nan
    return mu, var, skew


def row_stats(x: np.ndarray, transform: Callable[[np.ndarray], np.ndarray] | None) -> tuple[float, ...]:
    """Estadísticos de un rasgo (valores de todas las muestras), en el orden de ``STAT_NAMES``.

    Las fracciones son sobre los valores no-NA, salvo ``frac_na`` (sobre el total).
    """
    v = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(v)
    n = int(np.count_nonzero(valid))
    if n == 0:
        nan = math.nan
        return (0, nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan)
    vv = v if n == v.size else v[valid]
    mu, var, skew = _moments(vv)
    if transform is not None:
        mu_t, var_t, skew_t = _moments(np.asarray(transform(vv), dtype=np.float64))
    else:
        mu_t, var_t, skew_t = mu, var, skew
    return (
        n, mu, var, float(vv.min()), float(vv.max()), 1 - n / v.size,
        np.count_nonzero(vv == 0) / n, np.count_nonzero(vv < 0) / n,
        np.count_nonzero(vv == np.rint(vv)) / n,
        np.count_nonzero((vv >= 0) & (vv <= 1)) / n,
        skew, mu_t, var_t, skew_t,
    )


class ColumnStats:
    """Mismos estadísticos que ``row_stats`` pero acumulados por columna (filas = muestras).

    Para archivos orientados por muestra (RPPA): cada registro trae todos los
    rasgos de una muestra, y los momentos se acumulan con sumas de potencias.
    """

    def __init__(self, d: int, transform: Callable[[np.ndarray], np.ndarray] | None):
        self.d = d
        self.transform = transform
        self.n_rows = 0
        z = lambda: np.zeros(d, dtype=np.float64)  # noqa: E731
        self.n = np.zeros(d, dtype=np.int64)
        self.s1, self.s2, self.s3 = z(), z(), z()
        self.t1, self.t2, self.t3 = z(), z(), z()
        self.mn = np.full(d, np.inf)
        self.mx = np.full(d, -np.inf)
        self.n_zero = np.zeros(d, dtype=np.int64)
        self.n_neg = np.zeros(d, dtype=np.int64)
        self.n_int = np.zeros(d, dtype=np.int64)
        self.n_01 = np.zeros(d, dtype=np.int64)

    def update(self, x: np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64)
        valid = ~np.isnan(v)
        w = np.where(valid, v, 0.0)
        t = w if self.transform is None else np.where(valid, self.transform(v), 0.0)
        self.n_rows += 1
        self.n += valid
        self.s1 += w
        self.s2 += w * w
        self.s3 += w * w * w
        self.t1 += t
        self.t2 += t * t
        self.t3 += t * t * t
        self.mn = np.where(valid, np.minimum(self.mn, w), self.mn)
        self.mx = np.where(valid, np.maximum(self.mx, w), self.mx)
        self.n_zero += valid & (w == 0)
        self.n_neg += valid & (w < 0)
        self.n_int += valid & (w == np.rint(w))
        self.n_01 += valid & (w >= 0) & (w <= 1)

    @staticmethod
    def _from_sums(n: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = s1 / n
            var = np.maximum(s2 / n - mu * mu, 0.0)
            m3 = s3 / n - 3 * mu * s2 / n + 2 * mu**3
            skew = np.where(var > 1e-24, m3 / var**1.5, np.nan)
        return mu, var, skew

    def finalize(self) -> dict[str, np.ndarray]:
        n = self.n.astype(np.float64)
        nn = np.where(n > 0, n, np.nan)
        mu, var, skew = self._from_sums(nn, self.s1, self.s2, self.s3)
        mu_t, var_t, skew_t = self._from_sums(nn, self.t1, self.t2, self.t3)
        rows = max(self.n_rows, 1)
        out = {
            "n_validos": n, "media": mu, "varianza": var,
            "min": np.where(n > 0, self.mn, np.nan), "max": np.where(n > 0, self.mx, np.nan),
            "frac_na": 1 - n / rows, "frac_cero": self.n_zero / nn, "frac_neg": self.n_neg / nn,
            "frac_entero": self.n_int / nn, "frac_01": self.n_01 / nn, "asimetria": skew,
            "media_t": mu_t, "varianza_t": var_t, "asimetria_t": skew_t,
        }
        return {k: np.asarray(out[k], dtype=np.float64) for k in STAT_NAMES}


class SampleAccumulator:
    """Sumas por muestra (columna) sobre todos los rasgos de un archivo feature-major."""

    def __init__(self, n: int, transform: Callable[[np.ndarray], np.ndarray] | None):
        self.transform = transform
        self.n_valid = np.zeros(n, dtype=np.int64)
        self.sumsq = np.zeros(n, dtype=np.float64)
        self.sumsq_t = np.zeros(n, dtype=np.float64)
        self.n_rows = 0

    def update(self, x: np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64)
        valid = ~np.isnan(v)
        w = np.where(valid, v, 0.0)
        self.n_rows += 1
        self.n_valid += valid
        self.sumsq += w * w
        if self.transform is not None:
            t = np.where(valid, self.transform(v), 0.0)
            self.sumsq_t += t * t
        else:
            self.sumsq_t += w * w


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def quantile_summary(a: np.ndarray) -> dict[str, float | None]:
    f = _finite(a)
    if f.size == 0:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    qs = np.quantile(f, QUANTILES)
    out = {f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, qs)}
    out["media"] = float(f.mean())
    return out


def histogram(a: np.ndarray, bins: int = 20, log10: bool = False) -> dict[str, list]:
    f = _finite(a)
    if log10:
        f = np.log10(f[f > 0])
    if f.size == 0:
        return {"bordes": [], "conteos": [], "log10": log10}
    cnt, edges = np.histogram(f, bins=bins)
    return {"bordes": edges.tolist(), "conteos": cnt.tolist(), "log10": log10}


def summarize_feature_stats(st: Mapping[str, np.ndarray], n_columns: int) -> dict[str, Any]:
    """Cuantiles, histogramas, fracciones globales y diagnósticos automáticos."""
    nval = st["n_validos"]
    tot_valid = float(np.nansum(nval))

    def wfrac(name: str) -> float | None:
        f = st[name]
        m = np.isfinite(f)
        return float(np.sum(f[m] * nval[m]) / tot_valid) if tot_valid else None

    var = _finite(st["varianza"])
    var_t = _finite(st["varianza_t"])
    top1 = None
    if var.size:
        s = np.sort(var)[::-1]
        top1 = float(s[: max(1, s.size // 100)].sum() / s.sum()) if s.sum() > 0 else None
    pos = var[var > 0]
    global_ = {
        "n_rasgos": int(len(nval)),
        "n_columnas": int(n_columns),
        "frac_na": float(1 - tot_valid / (len(nval) * n_columns)) if len(nval) and n_columns else None,
        "frac_cero": wfrac("frac_cero"),
        "frac_neg": wfrac("frac_neg"),
        "frac_entero": wfrac("frac_entero"),
        "frac_01": wfrac("frac_01"),
        "min": float(np.nanmin(st["min"])) if np.isfinite(st["min"]).any() else None,
        "max": float(np.nanmax(st["max"])) if np.isfinite(st["max"]).any() else None,
        "rasgos_con_na": int(np.sum(st["frac_na"] > 0)),
        "rasgos_varianza_cero": int(np.sum(st["varianza"] <= 1e-12)),
        "frac_varianza_top1pct_rasgos": top1,
        "cociente_var_q99_q01": float(np.quantile(pos, 0.99) / np.quantile(pos, 0.01)) if pos.size > 10 else None,
        "cociente_var_t_q99_q01": (float(np.quantile(var_t[var_t > 0], 0.99) / np.quantile(var_t[var_t > 0], 0.01))
                                   if np.sum(var_t > 0) > 10 else None),
        "asimetria_mediana": float(np.median(_finite(st["asimetria"]))) if _finite(st["asimetria"]).size else None,
        "asimetria_t_mediana": float(np.median(_finite(st["asimetria_t"]))) if _finite(st["asimetria_t"]).size else None,
    }
    return {
        "global": global_,
        "cuantiles": {k: quantile_summary(st[k]) for k in STAT_NAMES},
        "histogramas": {
            "log10_varianza": histogram(st["varianza"], log10=True),
            "log10_varianza_t": histogram(st["varianza_t"], log10=True),
            "media_t": histogram(st["media_t"]),
            "frac_cero": histogram(st["frac_cero"]),
            "asimetria": histogram(st["asimetria"]),
            "asimetria_t": histogram(st["asimetria_t"]),
        },
        "diagnostico": diagnose(global_),
    }


def diagnose(g: Mapping[str, Any]) -> list[str]:
    """Reglas simples que traducen los estadísticos globales en notas sobre las medidas."""
    notes = []
    fneg, f01, fint, fz = g.get("frac_neg"), g.get("frac_01"), g.get("frac_entero"), g.get("frac_cero")
    if fneg is not None and fneg == 0:
        notes.append("Todos los valores son >= 0: los vectores crudos viven en un ortante y el coseno crudo "
                     "queda comprimido (ángulos pequeños); centrar (z-score) antes de coseno/SimHash.")
    elif fneg is not None and fneg < 0.01:
        notes.append(f"Hay pocos negativos ({100 * fneg:.3f} %): artefacto de normalización; recortar a 0 "
                     "antes de log2(x+1).")
    elif fneg is not None:
        notes.append(f"Datos con signo ({100 * fneg:.1f} % negativos): ya centrados o en escala log-ratio; "
                     "coseno/correlación aplican sin transformar; Jaccard requiere elegir un umbral.")
    if f01 is not None and f01 > 0.999:
        notes.append("Valores en [0, 1] (proporciones beta): varianza heterocedástica; M-value o z-score "
                     "antes de L2/coseno; la binarización beta > 0,5 tiene sentido biológico.")
    if fint is not None and fint > 0.99:
        notes.append("Valores enteros: datos de conteo; usar log2(x+1) o una transformación estabilizadora.")
    if fz is not None and fz > 0.3:
        notes.append(f"{100 * fz:.0f} % de ceros: datos dispersos; L1/Jaccard sobre presencia/ausencia son candidatos.")
    r = g.get("cociente_var_q99_q01")
    if r is not None and r > 100:
        notes.append(f"Escalas muy heterogéneas entre rasgos (var q99/q01 = {r:.3g}): L2/L1/coseno crudos los "
                     "dominan unos pocos rasgos; z-score por rasgo recomendado.")
    t = g.get("frac_varianza_top1pct_rasgos")
    if t is not None and t > 0.2:
        notes.append(f"El 1 % de rasgos de mayor varianza concentra el {100 * t:.0f} % de la varianza total cruda.")
    sk, skt = g.get("asimetria_mediana"), g.get("asimetria_t_mediana")
    if sk is not None and skt is not None and abs(sk) > 1 and abs(skt) < abs(sk):
        notes.append(f"La transformación reduce la asimetría mediana por rasgo de {sk:.2f} a {skt:.2f}.")
    if (g.get("frac_na") or 0) > 0:
        notes.append(f"NA = {100 * g['frac_na']:.3f} % de las celdas ({g['rasgos_con_na']} rasgos con algún NA): "
                     "imputar (media del rasgo = 0 tras z-score) o descartar rasgos.")
    return notes


# --------------------------------------------------------------------------
# Distancias y diagnósticos sobre la matriz sketch S × F
# --------------------------------------------------------------------------


def impute_feature_mean(X: np.ndarray) -> np.ndarray:
    """Reemplaza NaN por la media de su rasgo (columna) en la muestra; descarta columnas sin datos."""
    A = np.asarray(X, dtype=np.float64)
    keep = ~np.all(np.isnan(A), axis=0)
    A = A[:, keep]
    if np.isnan(A).any():
        mu = np.nanmean(A, axis=0)
        r, c = np.nonzero(np.isnan(A))
        A = A.copy()
        A[r, c] = mu[c]
    return A


def zscore_columns(A: np.ndarray) -> np.ndarray:
    """z-score por rasgo en la muestra; descarta rasgos de varianza nula."""
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    keep = sd > 1e-12
    return (A[:, keep] - mu[keep]) / sd[keep]


def l2_rows(A: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, nrm, out=np.zeros_like(A), where=nrm > 1e-12)


def cosine_distance(A: np.ndarray) -> np.ndarray:
    U = l2_rows(np.asarray(A, dtype=np.float64))
    D = 1.0 - U @ U.T
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def euclidean_distance(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    sq = np.einsum("ij,ij->i", A, A)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def manhattan_distance(A: np.ndarray) -> np.ndarray:
    return squareform(pdist(np.asarray(A, dtype=np.float64), metric="cityblock"))


def pearson_distance(A: np.ndarray) -> np.ndarray:
    """1 − r de Pearson entre muestras (coseno de los perfiles centrados por muestra)."""
    A = np.asarray(A, dtype=np.float64)
    return cosine_distance(A - A.mean(axis=1, keepdims=True))


def spearman_distance(A: np.ndarray) -> np.ndarray:
    """1 − ρ de Spearman: Pearson sobre los rangos de cada muestra (empates promediados)."""
    return pearson_distance(sps.rankdata(A, axis=1))


def set_intersections(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Bf = np.asarray(B, dtype=np.float64)
    return Bf @ Bf.T, Bf.sum(axis=1)


def jaccard_distance(B: np.ndarray) -> np.ndarray:
    """1 − |A∩B|/|A∪B| sobre filas binarias; si la unión es vacía la distancia es 1."""
    inter, s = set_intersections(B)
    union = s[:, None] + s[None, :] - inter
    D = 1.0 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(D, 0.0)
    return D


def hamming_distance(B: np.ndarray) -> np.ndarray:
    """|A Δ B| = |A| + |B| − 2|A∩B| (número de bits distintos)."""
    inter, s = set_intersections(B)
    D = s[:, None] + s[None, :] - 2.0 * inter
    np.fill_diagonal(D, 0.0)
    return D


def binary_cosine_distance(B: np.ndarray) -> np.ndarray:
    """1 − |A∩B|/√(|A|·|B|) (coseno de Ochiai); 1 si algún conjunto es vacío."""
    return cosine_distance(np.asarray(B, dtype=np.float64))


def knn_from_distances(D: np.ndarray, k: int) -> np.ndarray:
    """k vecinos leave-one-out (sin la propia muestra), empates por índice menor (``base.topk``)."""
    D2 = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(D2, np.inf)
    idx, _ = topk(D2, min(k, D.shape[0] - 1), largest=False)
    return idx


def concentration(D: np.ndarray) -> dict[str, Any]:
    """Contraste relativo (D_max − D_min)/D_min por consulta y CV de las distancias."""
    n = D.shape[0]
    off = ~np.eye(n, dtype=bool)
    rows = D[off].reshape(n, n - 1)
    dmin, dmax = rows.min(axis=1), rows.max(axis=1)
    pos = dmin > 0
    rc = (dmax[pos] - dmin[pos]) / dmin[pos]
    allv = rows.ravel()
    mean = float(allv.mean())
    return {
        "contraste_relativo_mediana": float(np.median(rc)) if rc.size else None,
        "contraste_relativo_q10": float(np.quantile(rc, 0.1)) if rc.size else None,
        "contraste_relativo_q90": float(np.quantile(rc, 0.9)) if rc.size else None,
        "consultas_dmin_cero": int(np.sum(~pos)),
        "cv_distancias": float(allv.std() / mean) if mean > 0 else None,
        "dmin_sobre_dmedia_mediana": float(np.median(dmin / rows.mean(axis=1))) if mean > 0 else None,
        "cuantiles_distancia": {f"q{int(q * 100):02d}": float(v)
                                for q, v in zip((0.05, 0.25, 0.5, 0.75, 0.95), np.quantile(allv, (0.05, 0.25, 0.5, 0.75, 0.95)))},
    }


def hubness(knn_idx: np.ndarray, n: int) -> dict[str, Any]:
    """Asimetría de N_k (k = columnas de knn_idx); > 1 indica hubs marcados."""
    nk = np.bincount(knn_idx[knn_idx >= 0].ravel(), minlength=n)
    k = knn_idx.shape[1]
    return {
        "k": int(k),
        "asimetria_Nk": float(sps.skew(nk)) if nk.std() > 0 else 0.0,
        "max_Nk": int(nk.max()),
        "frac_antihubs": float(np.mean(nk == 0)),
        "frac_hubs_2k": float(np.mean(nk > 2 * k)),
    }


def majority_vote(neighbor_labels: np.ndarray) -> int:
    """Clase mayoritaria; empate → la clase del vecino más cercano entre las empatadas."""
    vals, first, counts = np.unique(neighbor_labels, return_index=True, return_counts=True)
    best = first[counts == counts.max()]
    return int(neighbor_labels[best.min()])


def knn_purity(knn_idx: np.ndarray, y: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Pureza (fracción de vecinos de la misma clase) y exactitud del voto para cada k."""
    same = y[knn_idx] == y[:, None]
    out = {}
    for k in ks:
        k = min(k, knn_idx.shape[1])
        out[f"pureza@{k}"] = float(same[:, :k].mean())
        pred = np.array([majority_vote(y[row[:k]]) for row in knn_idx])
        out[f"voto@{k}"] = float(np.mean(pred == y))
    return out


def knn_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard medio entre los conjuntos de vecinos de cada consulta."""
    vals = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra[ra >= 0].tolist()), set(rb[rb >= 0].tolist())
        u = len(sa | sb)
        vals.append(len(sa & sb) / u if u else 1.0)
    return float(np.mean(vals))


def class_separation(D: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    intra, inter = D[same & off], D[~same]
    if not intra.size or not inter.size:
        return {"intra_media": None, "inter_media": None, "inter_sobre_intra": None}
    mi, me = float(intra.mean()), float(inter.mean())
    return {"intra_media": mi, "inter_media": me, "inter_sobre_intra": me / mi if mi > 0 else None}


def pca_dims(A: np.ndarray, fracs: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> dict[str, int | None]:
    """Componentes PCA para explicar cada fracción de varianza (tope: S − 1)."""
    Ac = np.asarray(A, dtype=np.float64) - np.mean(A, axis=0)
    s = np.linalg.svd(Ac, compute_uv=False)
    ev = s**2
    tot = ev.sum()
    if tot <= 0:
        return {f"pca_{int(f * 100)}": None for f in fracs}
    cum = np.cumsum(ev) / tot
    return {f"pca_{int(f * 100)}": int(np.searchsorted(cum, f) + 1) for f in fracs}


def twonn_dimension(D: np.ndarray) -> float | None:
    """TwoNN (Facco et al., 2017): d̂ = N / Σ log(r₂/r₁) con las distancias euclidianas."""
    D2 = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(D2, np.inf)
    part = np.partition(D2, 1, axis=1)[:, :2]
    r1, r2 = part[:, 0], part[:, 1]
    ok = r1 > 0
    if ok.sum() < 3:
        return None
    s = np.log(r2[ok] / r1[ok]).sum()
    return float(ok.sum() / s) if s > 0 else None


def eta_squared(values: np.ndarray, y: np.ndarray) -> float | None:
    """Fracción de la varianza de ``values`` explicada por la clase (ANOVA de una vía)."""
    v = np.asarray(values, dtype=np.float64)
    tot = ((v - v.mean()) ** 2).sum()
    if tot <= 0:
        return None
    between = sum(np.sum(y == c) * (v[y == c].mean() - v.mean()) ** 2 for c in np.unique(y))
    return float(between / tot)


def norm_profile(A: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Distribución de normas por muestra y cuánta información de tipo lleva la norma."""
    nrm = np.linalg.norm(A, axis=1)
    Dn = np.abs(nrm[:, None] - nrm[None, :])
    idx = knn_from_distances(Dn, 5)
    return {
        "cuantiles": quantile_summary(nrm),
        "cv": float(nrm.std() / nrm.mean()) if nrm.mean() > 0 else None,
        "eta2_norma_vs_tipo": eta_squared(nrm, y),
        "pureza@5_solo_norma": float((y[idx] == y[:, None]).mean()),
    }


def random_purity(y: np.ndarray) -> float:
    """Pureza esperada de vecinos al azar: Σ_t (n_t/n)·(n_t − 1)/(n − 1)."""
    n = len(y)
    _, c = np.unique(y, return_counts=True)
    return float(np.sum(c / n * (c - 1) / max(n - 1, 1)))


def analyze_distances(
    dists: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    ks: Sequence[int] = (1, 5, 10),
    k_hub: int = 10,
    k_agree: int = 10,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Diagnósticos de cada matriz de distancias y acuerdo entre ellas.

    Devuelve (resultados por métrica + acuerdo, vecinos k_agree por métrica).
    """
    n = len(y)
    kmax = max(max(ks), k_hub, k_agree)
    res: dict[str, Any] = {}
    neigh: dict[str, np.ndarray] = {}
    for name, D in dists.items():
        idx = knn_from_distances(D, kmax)
        neigh[name] = idx[:, :k_agree]
        res[name] = {
            **knn_purity(idx, y, ks),
            "concentracion": concentration(D),
            "hubness": hubness(idx[:, :k_hub], n),
            "separacion": class_separation(D, y),
        }
    names = list(dists)
    agree = {f"{a}|{b}": knn_agreement(neigh[a], neigh[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return {"metricas": res, "acuerdo": agree}, neigh


def matrix_variants(X: np.ndarray, data_type: str) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Variantes de preprocesamiento: (nombre, matriz S×F', binarización S×F'', descripción binarización)."""
    tname, tf = TRANSFORMS.get(data_type, ("ninguna", None))
    bdesc, bfun = NATURAL_BINARIZATION.get(data_type, ("x > 0", lambda M: M > 0))
    X = np.asarray(X, dtype=np.float64)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    raw = impute_feature_mean(X)
    Bnat = bfun(raw)
    out = [("crudo", raw, Bnat, bdesc)]
    prefix = "crudo"
    base = raw
    if tf is not None:
        base = impute_feature_mean(tf(X))
        out.append(("transformado", base, Bnat, bdesc))
        prefix = "transformado"
    Z = zscore_columns(base)
    Bz = Z > 1
    out.append((f"{prefix}+z", Z, Bz, Z_BINARIZATION))
    out.append((f"{prefix}+z+L2", l2_rows(Z), Bz, Z_BINARIZATION))
    return out


def analyze_matrix(
    X: np.ndarray,
    labels: Sequence[str],
    data_type: str,
    *,
    ks: Sequence[int] = (1, 5, 10),
    k_hub: int = 10,
    k_agree: int = 10,
    metrics: Sequence[str] = MATRIX_METRICS,
) -> dict[str, Any]:
    """Compara las medidas sobre la matriz sketch X (S muestras × F rasgos, puede traer NaN)."""
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    funcs = {
        "coseno": cosine_distance, "L2": euclidean_distance, "L1": manhattan_distance,
        "pearson": pearson_distance, "spearman": spearman_distance,
    }
    variants = matrix_variants(X, data_type)
    out: dict[str, Any] = {
        "n_muestras": int(X.shape[0]),
        "n_rasgos_sketch": int(X.shape[1]),
        "transformacion": TRANSFORMS.get(data_type, ("ninguna", None))[0],
        "clases": {str(c): int(v) for c, v in zip(classes, np.bincount(y))},
        "pureza_azar": random_purity(y),
        "variantes": {},
    }
    all_neigh: dict[tuple[str, str], np.ndarray] = {}
    for vname, M, B, bdesc in variants:
        dists = {}
        for m in metrics:
            if m in funcs:
                dists[m] = funcs[m](M)
            elif m == "jaccard":
                dists[m] = jaccard_distance(B)
            elif m == "hamming":
                dists[m] = hamming_distance(B)
        res, neigh = analyze_distances(dists, y, ks=ks, k_hub=k_hub, k_agree=k_agree)
        for m, idx in neigh.items():
            all_neigh[(vname, m)] = idx
        res.update({
            "n_rasgos_usados": int(M.shape[1]),
            "binarizacion": bdesc,
            "densidad_binaria": float(np.mean(B)),
            "dimension": {**pca_dims(M), "twonn": twonn_dimension(dists["L2"]) if "L2" in dists else None},
            "normas": norm_profile(M, y),
        })
        out["variantes"][vname] = res
    ref_v = variants[-1][0]
    ref = (ref_v, "coseno")
    out["referencia"] = f"{ref_v}/coseno"
    if ref in all_neigh:
        for (vname, m), idx in all_neigh.items():
            out["variantes"][vname]["metricas"][m]["acuerdo_con_referencia"] = knn_agreement(idx, all_neigh[ref])
    return out


# --------------------------------------------------------------------------
# Perfil de un tipo matricial consumiendo el stream
# --------------------------------------------------------------------------


def _progress(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def profile_matrix(
    root: Path | str,
    data_type: str,
    labels: Mapping[str, str],
    *,
    n_samples: int = 400,
    max_features: int = 20000,
    max_rows: int | None = None,
    seed: int = 2026,
    progress: int = 0,
    analyze: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Perfil de un tipo matricial (feature-major o RPPA) en una pasada del stream.

    Devuelve (resultado JSON-serializable, vectores de estadísticos por rasgo).
    RPPA (orientado por muestra) necesita dos pasadas: la primera solo lee las
    claves para estratificar; la segunda acumula estadísticos y guarda las S filas.
    """
    t0 = time.perf_counter()
    tname, tf = TRANSFORMS.get(data_type, ("ninguna", None))
    rng = np.random.default_rng(seed)
    svc = StreamService(root, data_types=[data_type], limit_per_type=max_rows)
    info: dict[str, Any] = {
        "tipo": data_type,
        "archivos": [s.path.name for s in svc.sources],
        "max_rows": max_rows,
        "semilla": seed,
        "transformacion": tname,
    }
    orientation = None
    stream = svc.stream()
    first = next(stream)
    if first.kind != "schema":
        raise ValueError(f"{data_type}: se esperaba un registro schema al inicio")
    orientation = first.fields["orientation"]
    columns = list(first.fields["columns"])
    info["orientacion"] = orientation
    feature_ids: list[str] = []
    end_fields: dict[str, Any] = {}

    if orientation == "feature":
        col_idx, y, sel = select_samples(columns, labels, n_samples, seed)
        info["seleccion"] = sel
        S = len(col_idx)
        res = ReservoirSampler(max_features, rng)
        Xr = np.empty((max_features, S), dtype=np.float32)
        arrival = np.full(max_features, -1, dtype=np.int64)
        res_ids: list[str | None] = [None] * max_features
        stats_rows: list[tuple] = []
        sacc = SampleAccumulator(len(columns), tf)
        n = 0
        for rec in stream:
            if rec.kind == "end":
                end_fields = dict(rec.fields)
                continue
            if rec.kind != "feature_row":
                continue
            x = rec.values
            stats_rows.append(row_stats(x, tf))
            feature_ids.append(rec.key)
            sacc.update(x)
            slot = res.offer()
            if slot >= 0:
                Xr[slot] = x[col_idx]
                arrival[slot] = n
                res_ids[slot] = rec.key
            n += 1
            if progress and n % progress == 0:
                el = time.perf_counter() - t0
                _progress(f"  .. {data_type}: {n:,} rasgos, {el:,.0f} s ({n / el:,.0f} rasgos/s)", True)
        fill = res.size
        order = np.argsort(arrival[:fill], kind="stable")
        X = Xr[:fill][order].T.copy()
        sketch_ids = [res_ids[i] for i in order]
        st = {k: np.asarray(v, dtype=np.float64) for k, v in zip(STAT_NAMES, zip(*stats_rows))} if stats_rows \
            else {k: np.zeros(0) for k in STAT_NAMES}
        n_cols = len(columns)
        # Normas y NA por muestra sobre todos los rasgos leídos (columnas elegibles).
        pat_to_col, _ = eligible_by_patient(columns, labels)
        elig = np.array(sorted(pat_to_col.values()), dtype=np.int64)
        y_elig = np.array([labels[patient_barcode(columns[i])] for i in elig])
        per_sample = {
            "frac_na_por_muestra": quantile_summary(1 - sacc.n_valid / max(sacc.n_rows, 1)),
            "norma_cruda_todas_columnas": quantile_summary(np.sqrt(sacc.sumsq)),
            "norma_transformada_todas_columnas": quantile_summary(np.sqrt(sacc.sumsq_t)),
        }
        if elig.size > 1:
            nt = np.sqrt(sacc.sumsq_t[elig])
            nr = np.sqrt(sacc.sumsq[elig])
            per_sample.update({
                "n_muestras_elegibles": int(elig.size),
                "eta2_norma_cruda_vs_tipo": eta_squared(nr, y_elig),
                "eta2_norma_transformada_vs_tipo": eta_squared(nt, y_elig),
                "norma_transformada_mediana_por_tipo": {t: float(np.median(nt[y_elig == t])) for t in np.unique(y_elig)},
            })
        info["por_muestra_pasada_completa"] = per_sample
    else:
        # Orientado por muestra: columnas = rasgos.
        keys = [first_key for first_key in _sample_keys(stream, end_fields)]
        info["muestras_en_archivo"] = len(keys)
        row_idx, y, sel = select_samples(keys, labels, n_samples, seed)
        info["seleccion"] = sel
        wanted = {int(i): j for j, i in enumerate(row_idx)}
        d = len(columns)
        colst = ColumnStats(d, tf)
        X = np.full((len(row_idx), d), np.nan, dtype=np.float32)
        tumor_type_check = Counter()
        i = -1
        for rec in svc.stream():
            if rec.kind == "end":
                end_fields = dict(rec.fields)
                continue
            if rec.kind != "sample_row":
                continue
            i += 1
            colst.update(rec.values)
            if i in wanted:
                X[wanted[i]] = rec.values
                tt = rec.fields.get("TumorType")
                if tt is not None:
                    tumor_type_check["coincide" if tt == y[wanted[i]] else "difiere"] += 1
        if tumor_type_check:
            sel["etiqueta_archivo_vs_cdr"] = dict(tumor_type_check)
        st = colst.finalize()
        feature_ids = columns
        if d > max_features:
            keep = np.sort(rng.choice(d, size=max_features, replace=False))
            X = X[:, keep]
            sketch_ids = [columns[j] for j in keep]
        else:
            sketch_ids = columns
        n_cols = colst.n_rows

    info["filas_leidas"] = end_fields.get("records")
    info["truncado"] = end_fields.get("truncated")
    info["n_columnas"] = len(columns) if orientation == "feature" else n_cols
    info["primer_rasgo"] = feature_ids[0] if feature_ids else None
    info["ultimo_rasgo"] = feature_ids[-1] if feature_ids else None
    info["estadisticos_por_rasgo"] = summarize_feature_stats(st, n_cols)
    info["sketch"] = {
        "S": int(X.shape[0]),
        "F": int(X.shape[1]),
        "metodo_rasgos": ("reservoir sampling (Algorithm R) sobre las filas del stream"
                          if orientation == "feature" else "todos los rasgos (o muestra aleatoria si d > F)"),
        "metodo_muestras": "estratificada por tipo de cáncer, cuotas iguales con llenado, 1 columna por paciente, códigos 01/03/06",
        "frac_na": float(np.mean(np.isnan(X))) if X.size else None,
        "ids_ejemplo": list(sketch_ids[:5]),
    }
    info["seg_lectura"] = round(time.perf_counter() - t0, 2)
    if analyze and X.shape[0] >= 3 and X.shape[1] >= 2:
        t1 = time.perf_counter()
        info["analisis"] = analyze_matrix(X, y, data_type)
        info["seg_analisis"] = round(time.perf_counter() - t1, 2)
    st_out = {k: v.astype(np.float32) for k, v in st.items()}
    st_out["feature_id"] = np.asarray(feature_ids, dtype=np.str_)
    return info, st_out


def _sample_keys(stream: Iterable, end_fields: dict) -> Iterable[str]:
    for rec in stream:
        if rec.kind == "end":
            end_fields.update(rec.fields)
        elif rec.kind == "sample_row":
            yield rec.key


# --------------------------------------------------------------------------
# Mutaciones (eventos del MAF)
# --------------------------------------------------------------------------


def profile_mutations(
    root: Path | str,
    labels: Mapping[str, str],
    *,
    n_samples: int = 400,
    seed: int = 2026,
    top_genes: int = 500,
    max_rows: int | None = None,
    progress: int = 0,
    analyze: bool = True,
) -> dict[str, Any]:
    """Perfil de los conjuntos de genes mutados por muestra (una pasada del MAF).

    - Para todas las muestras: número de eventos y de genes distintos (todas las
      variantes y solo no silenciosas), agrupando claves contiguas del stream.
    - Para un *pool* de pacientes elegido antes de leer (estratificado por tipo,
      con sobremuestreo ×3 porque no se sabe quién está en el MAF), se guardan
      los conjuntos y al final se toman S pacientes presentes.
    """
    t0 = time.perf_counter()
    by_type: dict[str, list[str]] = defaultdict(list)
    for p in sorted(labels):
        by_type[labels[p]].append(p)
    ordered, alloc = stratified_pick(by_type, n_samples, seed)
    pool = {p for t in ordered for p in ordered[t][: 3 * alloc[t] + 5]}

    sizes: dict[str, list[int]] = {}
    gene_freq_all: Counter = Counter()
    gene_freq_ns: Counter = Counter()
    pool_sets: dict[str, tuple[set, set]] = {}
    vc_counter: Counter = Counter()
    n_pass = 0
    finished: set[str] = set()
    noncontig = 0
    cur_key: str | None = None
    cur_all: set = set()
    cur_ns: set = set()
    cur_ev = [0, 0]

    def flush() -> None:
        nonlocal noncontig
        if cur_key is None:
            return
        if cur_key in finished:
            noncontig += 1
            prev = sizes[cur_key]
            sizes[cur_key] = [prev[0] + cur_ev[0], prev[1] + cur_ev[1], prev[2] + len(cur_all), prev[3] + len(cur_ns)]
        else:
            sizes[cur_key] = [cur_ev[0], cur_ev[1], len(cur_all), len(cur_ns)]
            finished.add(cur_key)
        gene_freq_all.update(cur_all)
        gene_freq_ns.update(cur_ns)
        if patient_barcode(cur_key) in pool:
            a, b = pool_sets.setdefault(cur_key, (set(), set()))
            a |= cur_all
            b |= cur_ns

    n = 0
    end_fields: dict[str, Any] = {}
    for rec in StreamService(root, data_types=["mutations"], limit_per_type=max_rows):
        if rec.kind == "end":
            end_fields = dict(rec.fields)
            continue
        if rec.kind != "event":
            continue
        key = rec.key
        if key != cur_key:
            flush()
            cur_key, cur_all, cur_ns, cur_ev = key, set(), set(), [0, 0]
        f = rec.fields
        gene = f.get("Hugo_Symbol") or ""
        vc = f.get("Variant_Classification") or ""
        vc_counter[vc] += 1
        if f.get("FILTER") == "PASS":
            n_pass += 1
        cur_ev[0] += 1
        cur_all.add(gene)
        if vc in NON_SILENT:
            cur_ev[1] += 1
            cur_ns.add(gene)
        n += 1
        if progress and n % progress == 0:
            el = time.perf_counter() - t0
            _progress(f"  .. mutations: {n:,} eventos, {el:,.0f} s ({n / el:,.0f} ev/s)", True)
    flush()

    # Una muestra tumoral por paciente con etiqueta.
    barcodes = sorted(sizes)
    pat_to_col, funnel = eligible_by_patient(barcodes, labels)
    pats = sorted(pat_to_col)
    chosen_bc = {p: barcodes[pat_to_col[p]] for p in pats}
    y_all = np.array([labels[p] for p in pats])
    ns_genes = np.array([sizes[chosen_bc[p]][3] for p in pats], dtype=np.float64)
    all_genes = np.array([sizes[chosen_bc[p]][2] for p in pats], dtype=np.float64)
    ns_events = np.array([sizes[chosen_bc[p]][1] for p in pats], dtype=np.float64)
    order = np.argsort(-ns_genes, kind="stable")[:10]
    info: dict[str, Any] = {
        "tipo": "mutations",
        "max_rows": max_rows,
        "semilla": seed,
        "eventos_leidos": end_fields.get("records"),
        "truncado": end_fields.get("truncated"),
        "muestras_en_maf": len(sizes),
        "claves_no_contiguas": noncontig,
        "frac_filter_pass": n_pass / n if n else None,
        "clasificaciones": dict(vc_counter.most_common()),
        "genes_distintos_todas": len(gene_freq_all),
        "genes_distintos_no_silenciosas": len(gene_freq_ns),
        "embudo": funnel,
        "tamanos_todos_los_pacientes": {
            "n_pacientes": len(pats),
            "genes_todas": quantile_summary(all_genes),
            "genes_no_silenciosas": quantile_summary(ns_genes),
            "eventos_no_silenciosos": quantile_summary(ns_events),
            "cv_genes_no_silenciosas": float(ns_genes.std() / ns_genes.mean()) if ns_genes.size and ns_genes.mean() > 0 else None,
            "frac_cero_no_silenciosas": float(np.mean(ns_genes == 0)) if ns_genes.size else None,
            "frac_ge_500_genes_ns": float(np.mean(ns_genes >= 500)) if ns_genes.size else None,
            "frac_ge_1000_genes_ns": float(np.mean(ns_genes >= 1000)) if ns_genes.size else None,
            "eta2_log_tamano_vs_tipo": eta_squared(np.log1p(ns_genes), y_all) if ns_genes.size > 1 else None,
            "mediana_genes_ns_por_tipo": {t: float(np.median(ns_genes[y_all == t])) for t in np.unique(y_all)},
            "top10_hipermutados": [
                {"muestra": chosen_bc[pats[i]], "tipo": str(y_all[i]), "genes_ns": int(ns_genes[i])} for i in order
            ],
        },
        "genes_mas_frecuentes_ns": [(g, c) for g, c in sorted(gene_freq_ns.items(), key=lambda t: (-t[1], t[0]))[:30]],
    }

    # Elección de S pacientes presentes en el MAF dentro del pool.
    present = {p for p in pats if chosen_bc[p] in pool_sets}
    sel_pats = []
    for t in sorted(ordered):
        sel_pats += [p for p in ordered[t] if p in present][: alloc[t]]
    sel_pats.sort()
    info["seleccion"] = {"pool": len(pool), "presentes_en_maf": len(present),
                         "elegidos_por_tipo": dict(sorted(Counter(labels[p] for p in sel_pats).items()))}
    top = [g for g, _ in sorted(gene_freq_ns.items(), key=lambda t: (-t[1], t[0]))[:top_genes]]
    top_set = set(top)
    variants = {
        "todas": [pool_sets[chosen_bc[p]][0] for p in sel_pats],
        "no_silenciosas": [pool_sets[chosen_bc[p]][1] for p in sel_pats],
        f"no_silenciosas_top{top_genes}": [pool_sets[chosen_bc[p]][1] & top_set for p in sel_pats],
    }
    info["seg_lectura"] = round(time.perf_counter() - t0, 2)
    if analyze and len(sel_pats) >= 3:
        t1 = time.perf_counter()
        info["analisis"] = analyze_sets(variants, [labels[p] for p in sel_pats])
        info["seg_analisis"] = round(time.perf_counter() - t1, 2)
    return info


def sets_to_matrix(sets: Sequence[set]) -> tuple[np.ndarray, list[str]]:
    genes = sorted(set().union(*sets)) if sets else []
    pos = {g: j for j, g in enumerate(genes)}
    B = np.zeros((len(sets), len(genes)), dtype=bool)
    for i, s in enumerate(sets):
        if s:
            B[i, [pos[g] for g in s]] = True
    return B, genes


def analyze_sets(
    variants: Mapping[str, Sequence[set]],
    labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5, 10),
    k_hub: int = 10,
    k_agree: int = 10,
) -> dict[str, Any]:
    """Jaccard, coseno binario y Hamming entre conjuntos de genes mutados."""
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    out: dict[str, Any] = {
        "n_muestras": len(y),
        "clases": {str(c): int(v) for c, v in zip(classes, np.bincount(y))},
        "pureza_azar": random_purity(y),
        "variantes": {},
    }
    for vname, sets in variants.items():
        B, genes = sets_to_matrix(sets)
        sz = B.sum(axis=1)
        dists = {"jaccard": jaccard_distance(B), "coseno_binario": binary_cosine_distance(B), "hamming": hamming_distance(B)}
        res, _ = analyze_distances(dists, y, ks=ks, k_hub=k_hub, k_agree=k_agree)
        per_gene = B.sum(axis=0)
        res.update({
            "universo_genes": len(genes),
            "densidad": float(B.mean()) if B.size else 0.0,
            "tamano_conjunto": quantile_summary(sz),
            "frac_conjuntos_vacios": float(np.mean(sz == 0)),
            "frac_genes_en_una_sola_muestra": float(np.mean(per_gene == 1)) if per_gene.size else None,
            "corr_spearman_tamano_vs_Nk_hamming": _size_vs_hub(sz, dists["hamming"], k_hub),
            "corr_spearman_tamano_vs_Nk_jaccard": _size_vs_hub(sz, dists["jaccard"], k_hub),
        })
        out["variantes"][vname] = res
    return out


def _size_vs_hub(sizes: np.ndarray, D: np.ndarray, k: int) -> float | None:
    """¿Los hubs son las muestras con pocos (o muchos) genes mutados? ρ de Spearman entre tamaño y N_k."""
    idx = knn_from_distances(D, k)
    nk = np.bincount(idx.ravel(), minlength=len(sizes))
    if np.std(sizes) == 0 or np.std(nk) == 0:
        return None
    return float(sps.spearmanr(sizes, nk).statistic)


# --------------------------------------------------------------------------
# Salidas: JSON por tipo y RESUMEN.md
# --------------------------------------------------------------------------


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(obj: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def _f(v: Any, nd: int = 2, pct: bool = False) -> str:
    if v is None:
        return "–"
    if isinstance(v, (int, np.integer)) and not pct:
        return f"{v:,}".replace(",", ".")
    v = float(v) * (100 if pct else 1)
    if not math.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:.{nd}f}".replace(".", ",") + (" %" if pct else "")


def _metric_tables(an: Mapping[str, Any], metrics: Sequence[str]) -> list[str]:
    variants = list(an["variantes"])
    head = "| métrica | " + " | ".join(variants) + " |"
    sep = "|---|" + "---|" * len(variants)
    blocks = []
    specs = [
        ("Pureza@5 de vecinos (voto@10 entre paréntesis)",
         lambda r: f"{_f(r['pureza@5'])} ({_f(r.get('voto@10'))})"),
        ("Contraste relativo mediano (Dmax−Dmin)/Dmin  [CV de distancias]",
         lambda r: f"{_f(r['concentracion']['contraste_relativo_mediana'])} [{_f(r['concentracion']['cv_distancias'])}]"),
        ("Hubness: asimetría de N_10  [máx N_10]",
         lambda r: f"{_f(r['hubness']['asimetria_Nk'])} [{_f(r['hubness']['max_Nk'])}]"),
        ("Acuerdo con la referencia (Jaccard de los 10-NN)",
         lambda r: _f(r.get("acuerdo_con_referencia"))),
    ]
    for title, fn in specs:
        if title.startswith("Acuerdo") and not any(
                "acuerdo_con_referencia" in an["variantes"][v]["metricas"].get(m, {}) for v in variants for m in metrics):
            continue
        rows = [f"**{title}**", "", head, sep]
        for m in metrics:
            cells = []
            for v in variants:
                r = an["variantes"][v]["metricas"].get(m)
                cells.append(fn(r) if r else "–")
            rows.append(f"| {m} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(rows))
    return blocks


def summary_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    """Genera RESUMEN.md a partir de los JSON de cada tipo."""
    L = ["# Perfil de los datos y aplicabilidad de las medidas de similitud", "",
         "Generado por `scripts/perfil_datos.py` (módulo `tcgaknn.perfil`). Cada sección resume "
         "`results/perfil/<tipo>.json`.", "",
         "- **Estadísticos por rasgo**: exactos sobre las filas leídas del stream (todas, salvo que se indique límite).",
         "- **Comparación de medidas**: sobre un *sketch* S × F (S muestras estratificadas por tipo de cáncer, "
         "una por paciente, códigos 01/03/06; F rasgos por reservoir sampling). Pureza@k = fracción de los k "
         "vecinos (leave-one-out) con el mismo tipo; voto@k = exactitud del voto por mayoría.",
         "- **Contraste relativo** cercano a 0 = distancias concentradas (vecinos poco distinguibles). "
         "**Hubness** (asimetría de N_10) > 1 = unos pocos puntos aparecen en muchas listas de vecinos.",
         "- **Referencia** = coseno sobre la última variante (transformado + z-score + L2), la verdad de terreno del plan (§2.5).",
         "- Jaccard y Hamming usan la binarización indicada por variante.", "",
         "Pendiente: número de copias (`copy_number`, `absolute_segments`) no se perfila: requiere llevar "
         "segmentos a genes.", ""]
    for r in results:
        t = r.get("tipo", "?")
        L.append(f"## {t}")
        L.append("")
        if t == "mutations":
            L += _mutations_md(r)
            continue
        lim = f"límite --max-rows {r['max_rows']:,}".replace(",", ".") if r.get("max_rows") else "archivo completo"
        L.append(f"- Archivo: `{', '.join(r.get('archivos', []))}` ({r.get('orientacion')}); filas leídas: "
                 f"{_f(r.get('filas_leidas'))} ({lim}; truncado = {r.get('truncado')}); columnas: {_f(r.get('n_columnas'))}; "
                 f"lectura {_f(r.get('seg_lectura'), 0)} s, análisis {_f(r.get('seg_analisis'), 0)} s.")
        if r.get("max_rows") and r.get("truncado"):
            L.append(f"- **Sesgo por límite**: solo las primeras filas en el orden del archivo "
                     f"(de `{r.get('primer_rasgo')}` a `{r.get('ultimo_rasgo')}`); los estadísticos y el reservoir "
                     "cubren solo ese prefijo.")
        g = r["estadisticos_por_rasgo"]["global"]
        q = r["estadisticos_por_rasgo"]["cuantiles"]
        L.append(f"- Valores: NA {_f(g['frac_na'], 3, True)}, ceros {_f(g['frac_cero'], 1, True)}, negativos "
                 f"{_f(g['frac_neg'], 3, True)}, enteros {_f(g['frac_entero'], 1, True)}, en [0,1] {_f(g['frac_01'], 1, True)}; "
                 f"rango [{_f(g['min'])}, {_f(g['max'])}]; rasgos con NA {_f(g['rasgos_con_na'])}, con varianza 0 "
                 f"{_f(g['rasgos_varianza_cero'])}.")
        L.append(f"- Varianza por rasgo (cruda): mediana {_f(q['varianza']['q50'], 3)}, q99/q01 = "
                 f"{_f(g['cociente_var_q99_q01'])}; transformada q99/q01 = {_f(g['cociente_var_t_q99_q01'])}; el 1 % de "
                 f"rasgos con más varianza concentra el {_f(g['frac_varianza_top1pct_rasgos'], 1, True)} de la varianza cruda. "
                 f"Asimetría mediana por rasgo: cruda {_f(g['asimetria_mediana'])}, transformada {_f(g['asimetria_t_mediana'])}.")
        ps = r.get("por_muestra_pasada_completa")
        if ps and ps.get("eta2_norma_transformada_vs_tipo") is not None:
            L.append(f"- Norma por muestra (pasada completa, {_f(ps['n_muestras_elegibles'])} muestras elegibles): "
                     f"η² norma cruda vs tipo = {_f(ps['eta2_norma_cruda_vs_tipo'])}, transformada = "
                     f"{_f(ps['eta2_norma_transformada_vs_tipo'])}.")
        sel = r.get("seleccion", {})
        if "etiqueta_archivo_vs_cdr" in sel:
            L.append(f"- Etiqueta del archivo vs CDR en la muestra: {sel['etiqueta_archivo_vs_cdr']}.")
        for note in r["estadisticos_por_rasgo"].get("diagnostico", []):
            L.append(f"- _{note}_")
        an = r.get("analisis")
        if not an:
            L.append("")
            continue
        L.append(f"- Sketch: S = {an['n_muestras']} muestras, {len(an['clases'])} tipos, F = {an['n_rasgos_sketch']} "
                 f"rasgos; pureza al azar = {_f(an['pureza_azar'], 3)}; transformación: {an['transformacion']}.")
        L.append("")
        L += [b + "\n" for b in _metric_tables(an, MATRIX_METRICS)]
        L.append("**Dimensión intrínseca y normas por variante**")
        L.append("")
        L.append("| variante | rasgos | PCA 90 % | PCA 50 % | TwoNN | CV norma | η² norma~tipo | pureza@5 solo norma | binarización (densidad) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v, rv in an["variantes"].items():
            d, nm = rv["dimension"], rv["normas"]
            L.append(f"| {v} | {_f(rv['n_rasgos_usados'])} | {_f(d.get('pca_90'))} | {_f(d.get('pca_50'))} | "
                     f"{_f(d.get('twonn'), 1)} | {_f(nm['cv'])} | {_f(nm['eta2_norma_vs_tipo'])} | "
                     f"{_f(nm['pureza@5_solo_norma'])} | {rv['binarizacion']} ({_f(rv['densidad_binaria'], 2)}) |")
        ref_v = list(an["variantes"])[-1]
        ag = an["variantes"][ref_v]["acuerdo"]
        L.append("")
        L.append(f"Acuerdo entre medidas en `{ref_v}` (Jaccard de 10-NN): " +
                 ", ".join(f"{k} {_f(v)}" for k, v in ag.items() if k.startswith("coseno|")) + ".")
        L.append("")
    return "\n".join(L) + "\n"


def _mutations_md(r: Mapping[str, Any]) -> list[str]:
    L = []
    tz = r["tamanos_todos_los_pacientes"]
    gq = tz["genes_no_silenciosas"]
    lim = f"límite {r['max_rows']}" if r.get("max_rows") else "archivo completo"
    L.append(f"- Eventos leídos: {_f(r.get('eventos_leidos'))} ({lim}); muestras en el MAF: {_f(r['muestras_en_maf'])}; "
             f"claves no contiguas: {r['claves_no_contiguas']}; FILTER = PASS: {_f(r['frac_filter_pass'], 1, True)}; "
             f"lectura {_f(r.get('seg_lectura'), 0)} s.")
    L.append(f"- Genes mutados no silenciosos por paciente ({_f(tz['n_pacientes'])} pacientes con etiqueta): mediana "
             f"{_f(gq['q50'], 0)}, q05 {_f(gq['q05'], 0)}, q95 {_f(gq['q95'], 0)}, máx {_f(gq['q100'], 0)}; CV "
             f"{_f(tz['cv_genes_no_silenciosas'])}; sin ninguna: {_f(tz['frac_cero_no_silenciosas'], 1, True)}; "
             f"≥ 500 genes: {_f(tz['frac_ge_500_genes_ns'], 1, True)}; ≥ 1000: {_f(tz['frac_ge_1000_genes_ns'], 1, True)}; "
             f"η²(log tamaño ~ tipo) = {_f(tz['eta2_log_tamano_vs_tipo'])}.")
    top = ", ".join(f"{h['tipo']} {h['genes_ns']}" for h in tz["top10_hipermutados"][:5])
    L.append(f"- Más hipermutados: {top}.")
    an = r.get("analisis")
    if not an:
        return L + [""]
    L.append(f"- Sketch: S = {an['n_muestras']} pacientes, {len(an['clases'])} tipos; pureza al azar = {_f(an['pureza_azar'], 3)}.")
    L.append("")
    L += [b + "\n" for b in _metric_tables(an, SET_METRICS)]
    L.append("| variante | universo de genes | densidad | tamaño mediano | vacíos | genes en 1 sola muestra | ρ(tamaño, N_10) Hamming | ρ(tamaño, N_10) Jaccard |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v, rv in an["variantes"].items():
        L.append(f"| {v} | {_f(rv['universo_genes'])} | {_f(rv['densidad'], 4)} | {_f(rv['tamano_conjunto']['q50'], 0)} | "
                 f"{_f(rv['frac_conjuntos_vacios'], 1, True)} | {_f(rv['frac_genes_en_una_sola_muestra'], 1, True)} | "
                 f"{_f(rv['corr_spearman_tamano_vs_Nk_hamming'])} | {_f(rv['corr_spearman_tamano_vs_Nk_jaccard'])} |")
    L.append("")
    return L


def write_summary(out_dir: Path | str) -> Path:
    """Regenera ``RESUMEN.md`` con todos los ``<tipo>.json`` presentes en ``out_dir``."""
    out_dir = Path(out_dir)
    results = []
    for t in ALL_TYPES:
        p = out_dir / f"{t}.json"
        if p.exists():
            results.append(json.loads(p.read_text(encoding="utf-8")))
    path = out_dir / "RESUMEN.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(summary_markdown(results), encoding="utf-8")
    tmp.replace(path)
    return path
