"""Métricas de evaluación: calidad de la búsqueda, voto k-NN y clasificación.

Calidad de la búsqueda (plan §4.2.1), con E_k(q) los k vecinos exactos y
A_k(q) los devueltos por el método (relleno -1 = no devuelto):

- ``recall_at_k``: |A_k ∩ E_k| / |E_k|, promediado sobre las consultas. Los
  -1 cuentan como no recuperados (|E_k| = k salvo que la base tenga < k filas).
- ``precision_at_k``: |A_k ∩ E_k| / |A_k|; difiere del recall solo si |A_k| < k.
- ``approx_ratio``: (1/k)·Σ_i dist(q, a_i)/dist(q, e_i), con ambas listas
  ordenadas de menor a mayor distancia; 1 = perfecto.
- ``label_purity``: fracción de los vecinos devueltos que comparten la etiqueta
  de la consulta (no depende de EX).

Voto k-NN (plan §3.7):

- ``knn_vote``: mayoría simple o ponderada; desempate por la clase del vecino
  más cercano entre las empatadas; los -1 se ignoran; sin vecinos, etiqueta
  vacía (cuenta como error).
- ``vote_agreement``: fracción de consultas en que dos votos coinciden.

Clasificación (plan §4.2.3), implementación propia verificada contra
scikit-learn (``accuracy_score``, ``f1_score(average="macro")``,
``balanced_accuracy_score``, ``confusion_matrix``):

- F1 de la clase c = 2·TP_c / (2·TP_c + FP_c + FN_c) (0 si el denominador es 0),
  equivalente a 2PR/(P + R); F1-macro = promedio sin ponderar sobre las clases.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- Exactitud balanceada = media del recall por clase, sobre las clases presentes
  en y_true (Brodersen et al., 2010).
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

La etiqueta vacía (predicción sin vecinos) no es una clase: no aparece en
``labels`` ni en la matriz de confusión, pero cuenta como error en la
exactitud y como falso negativo de la clase verdadera.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Calidad de la búsqueda
# --------------------------------------------------------------------------


def _as_idx(a: np.ndarray, k: int) -> np.ndarray:
    A = np.atleast_2d(np.asarray(a))
    if not np.issubdtype(A.dtype, np.integer):
        raise TypeError("los índices deben ser enteros")
    if A.shape[1] < k:
        A = np.pad(A, ((0, 0), (0, k - A.shape[1])), constant_values=-1)
    return A[:, :k].astype(np.int64)


def _hits(approx_idx: np.ndarray, exact_idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Por consulta: (|A ∩ E|, |A|, |E|), sin contar -1 y sin contar repetidos."""
    A = _as_idx(approx_idx, k)
    E = _as_idx(exact_idx, k)
    if A.shape[0] != E.shape[0]:
        raise ValueError(f"número de consultas distinto: {A.shape[0]} y {E.shape[0]}")
    hits = np.empty(A.shape[0], dtype=np.int64)
    n_a = np.empty(A.shape[0], dtype=np.int64)
    n_e = np.empty(A.shape[0], dtype=np.int64)
    for i, (a, e) in enumerate(zip(A, E)):
        sa = set(a[a >= 0].tolist())
        se = set(e[e >= 0].tolist())
        hits[i], n_a[i], n_e[i] = len(sa & se), len(sa), len(se)
    return hits, n_a, n_e


def recall_at_k(
    approx_idx: np.ndarray, exact_idx: np.ndarray, k: int, *, per_query: bool = False
) -> float | np.ndarray:
    """recall@k = |A_k ∩ E_k| / |E_k| (= /k si hay al menos k filas en la base).

    Los -1 de ``approx_idx`` cuentan como no recuperados; un índice repetido
    cuenta una sola vez.
    """
    hits, _, n_e = _hits(approx_idx, exact_idx, k)
    r = hits / np.maximum(n_e, 1)
    return r if per_query else float(r.mean()) if r.size else float("nan")


def precision_at_k(
    approx_idx: np.ndarray, exact_idx: np.ndarray, k: int, *, per_query: bool = False
) -> float | np.ndarray:
    """precision@k = |A_k ∩ E_k| / |A_k|; NaN en las consultas sin resultados (se omiten en la media)."""
    hits, n_a, _ = _hits(approx_idx, exact_idx, k)
    p = np.divide(hits, n_a, out=np.full(hits.shape, np.nan), where=n_a > 0)
    if per_query:
        return p
    return float(np.nanmean(p)) if np.any(n_a > 0) else float("nan")


def approx_ratio(
    dist_aprox: np.ndarray, dist_exactas: np.ndarray, *, per_query: bool = False, eps: float = 1e-12
) -> float | np.ndarray:
    """Razón de aproximación (1/k)·Σ_i d(q, a_i)/d(q, e_i).

    Ambas matrices (nq × k) son **distancias** ordenadas de menor a mayor (para
    coseno, pasar 1 − cos o el ángulo). Las posiciones con NaN en cualquiera
    de las dos (vecinos no devueltos) se omiten. Si d(q, e_i) ≤ eps, la razón
    vale 1 cuando también d(q, a_i) ≤ eps y se omite en otro caso (no está
    definida). Las consultas sin posiciones válidas dan NaN.
    """
    A = np.atleast_2d(np.asarray(dist_aprox, dtype=np.float64))
    E = np.atleast_2d(np.asarray(dist_exactas, dtype=np.float64))
    if A.shape != E.shape:
        raise ValueError(f"formas distintas: {A.shape} y {E.shape}")
    valid = ~np.isnan(A) & ~np.isnan(E)
    zero_e = E <= eps
    ratio = np.divide(A, E, out=np.ones_like(A), where=~zero_e)
    valid &= ~(zero_e & (A > eps))
    cnt = valid.sum(axis=1)
    per_q = np.divide(np.where(valid, ratio, 0.0).sum(axis=1), cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0)
    if per_query:
        return per_q
    return float(np.nanmean(per_q)) if np.any(cnt > 0) else float("nan")


def label_purity(
    neighbor_idx: np.ndarray,
    base_labels: Sequence[Any] | np.ndarray,
    query_labels: Sequence[Any] | np.ndarray,
    *,
    count_missing: bool = False,
    per_query: bool = False,
) -> float | np.ndarray:
    """Fracción de los vecinos devueltos con la misma etiqueta que la consulta.

    Por defecto el denominador es el número de vecinos devueltos (sin -1) y las
    consultas sin vecinos dan NaN; con ``count_missing`` el denominador es k y
    los faltantes cuentan como no coincidentes.
    """
    N = np.atleast_2d(np.asarray(neighbor_idx, dtype=np.int64))
    bl = np.asarray(base_labels)
    ql = np.asarray(query_labels)
    if ql.shape[0] != N.shape[0]:
        raise ValueError("query_labels debe tener una etiqueta por consulta")
    valid = N >= 0
    same = valid & (bl[np.maximum(N, 0)] == ql[:, None])
    den = np.full(N.shape[0], N.shape[1]) if count_missing else valid.sum(axis=1)
    p = np.divide(same.sum(axis=1), den, out=np.full(N.shape[0], np.nan), where=den > 0)
    if per_query:
        return p
    return float(np.nanmean(p)) if np.any(den > 0) else float("nan")


# --------------------------------------------------------------------------
# Voto k-NN
# --------------------------------------------------------------------------


def _default_empty(labels: np.ndarray) -> Any:
    """Etiqueta vacía por tipo: "" para texto, -1 para enteros, None para objetos."""
    if labels.dtype.kind in "US":
        return labels.dtype.type()
    if labels.dtype.kind in "iu":
        return -1
    if labels.dtype.kind == "f":
        return np.nan
    return None


def _is_auto(v: Any) -> bool:
    return isinstance(v, str) and v == "auto"


def _fits(value: Any, dtype: np.dtype) -> bool:
    """True si ``value`` se puede guardar sin pérdida en un arreglo de ``dtype``."""
    if value is None or dtype == object:
        return dtype == object
    try:
        back = np.array([value], dtype=dtype)[0]
    except (ValueError, TypeError, OverflowError):
        return False
    if isinstance(value, float) and np.isnan(value):
        return bool(np.isnan(back))
    return bool(back == value)


def knn_vote(
    neighbor_idx: np.ndarray,
    base_labels: Sequence[Any] | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    empty_label: Any = "auto",
) -> np.ndarray:
    """Predicción por voto de los vecinos (plan §3.7).

    - ``neighbor_idx`` (nq × k): vecinos ordenados del más al menos cercano;
      -1 = no hay vecino en esa posición (se ignora).
    - Sin ``weights``: mayoría simple, ŷ = argmax_c #{vecinos de clase c}.
    - Con ``weights`` (nq × k, p. ej. la similitud coseno): ŷ = argmax_c Σ w_i
      sobre los vecinos de clase c; los pesos NaN se ignoran como los -1.
    - Desempate: entre las clases empatadas en el máximo, gana la del vecino
      más cercano (primera columna cuya clase está empatada).
    - Consulta sin vecinos válidos → ``empty_label`` ("" para etiquetas de
      texto, -1 para enteras, None para objetos), que cuenta como error.

    Devuelve un arreglo de largo nq con el dtype de ``base_labels`` (objeto si
    la etiqueta vacía no cabe en ese dtype).
    """
    N = np.atleast_2d(np.asarray(neighbor_idx, dtype=np.int64))
    bl = np.asarray(base_labels)
    nq, k = N.shape
    classes, codes = np.unique(bl, return_inverse=True)
    C = classes.size
    valid = N >= 0
    if weights is None:
        W = np.ones(N.shape, dtype=np.float64)
    else:
        W = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        if W.shape != N.shape:
            raise ValueError(f"weights tiene forma {W.shape}, se esperaba {N.shape}")
        valid &= ~np.isnan(W)
    nc = np.where(valid, codes[np.maximum(N, 0)] if C else 0, 0)  # clase de cada vecino
    votes = np.zeros((nq, max(C, 1)), dtype=np.float64)
    rows = np.broadcast_to(np.arange(nq)[:, None], N.shape)
    np.add.at(votes, (rows[valid], nc[valid]), W[valid])
    # Clases empatadas en el máximo (solo entre las que recibieron algún voto)
    voted = np.zeros(votes.shape, dtype=bool)
    voted[rows[valid], nc[valid]] = True
    vmax = np.where(voted, votes, -np.inf).max(axis=1, keepdims=True)
    tol = 1e-9 * np.maximum(np.abs(vmax), 1.0)
    tied = voted & (np.abs(votes - vmax) <= tol)
    # Primera columna (vecino más cercano) cuya clase está empatada
    cand = valid & tied[rows, nc]
    has = cand.any(axis=1)
    first = np.argmax(cand, axis=1)
    pred_code = nc[np.arange(nq), first]

    empty = _default_empty(bl) if _is_auto(empty_label) else empty_label
    if _fits(empty, bl.dtype):
        out = np.full(nq, empty, dtype=bl.dtype)
    else:
        out = np.empty(nq, dtype=object)
        out[:] = [empty] * nq
    if C:
        out[has] = classes[pred_code[has]]
    return out


def vote_agreement(pred_a: Sequence[Any] | np.ndarray, pred_b: Sequence[Any] | np.ndarray) -> float:
    """Fracción de consultas en que dos predicciones coinciden (p. ej. voto aproximado vs exacto)."""
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if a.shape != b.shape:
        raise ValueError(f"formas distintas: {a.shape} y {b.shape}")
    return float(np.mean(a == b)) if a.size else float("nan")


# --------------------------------------------------------------------------
# Clasificación
# --------------------------------------------------------------------------


def _is_empty(y: np.ndarray, empty: Any) -> np.ndarray:
    if empty is None:
        return np.array([v is None for v in y], dtype=bool) if y.dtype == object else np.zeros(y.shape, bool)
    if isinstance(empty, float) and np.isnan(empty):
        return np.array([isinstance(v, float) and np.isnan(v) for v in y.tolist()], dtype=bool)
    return y == empty


def classification_report(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    empty_label: Any = "auto",
) -> dict[str, Any]:
    """Exactitud, F1-macro, exactitud balanceada y matriz de confusión.

    ``labels``: clases sobre las que se promedia el F1 y que indexan la matriz
    de confusión (por defecto, la unión ordenada de y_true y y_pred sin la
    etiqueta vacía). Las predicciones vacías cuentan como error y como falso
    negativo de su clase verdadera, pero no generan una columna.

    Devuelve un dict con ``accuracy``, ``f1_macro``, ``balanced_accuracy``,
    ``confusion`` (int64, filas = verdad, columnas = predicción),
    ``confusion_norm`` (normalizada por fila con el soporte de cada clase),
    ``labels``, ``precision``, ``recall``, ``f1``, ``support`` (por clase),
    ``n`` y ``n_empty``.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError(f"y_true e y_pred deben ser vectores del mismo largo: {yt.shape}, {yp.shape}")
    empty = _default_empty(yp) if _is_auto(empty_label) else empty_label
    is_empty = _is_empty(yp, empty)
    if labels is None:
        lab = np.unique(np.concatenate([yt, yp[~is_empty]]))
    else:
        lab = np.asarray(labels)
    L = lab.size
    pos = {v: i for i, v in enumerate(lab.tolist())}
    ti = np.array([pos.get(v, -1) for v in yt.tolist()], dtype=np.int64)
    pi = np.array([pos.get(v, -1) for v in yp.tolist()], dtype=np.int64)
    pi[is_empty] = -1

    both = (ti >= 0) & (pi >= 0)
    cm = np.zeros((L, L), dtype=np.int64)
    np.add.at(cm, (ti[both], pi[both]), 1)
    tp = np.diag(cm).astype(np.float64)
    support = np.bincount(ti[ti >= 0], minlength=L).astype(np.float64)  # todas las verdaderas de c
    pred_cnt = np.bincount(pi[pi >= 0], minlength=L).astype(np.float64)  # todas las predichas como c
    fp = pred_cnt - tp
    fn = support - tp
    precision = np.divide(tp, pred_cnt, out=np.zeros(L), where=pred_cnt > 0)
    recall = np.divide(tp, support, out=np.zeros(L), where=support > 0)
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(L), where=den > 0)
    present = support > 0

    n = yt.size
    correct = (yt == yp) & ~is_empty
    return {
        "accuracy": float(correct.mean()) if n else float("nan"),
        "f1_macro": float(f1.mean()) if L else float("nan"),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else float("nan"),
        "confusion": cm,
        "confusion_norm": np.divide(cm, support[:, None], out=np.zeros((L, L)), where=support[:, None] > 0),
        "labels": lab,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
        "n": int(n),
        "n_empty": int(is_empty.sum()),
    }


def top_confusions(confusion: np.ndarray, labels: Sequence[Any], n: int = 10, *, normalized: bool = True) -> list[tuple[Any, Any, float]]:
    """Los n pares (verdad, predicción) fuera de la diagonal con más confusión.

    Con ``normalized`` se ordena por la fracción de la fila (tasa de error del
    par); si no, por el conteo. Empates por (fila, columna) menor.
    """
    cm = np.asarray(confusion, dtype=np.float64)
    if normalized:
        rs = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    off = cm.copy()
    np.fill_diagonal(off, 0.0)
    flat = off.ravel()
    order = np.lexsort((np.arange(flat.size), -flat))
    lab = list(labels)
    out = []
    for f in order[:n]:
        if flat[f] <= 0:
            break
        i, j = divmod(int(f), cm.shape[1])
        out.append((lab[i], lab[j], float(flat[f])))
    return out
