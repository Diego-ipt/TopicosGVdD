from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from tcgaknn.algoritmos.evaluacion import (
    approx_ratio,
    classification_report,
    knn_vote,
    label_purity,
    precision_at_k,
    recall_at_k,
    top_confusions,
    vote_agreement,
)

# --------------------------------------------------------------------------
# recall / precision
# --------------------------------------------------------------------------


def test_recall_counts_padding_as_miss() -> None:
    E = np.array([[1, 2, 3], [4, 5, 6]])
    A = np.array([[1, -1, -1], [6, 5, 4]])
    np.testing.assert_allclose(recall_at_k(A, E, 3, per_query=True), [1 / 3, 1.0])
    assert recall_at_k(A, E, 3) == pytest.approx(2 / 3)
    # precision solo sobre lo devuelto
    np.testing.assert_allclose(precision_at_k(A, E, 3, per_query=True), [1.0, 1.0])


def test_recall_duplicates_and_k_truncation() -> None:
    E = np.array([[1, 2, 3, 9]])
    A = np.array([[1, 1, 1, 2]])  # repetidos cuentan una vez; solo se miran los primeros k
    assert recall_at_k(A, E, 3) == pytest.approx(1 / 3)
    assert recall_at_k(A, E, 4) == pytest.approx(2 / 4)
    # k mayor que las columnas: se rellena con -1; base con < k filas: |E| < k
    assert recall_at_k(np.array([[0, 1]]), np.array([[1, 0]]), 5) == 1.0
    assert recall_at_k(np.array([[-1, -1]]), np.array([[-1, -1]]), 2) == 0.0
    assert np.isnan(recall_at_k(np.zeros((0, 3), int), np.zeros((0, 3), int), 3))
    with pytest.raises(TypeError):
        recall_at_k(np.array([[0.0]]), np.array([[0]]), 1)
    with pytest.raises(ValueError):
        recall_at_k(np.array([[0]]), np.array([[0], [1]]), 1)


def test_precision_nan_without_results() -> None:
    p = precision_at_k(np.array([[-1, -1], [0, 3]]), np.array([[0, 1], [0, 1]]), 2, per_query=True)
    assert np.isnan(p[0]) and p[1] == 0.5
    assert precision_at_k(np.array([[-1, -1], [0, 3]]), np.array([[0, 1], [0, 1]]), 2) == 0.5


def test_approx_ratio() -> None:
    E = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, np.nan]])
    A = np.array([[2.0, 2.0], [0.0, 2.0], [np.nan, np.nan]])
    r = approx_ratio(A, E, per_query=True)
    np.testing.assert_allclose(r[:2], [1.5, 1.5])
    assert np.isnan(r[2])
    assert approx_ratio(A, E) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        approx_ratio(A[:, :1], E)


# --------------------------------------------------------------------------
# label_purity
# --------------------------------------------------------------------------


def test_label_purity() -> None:
    base = np.array(["a", "b", "a", "c"])
    N = np.array([[0, 2, 1], [1, -1, -1], [-1, -1, -1]])
    q = np.array(["a", "a", "c"])
    p = label_purity(N, base, q, per_query=True)
    np.testing.assert_allclose(p[:2], [2 / 3, 0.0])
    assert np.isnan(p[2])
    pm = label_purity(N, base, q, count_missing=True, per_query=True)
    np.testing.assert_allclose(pm, [2 / 3, 0.0, 0.0])
    assert label_purity(N, base, q) == pytest.approx(1 / 3)
    # -1 no debe leer la última etiqueta (base[-1] = "c" coincidiría con q[2])
    assert label_purity(N, base, q, count_missing=True) == pytest.approx(2 / 9)


# --------------------------------------------------------------------------
# Voto
# --------------------------------------------------------------------------


def test_vote_majority_and_tiebreak_nearest() -> None:
    base = np.array(["a", "b", "c", "b", "a"])
    N = np.array(
        [
            [2, 1, 3, -1],  # b gana 2-1
            [0, 1, 4, 3],  # a 2, b 2: empate, gana a (más cercano)
            [1, 0, 3, 4],  # empate a-b, gana b (más cercano)
            [2, 0, 1, -1],  # triple empate: c
            [-1, -1, -1, -1],  # sin vecinos
        ]
    )
    pred = knn_vote(N, base)
    assert pred.tolist() == ["b", "a", "b", "c", ""]


def test_vote_tiebreak_ignores_non_tied_nearest() -> None:
    # El más cercano es c (1 voto) pero el máximo lo empatan a y b: gana b,
    # el más cercano ENTRE los empatados.
    base = np.array(["a", "b", "c"])
    N = np.array([[2, 1, 0, 1, 0]])
    assert knn_vote(N, base).tolist() == ["b"]


def test_vote_weighted() -> None:
    base = np.array([0, 1, 1])
    N = np.array([[0, 1, 2], [0, 1, 2]])
    W = np.array([[0.9, 0.4, 0.4], [0.8, 0.4, 0.4]])
    assert knn_vote(N, base, W).tolist() == [0, 0]  # 0.9 > 0.8 ; 0.8 = 0.8 → más cercano
    W2 = np.array([[0.5, 0.4, 0.4], [np.nan, 0.4, 0.1]])
    assert knn_vote(N, base, W2).tolist() == [1, 1]
    with pytest.raises(ValueError):
        knn_vote(N, base, W[:, :2])


def test_vote_empty_labels_by_dtype() -> None:
    N = np.array([[-1], [0]])
    assert knn_vote(N, np.array([3])).tolist() == [-1, 3]
    assert knn_vote(N, np.array(["x"])).tolist() == ["", "x"]
    out = knn_vote(N, np.array(["x"]), empty_label=None)
    assert out.dtype == object and out.tolist() == [None, "x"]


def test_vote_agreement() -> None:
    assert vote_agreement(["a", "b", "c", ""], ["a", "c", "c", ""]) == 0.75
    assert np.isnan(vote_agreement([], []))
    with pytest.raises(ValueError):
        vote_agreement(["a"], ["a", "b"])


# --------------------------------------------------------------------------
# Clasificación contra scikit-learn
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classification_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    yt = rng.choice(list("abcd"), 300)
    yp = np.where(rng.random(300) < 0.6, yt, rng.choice(list("abcde"), 300))
    r = classification_report(yt, yp)
    assert r["accuracy"] == pytest.approx(accuracy_score(yt, yp))
    assert r["f1_macro"] == pytest.approx(f1_score(yt, yp, average="macro", zero_division=0))
    with pytest.warns(UserWarning):  # 'e' predicha pero ausente en y_true
        ba = balanced_accuracy_score(yt, yp)
    assert r["balanced_accuracy"] == pytest.approx(ba)
    np.testing.assert_array_equal(r["confusion"], confusion_matrix(yt, yp, labels=r["labels"]))
    assert r["labels"].tolist() == list("abcde") and r["n_empty"] == 0


def test_classification_empty_predictions_are_errors() -> None:
    yt = np.array(["a", "a", "b", "b", "c"])
    yp = np.array(["a", "", "b", "a", ""])
    r = classification_report(yt, yp)
    labs = ["a", "b", "c"]
    assert r["labels"].tolist() == labs and r["n_empty"] == 2
    assert r["accuracy"] == pytest.approx(0.4)
    assert r["f1_macro"] == pytest.approx(f1_score(yt, yp, labels=labs, average="macro", zero_division=0))
    assert r["balanced_accuracy"] == pytest.approx(recall_score(yt, yp, labels=labs, average="macro", zero_division=0))
    assert r["confusion"].sum() == 3
    np.testing.assert_array_equal(r["support"], [2, 2, 1])


def test_classification_int_labels_and_top_confusions() -> None:
    yt = np.array([0, 0, 0, 1, 1, 2])
    yp = np.array([1, 1, 0, 1, -1, 0])
    r = classification_report(yt, yp)
    assert r["labels"].tolist() == [0, 1, 2] and r["n_empty"] == 1
    assert r["accuracy"] == pytest.approx(2 / 6)
    tc = top_confusions(r["confusion"], r["labels"], 2)
    assert tc[0] == (2, 0, 1.0) and tc[1][:2] == (0, 1)
    assert top_confusions(r["confusion"], r["labels"], 5, normalized=False)[0] == (0, 1, 2.0)
    with pytest.raises(ValueError):
        classification_report([0, 1], [0])
