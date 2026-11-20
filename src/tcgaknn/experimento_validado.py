"""Protocolo ANN sin fuga: tunear SOLO en validación, congelar y evaluar TEST.

Comparar siempre el MISMO ground truth exacto, la MISMA base y el MISMO
preprocesamiento. `val` nunca se indexa y `test` nunca elige hiperparámetros.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .algoritmos.base import l2_normalize
from .algoritmos.coseno.lsh import BandedLSH
from .algoritmos.coseno.simhash import SimHashKNN
from .algoritmos.coseno.simhash_rp import SimHashRPKNN
from .algoritmos.euclidiana.pca_ref import PCAKNN
from .algoritmos.euclidiana.projection import ProjectedKNN
from .algoritmos.evaluacion import classification_report, knn_vote, recall_at_k
from .algoritmos.exacto.exact import ExactKNN
from .algoritmos.lp.e2lsh import E2LSH, suggest_w
from .preparacion import load_prepared
from .splits import Split, assert_disjoint


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def configs_for(metric: str, seed: int, *, w_reference: float | None = None,
                methods: set[str] | None = None) -> list[dict]:
    """Grilla acotada para validar. Se puede extender sin tocar el conjunto test."""
    candidates: list[dict] = [{"method": "exact"}]
    if metric in {"cosine", "l2"}:
        candidates += [{"method": "rp", "m": m} for m in (128, 256)]
    if metric == "cosine":
        candidates += [{"method": "simhash", "b": b} for b in (256, 512)]
        candidates += [{"method": "simhash_rp", "b": b, "m": 256, "rerank_factor": f}
                       for b, f in ((256, 10), (512, 20))]
        candidates += [{"method": "lsh", "L": L, "r": r, "m": 256, "fallback": fallback}
                       for L, r, fallback in ((16, 8, False), (32, 8, False),
                                              (32, 6, False), (32, 8, True))]
        candidates += [{"method": "pca", "m": m} for m in (64, 128)]
    if metric in {"cosine", "l1", "l2"}:
        if w_reference is None:
            raise ValueError("E2LSH requiere w de validación")
        p = 1 if metric == "l1" else 2
        candidates += [{"method": "e2lsh", "p": p, "w": w_reference * factor, "r": r,
                        "L": L, "rerank": "proj"}
                       for factor, r, L in ((0.5, 3, 16), (1.0, 3, 16), (2.0, 3, 32))]
    if methods:
        unknown = methods - {c["method"] for c in candidates}
        if unknown:
            raise ValueError(f"Métodos no compatibles con {metric}: {sorted(unknown)}")
        candidates = [c for c in candidates if c["method"] in methods or c["method"] == "exact"]
    return [dict(c, seed=seed) for c in candidates]


def make_model(config: dict, metric: str):
    c = config
    name, seed = c["method"], c.get("seed", 2026)
    if name == "exact":
        return ExactKNN(metric, block_size=16)
    if name == "rp":
        if metric not in {"cosine", "l2"}:
            raise ValueError("RP no es un baseline L1")
        return ProjectedKNN(m=c["m"], normalize=(metric == "cosine"), seed=seed, query_block=16)
    if name == "pca":
        if metric != "cosine":
            raise ValueError("PCAKNN original normaliza filas: solo comparable con coseno")
        return PCAKNN(m=c["m"], keep_data=False, seed=seed, block=16)
    if name == "simhash":
        if metric != "cosine":
            raise ValueError("SimHash solo aproxima distancia angular/coseno")
        return SimHashKNN(b=c["b"], rerank_factor=0, seed=seed)
    if name == "simhash_rp":
        if metric != "cosine":
            raise ValueError("SimHash+RP solo aproxima coseno")
        return SimHashRPKNN(b=c["b"], m=c["m"], rerank_factor=c["rerank_factor"], seed=seed)
    if name == "lsh":
        if metric != "cosine":
            raise ValueError("LSH por bandas SimHash solo aproxima coseno")
        return BandedLSH(L=c["L"], r=c["r"], m=c["m"], seed=seed,
                         rerank="m", fallback=c.get("fallback", False))
    if name == "e2lsh":
        if metric == "l1" and c["p"] != 1 or metric != "l1" and c["p"] != 2:
            raise ValueError("E2LSH p debe corresponder a la distancia")
        return E2LSH(p=c["p"], w=c["w"], r=c["r"], L=c["L"], seed=seed,
                     rerank=c.get("rerank", "proj"))
    raise ValueError(f"Algoritmo desconocido: {name}")


def _resident_bytes(model, qstats: dict) -> int | None:
    if hasattr(model, "resident_bytes"):
        return int(model.resident_bytes())
    if hasattr(model, "index_bytes"):
        b = model.index_bytes
        if callable(b):
            b = b()
        if isinstance(b, dict):
            return int(b.get("total", b.get("total_sin_omega", 0) + b.get("omega", 0)))
        return int(b)
    if isinstance(qstats.get("index_bytes"), (int, float)):
        # Para SimHash se agrega la matriz Ω residente.
        return int(qstats["index_bytes"] + getattr(getattr(model, "Omega", None), "nbytes", 0)
                 + (0 if getattr(model, "base", None) is None else model.base.nbytes))
    return None


def _predict_metrics(y: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> dict:
    rep = classification_report(y, pred, labels=labels)
    support = rep["support"]
    # Columna adicional para abstenciones. No se silencian en la matriz.
    confusion_with_abstention = np.column_stack([
        rep["confusion"],
        [int(np.sum((y == label) & (pred == ""))) for label in labels],
    ])
    def wilson(tp: int, n: int, z: float = 1.95996398454) -> list[float]:
        if n == 0:
            return [float("nan"), float("nan")]
        phat = tp / n
        den = 1 + z*z / n
        mid = (phat + z*z / (2*n)) / den
        half = z * np.sqrt(phat * (1 - phat) / n + z*z / (4*n*n)) / den
        return [float(max(0, mid-half)), float(min(1, mid+half))]

    per_class = [{"type": str(label), "support": int(sup), "sensitivity": float(r),
                  "sensitivity_ci95_wilson": wilson(int(round(r * sup)), int(sup)),
                  "fnr": float(1 - r), "f1": float(f1)}
                 for label, sup, r, f1 in zip(labels, support, rep["recall"], rep["f1"])]
    return {"accuracy": rep["accuracy"], "f1_macro": rep["f1_macro"],
            "balanced_accuracy": rep["balanced_accuracy"], "n_no_prediction": rep["n_empty"],
            "coverage": 1 - rep["n_empty"] / len(y), "n": int(len(y)), "per_class": per_class,
            "confusion": rep["confusion"].tolist(),
            "confusion_with_abstention": confusion_with_abstention.tolist(),
            "confusion_columns": [str(x) for x in labels] + ["SIN_PREDICCION"]}


def evaluate(model, Xtrain: np.ndarray, Xq: np.ndarray, ytrain: np.ndarray,
             yq: np.ndarray, truth: np.ndarray, labels: np.ndarray, *, k: int,
             exact_pred: np.ndarray | None = None, seed: int = 2026) -> dict:
    t0 = time.perf_counter()
    model.fit(Xtrain)
    build_seconds = time.perf_counter() - t0
    for i in range(min(2, len(Xq))):
        model.query(Xq[i:i + 1], k)  # calentamiento fuera de la medición
    ids = np.full((len(Xq), k), -1, dtype=np.int64)
    lat = []
    candidates = []
    full_scans = []
    fallback_counts = []
    last_stats: dict = {}
    for i in range(len(Xq)):
        t0 = time.perf_counter()
        idx, _, stats = model.query(Xq[i:i + 1], k)  # incluye firma/proyección de la consulta
        lat.append((time.perf_counter() - t0) * 1000)
        ids[i] = idx[0]
        cand = stats.get("n_candidates", np.array([len(Xtrain)]))
        candidates.append(int(np.asarray(cand).ravel()[0]))
        fallback_count = int(np.asarray(stats.get("n_fallback", [0])).ravel()[0])
        fallback_counts.append(fallback_count)
        full_scans.append(bool(getattr(model, "fallback", False) and fallback_count > 0))
        last_stats = stats
    y_pred = knn_vote(ids, ytrain)
    out = _predict_metrics(yq, y_pred, labels)
    if exact_pred is not None:
        # Bootstrap emparejado: diferencia de exactitud aproximado - EX sobre
        # las mismas consultas, no independiente ni sobre muestras distintas.
        difference = (y_pred == yq).astype(np.float64) - (exact_pred == yq).astype(np.float64)
        rng = np.random.default_rng(seed + 1049)
        draws = rng.integers(0, len(yq), size=(500, len(yq)))
        d = difference[draws].mean(axis=1)
        out["delta_accuracy_ci95_bootstrap"] = np.quantile(d, [0.025, 0.975]).tolist()
    out.update({
        "recall_at_k": float(recall_at_k(ids, truth, k)),
        "fraction_candidates": float(np.mean(candidates) / len(Xtrain)),
        "mean_candidates": float(np.mean(candidates)),
        "mean_fallback_extra": float(np.mean(fallback_counts)),
        "fraction_fallback_full_hamming_scan": float(np.mean(full_scans)),
        "fraction_with_less_than_k": float(np.mean((ids >= 0).sum(axis=1) < k)),
        "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
        "build_seconds": build_seconds, "resident_bytes_estimate": _resident_bytes(model, last_stats),
    })
    return out


def _data(data_dir: Path, metric: str, *, phase: str, verify_hash: bool = False):
    X, npz, meta = load_prepared(data_dir, verify_hash=verify_hash)
    parts = Split(*(npz[k] for k in ("train", "val", "test")))
    assert_disjoint(parts, len(npz["labels"]))
    target = parts.val if phase == "val" else parts.test
    # Materializar solo las filas necesarias en RAM; cada paciente = una fila.
    A = np.asarray(X[parts.train], dtype=np.float32)
    Q = np.asarray(X[target], dtype=np.float32)
    if metric == "cosine":
        # No pierde información extra respecto a coseno exacto; todas las
        # aproximaciones se evalúan respecto del MISMO espacio unitario.
        A, Q = l2_normalize(A), l2_normalize(Q)
    if not np.isfinite(A).all() or not np.isfinite(Q).all():
        raise ValueError("Preprocesamiento dejó valores no finitos")
    if np.any(np.linalg.norm(A, axis=1) < 1e-9) or np.any(np.linalg.norm(Q, axis=1) < 1e-9):
        raise ValueError("Hay muestras de expresión sin señal; revisar criterio de exclusión")
    y = npz["labels"]
    labels = np.unique(y)
    return A, Q, y[parts.train], y[target], labels, meta


def validate(data_dir: Path | str, out: Path | str, *, metric: str = "cosine", k: int = 10,
             methods: set[str] | None = None, seed: int = 2026, verify_hash: bool = False) -> dict:
    data_dir, out = Path(data_dir), Path(out)
    A, Q, ya, yq, labels, meta = _data(data_dir, metric, phase="val", verify_hash=verify_hash)
    if len(A) < k:
        raise ValueError("La base de training tiene menos de k pacientes")
    exact = ExactKNN(metric, block_size=16).fit(A)
    truth, truth_scores, _ = exact.query(Q, k)
    exact_pred = knn_vote(truth, ya)
    exact_class = _predict_metrics(yq, exact_pred, labels)
    if metric == "cosine":
        kth_dist = np.sqrt(np.maximum(0, 2 - 2 * truth_scores[:, -1]))
    else:
        kth_dist = truth_scores[:, -1]
    w_ref = suggest_w(kth_dist, factor=1.0)
    configs = configs_for(metric, seed, w_reference=w_ref, methods=methods)
    results = []
    for config in configs:
        name = config["method"]
        print(f"[validación] {config}", flush=True)
        try:
            model = make_model(config, metric)
            scores = evaluate(model, A, Q, ya, yq, truth, labels, k=k,
                              exact_pred=exact_pred, seed=seed)
            scores["delta_accuracy_vs_exact"] = scores["accuracy"] - exact_class["accuracy"]
            scores["delta_f1_vs_exact"] = scores["f1_macro"] - exact_class["f1_macro"]
            # Evitar ocultar pérdidas importantes en clases con soporte suficiente.
            # Para clases con <10 consultas, reportar IC, no imponer umbrales inestables.
            ref_recall = {r["type"]: r["sensitivity"] for r in exact_class["per_class"]}
            drops = [ref_recall[r["type"]] - r["sensitivity"]
                     for r in scores["per_class"] if r["support"] >= 10]
            scores["max_class_sensitivity_loss_support_ge_10"] = max(drops, default=None)
            scores["meets_quality_targets"] = (scores["recall_at_k"] >= 0.90
                and scores["delta_accuracy_vs_exact"] >= -0.02
                and scores["delta_f1_vs_exact"] >= -0.03
                and (not drops or max(drops) <= 0.05))
            results.append({"config": config, "metrics": scores})
            del model
            gc.collect()
        except (MemoryError, ValueError, RuntimeError) as exc:
            results.append({"config": config, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {exc}", flush=True)
    # Elegir una configuración por algoritmo solo con VAL. Si ninguna cumple
    # las metas, reportar explicitamente que no se alcanzaron.
    selected = []
    for method in sorted({r["config"]["method"] for r in results}):
        valid = [r for r in results if r["config"]["method"] == method and "metrics" in r]
        if not valid:
            continue
        qualified = [r for r in valid if r["metrics"]["meets_quality_targets"]]
        pool = qualified if qualified else valid
        best = sorted(pool, key=lambda r: (-r["metrics"]["recall_at_k"],
                                            r["metrics"]["latency_ms_median"]))[0]
        selected.append({"config": best["config"], "qualified": bool(qualified),
                         "validation_recall_at_k": best["metrics"]["recall_at_k"]})
    pack = {"phase": "validation", "dataset_id": meta["dataset_id"], "metric": metric,
            "k": k, "seed": seed, "exact_classification": exact_class, "results": results}
    _save(out / "validation.json", pack)
    freeze = {"phase": "frozen_after_validation", "dataset_id": meta["dataset_id"],
              "metric": metric, "k": k, "seed": seed, "selected": selected,
              "quality_thresholds": {"recall_at_k": 0.90, "max_accuracy_loss": 0.02,
                                     "max_f1_loss": 0.03, "max_class_sensitivity_loss_support_ge_10": 0.05}}
    _save(out / "configuraciones_congeladas.json", freeze)
    return pack


def test(data_dir: Path | str, selection_path: Path | str, out: Path | str,
         *, verify_hash: bool = False) -> dict:
    data_dir, out = Path(data_dir), Path(out)
    saved = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if saved.get("phase") != "frozen_after_validation":
        raise ValueError("Se requiere configuraciones_congeladas.json generado en validación")
    metric, k = saved["metric"], int(saved["k"])
    A, Q, ya, yq, labels, meta = _data(data_dir, metric, phase="test", verify_hash=verify_hash)
    if meta["dataset_id"] != saved["dataset_id"]:
        raise ValueError("La selección corresponde a otra cohorte, partición o preprocesamiento")
    exact = ExactKNN(metric, block_size=16).fit(A)
    truth, _, _ = exact.query(Q, k)
    ex_class = _predict_metrics(yq, knn_vote(truth, ya), labels)
    results = []
    for entry in saved["selected"]:
        conf = entry["config"]
        print(f"[TEST final] {conf}", flush=True)
        model = make_model(conf, metric)
        exact_pred = knn_vote(truth, ya)
        metrics = evaluate(model, A, Q, ya, yq, truth, labels, k=k,
                           exact_pred=exact_pred, seed=saved["seed"])
        metrics["delta_accuracy_vs_exact"] = metrics["accuracy"] - ex_class["accuracy"]
        metrics["delta_f1_vs_exact"] = metrics["f1_macro"] - ex_class["f1_macro"]
        results.append({"config": conf, "qualified_on_validation": entry["qualified"], "metrics": metrics})
        del model
        gc.collect()
    report = {"phase": "test_final", "dataset_id": meta["dataset_id"], "metric": metric,
              "k": k, "exact_classification": ex_class, "results": results}
    _save(out / "test_final.json", report)
    return report
