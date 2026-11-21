"""Un único protocolo por n: validar -> seleccionar -> congelar -> test UNA vez.

LSH/coseno y E2LSH-p2 comparan con exacto-coseno sobre Z/||Z||2.
E2LSH-p1 compara con exacto-L1 sobre Z (z-scores sin normalizar filas).
Nunca se ajusta w ni se eligen hiperparámetros usando test.
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from .algoritmos.base import topk
from .algoritmos.coseno.lsh import BandedLSH, s_curve
from .algoritmos.evaluacion import classification_report, knn_vote, recall_at_k
from .algoritmos.lp.e2lsh import E2LSH, candidate_probability, suggest_w
from .preparacion_streaming import load_sketches, peak_rss_mib
from .splits import Split, assert_disjoint

METHODS = frozenset({"exact", "exact_l1", "lsh", "e2lsh_p1", "e2lsh_p2"})
# Las columnas guardadas se reparten entre réplicas, sin desperdiciar el resto.
COS_GRID = ((8, 4), (16, 4), (16, 6), (32, 6))
LP_GRID = ((8, 2), (16, 2), (16, 4), (32, 4), (16, 8), (32, 8))
W_FACTORS = (0.25, 0.5, 0.75, 1.0)
MAX_CANDIDATE_FRACTION = 0.50
MIN_COVERAGE = 0.90
# Política preregistrada antes de la siguiente corrida real. NO ajustar con test.
# Los valores originales 0.02/0.03/0.05 y recall >= 0.90 se recuperan
# explícitamente. Si no son factibles, registrar el incumplimiento.
QUALITY_POLICY = {
    "max_accuracy_loss": 0.02,
    "max_f1_loss": 0.03,
    "max_class_sensitivity_loss_support_ge_10": 0.05,
    "min_recall_at_k": 0.90,
    "min_coverage": MIN_COVERAGE,
    "max_candidate_fraction": MAX_CANDIDATE_FRACTION,
}
FALLBACK_ACCURACY_TOLERANCE = 0.02  # solo validación; no convierte un fracaso en éxito



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


def annotate_quality(scores: dict, exact_class: dict) -> None:
    """Criterios preregistrados: no se modifican usando test ni para rescatar resultados."""
    scores["delta_accuracy_vs_exact"] = scores["accuracy"] - exact_class["accuracy"]
    scores["delta_f1_vs_exact"] = scores["f1_macro"] - exact_class["f1_macro"]
    ref_recall = {r["type"]: r["sensitivity"] for r in exact_class["per_class"]}
    drops = [ref_recall[r["type"]] - r["sensitivity"]
             for r in scores["per_class"] if r["support"] >= 10]
    scores["max_class_sensitivity_loss_support_ge_10"] = max(drops, default=None)
    scores["meets_quality_targets"] = (
        scores["delta_accuracy_vs_exact"] >= -QUALITY_POLICY["max_accuracy_loss"]
        and scores["delta_f1_vs_exact"] >= -QUALITY_POLICY["max_f1_loss"]
        and scores["coverage"] >= QUALITY_POLICY["min_coverage"]
        and scores["recall_at_k"] >= QUALITY_POLICY["min_recall_at_k"]
        and (not drops or max(drops) <= QUALITY_POLICY["max_class_sensitivity_loss_support_ge_10"])
    )
    # Distintos objetivos: calidad predictiva y reducción del conjunto candidato.
    scores["meets_candidate_budget"] = scores["fraction_candidates"] <= QUALITY_POLICY["max_candidate_fraction"]
    scores["meets_efficiency_and_quality"] = (
        scores["meets_quality_targets"] and scores["meets_candidate_budget"])


def select_frozen(results: list[dict]) -> list[dict]:
    """Congela en validación priorizando calidad y fracción real de candidatos.

    1) Calidad + <=50% de candidatos: elegir MENOS candidatos (latencia desempata).
    2) Si no, elegir una configuración DIAGNÓSTICA, marcada como no calificada,
       minimizando candidatos entre alternativas de validación casi tan precisas
       como la mejor (tolerancia preregistrada 0.02). Nunca ajustar tras test.
    La razón y la ausencia de configuración que cumpla los objetivos se guardan.
    """
    selected = []
    for method, replica in sorted({(r["config"]["method"], r["config"].get("replica", 0))
                                    for r in results}):
        valid = [r for r in results if (r["config"]["method"], r["config"].get("replica", 0)) ==
                 (method, replica) and "metrics" in r]
        if not valid:
            continue
        if method.startswith("exact"):
            baseline = valid[0]
            selected.append({"config": baseline["config"], "qualified": True,
                             "selection_reason": "exact_reference_no_candidate_budget",
                             "validation_accuracy": baseline["metrics"]["accuracy"],
                             "validation_vote_concordance": 1.0,
                             "validation_fraction_candidates": 1.0,
                             "validation_recall_at_k": 1.0})
            continue
        qualified = [r for r in valid if r["metrics"]["meets_efficiency_and_quality"]]
        if qualified:
            pool, reason = qualified, "quality_and_candidate_budget"
        else:
            good_quality = [r for r in valid if r["metrics"]["meets_quality_targets"]]
            if good_quality:
                pool, reason = good_quality, "quality_only_budget_not_met"
            else:
                best_accuracy = max(r["metrics"]["accuracy"] for r in valid)
                best_coverage = max(r["metrics"]["coverage"] for r in valid)
                near_best = [r for r in valid
                             if r["metrics"]["accuracy"] >= best_accuracy - FALLBACK_ACCURACY_TOLERANCE
                             and r["metrics"]["coverage"] >= max(0.5, best_coverage - 0.05)]
                pool = near_best or valid
                reason = "diagnostic_unqualified_near_best_accuracy_then_candidates"
        def key(row):
            m = row["metrics"]
            latency = m.get("latency_ms_median")
            return (m["fraction_candidates"], float("inf") if latency is None else latency,
                    -m["accuracy"], -m.get("vote_concordance_vs_exact", 0.0))
        best = min(pool, key=key)
        selected.append({"config": best["config"], "qualified": bool(qualified),
                         "selection_reason": reason,
                         "validation_accuracy": best["metrics"]["accuracy"],
                         "validation_vote_concordance": best["metrics"].get("vote_concordance_vs_exact"),
                         "validation_fraction_candidates": best["metrics"]["fraction_candidates"],
                         "validation_recall_at_k": best["metrics"]["recall_at_k"]})
    return selected

def _sizes(base_size: int, k: int, requested: list[int] | None) -> list[int]:
    if base_size < k or k < 1:
        raise ValueError("Se requiere 1 <= k <= número de pacientes train")
    if requested is None:
        requested = [n for n in (50, 100, 250, 500, 1000, 2000, 4000) if n <= base_size]
        requested.append(base_size)
    elif any(n < k or n > base_size for n in requested):
        raise ValueError(f"Todos los tamaños deben estar en [{k}, {base_size}]")
    valid = sorted({int(n) for n in requested if k <= n <= base_size})
    if not valid:
        raise ValueError("La grilla de tamaños no contiene bases posibles")
    return valid


def _phase(path: Path, phase: str, verify_hash: bool):
    Y, Y2, Y1, sq, G, L1, spl, meta = load_sketches(path, phase=phase, verify_hash=verify_hash)
    parts = Split(*(spl[key] for key in ("train", "val", "test")))
    assert_disjoint(parts, len(spl["labels"]))
    order = spl["base_order"]
    queries = spl[f"query_{phase}"]
    if not set(order.tolist()) == set(parts.train.tolist()):
        raise ValueError("base_order debe ser una permutación de train")
    if not set(queries.tolist()).issubset(set((parts.val if phase == "val" else parts.test).tolist())):
        raise ValueError("Hay consultas fuera de la fase solicitada")
    if G.shape != (len(queries), len(order)) or L1.shape != G.shape:
        raise ValueError("Dimensiones incorrectas de Gram/L1")
    labels = spl["labels"]
    if np.any(sq[np.concatenate((order, queries))] <= 1e-12):
        raise ValueError("Pacientes con norma cero; revisar metadata['zero_norm_patients']")
    return dict(Y=Y, Y2=Y2, Y1=Y1, sq=sq, G=G, L1=L1, order=order,
                query=queries, labels=labels, classes=np.unique(labels), meta=meta)


def _truth(data: dict, n: int, k: int, metric: str):
    """Exacto en datos respaldados en disco, con latencias por consulta medidas.

    Estos tiempos son EXCLUSIVAMENTE de recuperación top-k a partir de scores
    exactos precomputados; NO miden una búsqueda exacta sobre genes de novo.
    """
    raw = data["G"] if metric == "cosine" else data["L1"]
    nq = len(data["query"])
    result = np.full((nq, k), -1, dtype=np.int64)
    kth = np.full(nq, np.nan, dtype=np.float64)
    latency = np.empty(nq, dtype=np.float64)
    base_norm = np.sqrt(data["sq"][data["order"][:n]]) if metric == "cosine" else None
    for i in range(nq):
        start = time.perf_counter()
        row = np.asarray(raw[i, :n], dtype=np.float64)
        if metric == "cosine":
            den = np.sqrt(data["sq"][data["query"][i]]) * base_norm
            row = np.divide(row, den, out=np.zeros_like(row), where=den > 1e-24)
        ids, values = topk(row[None, :], k, largest=(metric == "cosine"))
        latency[i] = (time.perf_counter() - start) * 1e3
        result[i] = ids[0]
        if metric == "cosine":
            kth[i] = np.sqrt(max(0, 2 - 2 * np.clip(values[0, k-1], -1, 1)))
        else:
            kth[i] = values[0, k-1]
    timing = {"median_ms": float(np.median(latency)),
              "p95_ms": float(np.percentile(latency, 95)),
              "total_seconds": float(np.sum(latency) / 1000),
              "stage": "cached_exact_scores_topk_only"}
    return result, kth, timing

def _configs(seed: int, cols: int, lp_cols: int, w: dict[str, float],
             methods: set[str] | None = None, replicas: int = 1) -> list[dict]:
    """Grilla por n; cada réplica usa TODO su bloque disjunto de columnas."""
    methods = METHODS if methods is None else set(methods) | {"exact", "exact_l1"}
    invalid = methods - METHODS
    if invalid:
        raise ValueError(f"Métodos no soportados por el streaming: {sorted(invalid)}")
    if replicas < 1:
        raise ValueError("replicas debe ser >= 1")
    cos_block, lp_block = cols // replicas, lp_cols // replicas
    if "lsh" in methods and cos_block < min(L*r for L, r in COS_GRID):
        raise ValueError("No hay columnas coseno suficientes por réplica para LSH")
    if ({"e2lsh_p1", "e2lsh_p2"} & methods) and lp_block < min(L*r for L, r in LP_GRID):
        raise ValueError("No hay columnas E2LSH suficientes por réplica")
    out = [{"method": "exact", "seed": seed}, {"method": "exact_l1", "seed": seed}]
    for replica in range(replicas):
        conf_seed = seed + replica * 10007
        if "lsh" in methods:
            for L, r in COS_GRID:
                if L * r <= cos_block:
                    out.append({"method": "lsh", "L": L, "r": r, "m": cos_block,
                                "fallback": False, "seed": conf_seed, "replica": replica})
        for p in (1, 2):
            method = f"e2lsh_p{p}"
            if method not in methods:
                continue
            if method not in w:
                raise ValueError(f"Falta calibrar el ancho w para {method}")
            for L, r in LP_GRID:
                if L * r > lp_block:
                    continue
                for factor in W_FACTORS:
                    out.append({"method": method, "p": p, "L": L, "r": r,
                                "w": w[method] * factor, "w_factor": factor,
                                "rerank": "proj", "rerank_cols": lp_block,
                                "seed": conf_seed, "replica": replica})
    return out

def _normalized_projected(data: dict, indices: np.ndarray, m: int, replica: int,
                          replicas: int) -> np.ndarray:
    block = data["Y"].shape[1] // replicas
    offset = replica * block
    P = data["Y"][indices, offset:offset+m] / np.sqrt(m)
    return (P / np.sqrt(data["sq"][indices, None])).astype(np.float32)


def _stable_projected(data: dict, indices: np.ndarray, p: int, n_funcs: int,
                      replica: int, replicas: int) -> np.ndarray:
    block = data["Y1" if p == 1 else "Y2"].shape[1] // replicas
    offset = replica * block
    Y = data["Y1" if p == 1 else "Y2"][indices, offset:offset+n_funcs]
    return (Y if p == 1 else Y / np.sqrt(data["sq"][indices, None])).astype(np.float64)


def _build(config: dict, data: dict, n: int, replicas: int, *, with_index: bool = False):
    """Índice sobre B_n; devuelve query(i), bytes residentes, build_seconds."""
    c = config
    base = data["order"][:n]
    q = data["query"]
    replica = c.get("replica", 0)
    t0 = time.perf_counter()
    if c["method"] == "lsh":
        nbits = c["L"] * c["r"]
        offset = replica * (data["Y"].shape[1] // replicas)
        B = data["Y"][base, offset:offset+nbits] >= 0
        Q = data["Y"][q, offset:offset+nbits] >= 0
        P = _normalized_projected(data, base, c["m"], replica, replicas)
        Pq = _normalized_projected(data, q, c["m"], replica, replicas)
        idx = BandedLSH(L=c["L"], r=c["r"], m=c["m"], seed=c["seed"],
                        rerank="m", fallback=False).fit_bits(B, rerank_data=P)
        seconds = time.perf_counter()-t0
        result = ((lambda i, k: idx.query_bits(Q[i:i+1], k, rerank_data=Pq[i:i+1])),
                  int(idx.index_bytes()["total_sin_omega"]), seconds)
        return (*result, idx) if with_index else result
    p = c["p"]
    n_funcs = c["L"] * c["r"]
    Y = _stable_projected(data, base, p, n_funcs, replica, replicas)
    Q = _stable_projected(data, q, p, n_funcs, replica, replicas)
    Yr = _stable_projected(data, base, p, c["rerank_cols"], replica, replicas)
    Qr = _stable_projected(data, q, p, c["rerank_cols"], replica, replicas)
    idx = E2LSH(p=p, w=c["w"], r=c["r"], L=c["L"], seed=c["seed"],
                rerank=c["rerank"]).fit_projections(Y, rerank_projections=Yr)
    seconds = time.perf_counter()-t0
    result = ((lambda i, k: idx.query_projections(Q[i:i+1], k,
               rerank_query_projections=Qr[i:i+1])), idx.index_bytes(), seconds)
    return (*result, idx) if with_index else result

def _timed_queries(query_fn, n_queries: int, k: int, warmup: int = 3):
    """Portado de main: latencia individual del índice (sketch de la consulta ya disponible)."""
    for i in range(min(warmup, n_queries)):
        query_fn(i, k)
    ids = np.full((n_queries, k), -1, dtype=np.int64)
    candidates = np.empty(n_queries, dtype=np.int64)
    milliseconds = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        t0 = time.perf_counter()
        idx, _, stats = query_fn(i, k)
        milliseconds[i] = (time.perf_counter()-t0) * 1e3
        ids[i] = idx[0]
        candidates[i] = int(np.asarray(stats.get("n_candidates", [-1])).ravel()[0])
    return ids, candidates, milliseconds


def _exact_metrics(truth: np.ndarray, base_labels: np.ndarray, query_labels: np.ndarray,
                   classes: np.ndarray, n: int, metric: str, data: dict,
                   timing: dict) -> dict:
    pred = knn_vote(truth, base_labels)
    out = _predict_metrics(query_labels, pred, classes)
    out.update({"recall_at_k": 1.0, "vote_concordance_vs_exact": 1.0,
                "fraction_candidates": 1.0, "mean_candidates": float(n),
                "fraction_with_less_than_k": 0.0,
                "build_seconds": 0.0,
                "exact_shared_stream_pass_seconds": data["meta"]["streaming_run"]["pass_seconds"],
                "latency_ms_median": timing["median_ms"],
                "latency_ms_p95": timing["p95_ms"],
                "latency_stage": timing["stage"],
                "resident_bytes_estimate": 0,
                "ground_truth_disk_bytes_for_n": int(len(data["query"]) * n * 8),
                "process_peak_rss_mib_cumulative": peak_rss_mib(),
                "note": "EXACTO: lectura de distancias precomputadas y top-k, NO búsqueda de novo sobre genes.",
                "delta_accuracy_vs_exact": 0.0, "delta_f1_vs_exact": 0.0,
                "max_class_sensitivity_loss_support_ge_10": 0.0,
                "meets_quality_targets": True, "meets_candidate_budget": False,
                "meets_efficiency_and_quality": False})
    return out


def _evaluate(config: dict, data: dict, n: int, k: int, truth: np.ndarray,
              exact_cls: dict, near_dist: np.ndarray, exact_pred: np.ndarray,
              exact_timing: dict, replicas: int):
    method = config["method"]
    base = data["order"][:n]
    ya, yq = data["labels"][base], data["labels"][data["query"]]
    if method.startswith("exact"):
        return _exact_metrics(truth, ya, yq, data["classes"], n,
                              "l1" if method == "exact_l1" else "cosine", data,
                              exact_timing)
    query_fn, resident, build = _build(config, data, n, replicas)
    approx, cands, lat = _timed_queries(query_fn, len(yq), k)
    pred = knn_vote(approx, ya)
    scores = _predict_metrics(yq, pred, data["classes"])
    scores.update({
        "recall_at_k": float(recall_at_k(approx, truth, k)),
        "vote_concordance_vs_exact": float(np.mean(pred == exact_pred)),
        "fraction_candidates": float(cands.mean() / n),
        "mean_candidates": float(cands.mean()),
        "fraction_with_less_than_k": float(np.mean(np.sum(approx >= 0, axis=1) < k)),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_stage": "sketch_precomputed_index_lookup_and_rerank",
        "build_seconds": float(build), "resident_bytes_estimate": int(resident),
        "process_peak_rss_mib_cumulative": peak_rss_mib(),
        "query_projection_precomputed_in_stream": True,
        "rerank_cols_used": int(config.get("m", config.get("rerank_cols", 0))),
        "hash_cols_used": int(config["L"] * config["r"]),
    })
    if method == "lsh":
        theta = np.arccos(np.clip(1-near_dist**2 / 2, -1, 1))
        scores["candidate_probability_theory_near_k"] = float(np.mean(s_curve(theta, config["L"], config["r"])))
    else:
        scores["candidate_probability_theory_near_k"] = float(np.mean(candidate_probability(
            near_dist, config["w"], config["p"], config["r"], config["L"])))
    # Los tiempos cached-exact son una referencia conservadora de extracción,
    # no un speedup de búsqueda vectorial completa (ambos precálculos son diferentes).
    scores["latency_ratio_vs_cached_exact"] = (
        float(exact_timing["median_ms"] / scores["latency_ms_median"])
        if scores["latency_ms_median"] > 0 else None)
    annotate_quality(scores, exact_cls)
    return scores

def _truths(data: dict, n: int, k: int):
    cosine = _truth(data, n, k, "cosine")
    l1 = _truth(data, n, k, "l1")
    return {"cosine": cosine, "l1": l1}


def _write_csv(path: Path, report: dict) -> None:
    """CSV plano para tablas/gráficas, una fila por método, n y réplica."""
    headers = ["phase", "n", "method", "metric", "replica", "k", "n_queries",
               "L", "r", "w", "w_factor", "m", "hash_cols_used", "rerank_cols_used",
               "accuracy", "balanced_accuracy", "f1_macro", "vote_concordance_vs_exact",
               "recall_at_k", "coverage", "fraction_candidates", "mean_candidates",
               "fraction_with_less_than_k", "delta_accuracy_vs_exact", "delta_f1_vs_exact",
               "meets_quality_targets", "meets_candidate_budget", "meets_efficiency_and_quality",
               "build_seconds", "latency_ms_median", "latency_ms_p95",
               "latency_stage", "latency_ratio_vs_cached_exact", "resident_bytes_estimate",
               "exact_stream_ms_per_query_amortized", "amortized_ratio_vs_exact_stream",
               "process_peak_rss_mib_cumulative", "qualified_on_validation",
               "selection_reason", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for step in report["results"]:
            for row in step["results"]:
                cfg, met = row["config"], row.get("metrics", {})
                obj = {"phase": report["phase"], "n": step["n"], "metric": row["metric"],
                       "k": report["k"], "n_queries": report["n_queries"],
                       "qualified_on_validation": row.get("qualified_on_validation"),
                       "selection_reason": row.get("selection_reason"),
                       "error": row.get("error")}
                obj.update({key: cfg.get(key) for key in ("method", "replica", "L", "r", "w", "w_factor", "m")})
                # met["n"] significa número de consultas, NO tamaño de base.
                # Mantener siempre el n de step["n"] para CSV y gráficos.
                obj.update({key: met.get(key) for key in headers if key in met and key != "n"})
                writer.writerow({key: obj.get(key) for key in headers})

def _attach_exact_benchmark(report: dict, data_dir: Path, raw: Path | str, out: Path) -> None:
    """Compara computación exacta real desde A1 con búsqueda ANN de sketch listo.

    La razón es exploratoria: el cociente NO es speedup end-to-end porque el
    tiempo de indexar/proyectar ANN se excluye de la latencia de consulta.
    """
    from .benchmark_exact_streaming import benchmark_exact_streaming, save_exact_benchmark
    phase = "val" if report["phase"] == "validation" else "test"
    rows = benchmark_exact_streaming(data_dir, raw, phase, report["sizes"], report["k"])
    save_exact_benchmark(rows, out / f"exact_streaming_{phase}.csv")
    lookup = {(row["n"], row["metric"]): row for row in rows}
    report["exact_streaming_benchmark_stage"] = "new_streaming_A1_full_gene_exact_search"
    for step in report["results"]:
        for row in step["results"]:
            metrics = row.get("metrics")
            if not metrics:
                continue
            baseline = lookup[(step["n"], row["metric"])]
            ref = baseline["amortized_compute_ms_per_query"]
            metrics["exact_stream_ms_per_query_amortized"] = ref
            # Cociente exploratorio, no confundir con speedup de sistema completo.
            lat = metrics.get("latency_ms_median")
            if row["config"]["method"].startswith("exact"):
                metrics["amortized_ratio_vs_exact_stream"] = None
            else:
                metrics["amortized_ratio_vs_exact_stream"] = float(ref/lat) if lat else None


def validate_streaming(data_dir: Path | str, out: Path | str, *, k: int = 10,
                       sizes: list[int] | None = None, methods: set[str] | None = None,
                       seed: int | None = None, verify_hash: bool = False,
                       replicas: int = 1, exact_benchmark_raw: Path | str | None = None) -> dict:
    """Calibra w y selecciona una configuración por algoritmo y tamaño n SOLO con val."""
    data_dir, out = Path(data_dir), Path(out)
    data = _phase(data_dir, "val", verify_hash)
    if seed is not None and seed != data["meta"]["seed"]:
        raise ValueError("La semilla debe coincidir con la usada para acumular proyecciones")
    seed = data["meta"]["seed"]
    sizes = _sizes(len(data["order"]), k, sizes)
    selection, all_results = [], []
    for n in sizes:
        truth = _truths(data, n, k)
        _, l2_dist, _ = truth["cosine"]
        _, l1_dist, _ = truth["l1"]
        ws = {"e2lsh_p2": suggest_w(l2_dist, factor=1.0),
              "e2lsh_p1": suggest_w(l1_dist, factor=1.0)}
        configurations = _configs(seed, data["meta"]["cols"], data["meta"]["lp_cols"], ws, methods, replicas)
        base_y = data["labels"][data["order"][:n]]
        query_y = data["labels"][data["query"]]
        ex_class = {m: _predict_metrics(query_y, knn_vote(v[0], base_y), data["classes"])
                    for m, v in truth.items()}
        ex_pred = {m: knn_vote(v[0], base_y) for m, v in truth.items()}
        label_counts = {str(c): int(np.sum(base_y == c)) for c in data["classes"]}
        rows = []
        for config in configurations:
            metric = "l1" if config["method"] in {"e2lsh_p1", "exact_l1"} else "cosine"
            print(f"[validación] n={n}: {config}", flush=True)
            try:
                scores = _evaluate(config, data, n, k, truth[metric][0], ex_class[metric],
                                   truth[metric][1], ex_pred[metric], truth[metric][2], replicas)
                rows.append({"config": config, "metric": metric, "metrics": scores})
            except (MemoryError, ValueError, RuntimeError) as exc:
                rows.append({"config": config, "metric": metric, "error": f"{type(exc).__name__}: {exc}"})
            gc.collect()
        chosen = select_frozen(rows)
        selection.append({"n": n, "w_reference": ws, "selected": chosen})
        all_results.append({"n": n, "w_reference": ws, "exact_classification": ex_class,
                            "base_class_counts": label_counts, "results": rows})
    pack = {"phase": "validation", "mode": "streaming", "dataset_id": data["meta"]["dataset_id"],
            "k": k, "sizes": sizes, "seed": seed, "replicas": replicas, "n_queries": len(data["query"]),
            "results": all_results}
    if exact_benchmark_raw is not None:
        _attach_exact_benchmark(pack, data_dir, exact_benchmark_raw, out)
    _save(out / "validation.json", pack)
    _write_csv(out / "validation.csv", pack)
    freeze = {"phase": "frozen_after_validation", "mode": "streaming",
              "dataset_id": data["meta"]["dataset_id"], "k": k, "sizes": sizes,
              "seed": seed, "replicas": replicas, "selected_by_size": selection,
              "quality_thresholds": QUALITY_POLICY.copy(),
              "policy_status": "preregistered_for_next_real_run; do_not_modify_after_test",
              "selection_rule": "strict_quality_and_candidates_then_diagnostic_unqualified_fallback"}
    _save(out / "configuraciones_congeladas.json", freeze)
    return pack


def final_test_streaming(data_dir: Path | str, selection_path: Path | str, out: Path | str,
                         *, verify_hash: bool = False,
                         exact_benchmark_raw: Path | str | None = None) -> dict:
    """Evalúa el conjunto test una vez por n y método, sin modificar selección ni w."""
    data_dir, out = Path(data_dir), Path(out)
    data = _phase(data_dir, "test", verify_hash)
    frozen = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if frozen.get("phase") != "frozen_after_validation" or frozen.get("mode") != "streaming":
        raise ValueError("Se requiere una selección congelada del protocolo streaming")
    if frozen.get("dataset_id") != data["meta"]["dataset_id"]:
        raise ValueError("La selección corresponde a otra cohorte, semilla o preprocesamiento")
    if frozen.get("quality_thresholds") != QUALITY_POLICY:
        raise ValueError("La política de calidad congelada no coincide con el protocolo preregistrado; "
                         "debe repetir validación, no reajustar el test")
    k = int(frozen["k"])
    replicas = int(frozen.get("replicas", 1))
    all_results = []
    for size in frozen["selected_by_size"]:
        n = int(size["n"])
        truth = _truths(data, n, k)
        base_y = data["labels"][data["order"][:n]]
        query_y = data["labels"][data["query"]]
        exact_class = {m: _predict_metrics(query_y, knn_vote(v[0], base_y), data["classes"])
                       for m, v in truth.items()}
        exact_pred = {m: knn_vote(v[0], base_y) for m, v in truth.items()}
        label_counts = {str(c): int(np.sum(base_y == c)) for c in data["classes"]}
        rows = []
        for entry in size["selected"]:
            config = entry["config"]
            metric = "l1" if config["method"] in {"e2lsh_p1", "exact_l1"} else "cosine"
            print(f"[test final] n={n}: {config}", flush=True)
            scores = _evaluate(config, data, n, k, truth[metric][0], exact_class[metric],
                               truth[metric][1], exact_pred[metric], truth[metric][2], replicas)
            rows.append({"config": config, "metric": metric, "qualified_on_validation": entry["qualified"],
                         "selection_reason": entry.get("selection_reason"), "metrics": scores})
            gc.collect()
        all_results.append({"n": n, "exact_classification": exact_class,
                            "base_class_counts": label_counts, "results": rows})
    report = {"phase": "test_final", "mode": "streaming", "dataset_id": data["meta"]["dataset_id"],
              "sizes": frozen["sizes"], "k": k, "seed": frozen["seed"], "replicas": frozen.get("replicas", 1),
              "n_queries": len(data["query"]), "results": all_results}
    if exact_benchmark_raw is not None:
        _attach_exact_benchmark(report, data_dir, exact_benchmark_raw, out)
    _save(out / "test_final.json", report)
    _write_csv(out / "test_final.csv", report)
    return report
