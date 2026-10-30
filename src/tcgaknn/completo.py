"""Expresión completa por stream con el mejor algoritmo (SimHash + re-ranking con RP).

En **una** pasada del stream de expresión (todos los genes, todos los pacientes
elegibles) se construyen:

- los sketches del método: proyección RP de dimensión m y firmas SimHash de b
  bits (``StreamingProjectionAccumulator``, Ω generada por ID de gen);
- la verdad de terreno: coseno exacto entre las consultas fijas Q y la base B
  (``StreamingExactKNN``), sin guardar la matriz completa.

Luego se evalúa:

- **búsqueda**: recall@k, pérdida de similitud del top-k y razón de
  aproximación en distancia coseno, y escalamiento en bases anidadas;
- **error de estimación**: coseno estimado por SimHash (cos(π·Ham/b)) y por RP
  (producto punto de las proyecciones) frente al coseno real, en todos los
  pares Q × B y en los k vecinos verdaderos;
- **clasificación** del tipo de cáncer por voto: exactitud, F1-macro,
  exactitud balanceada, F1 por clase y acuerdo de voto, con IC bootstrap de la
  diferencia aproximado − exacto.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tcgaknn.algoritmos import SimHashKNN, StreamingExactKNN, StreamingProjectionAccumulator, topk
from tcgaknn.algoritmos.coseno import cosine_estimate, hamming_matrix, pack_bits
from tcgaknn.algoritmos.evaluacion import classification_report, knn_vote, recall_at_k, top_confusions, vote_agreement
from tcgaknn.escalabilidad import _peak_rss_mib, _standardize
from tcgaknn.perfil import load_labels, log2p1, select_samples
from tcgaknn.stream import StreamService


def stratified_queries(y: np.ndarray, n_queries: int, seed: int, min_per_class: int = 2) -> np.ndarray:
    """Posiciones de consulta estratificadas por clase (proporcional, con un mínimo por clase)."""
    rng = np.random.default_rng(seed + 1)
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(min_per_class, np.round(n_queries * counts / counts.sum()).astype(int))
    quota = np.minimum(quota, counts - 1)  # deja al menos uno en la base
    out = []
    for c, qn in zip(classes, quota):
        pos = np.flatnonzero(y == c)
        out.append(rng.choice(pos, size=max(qn, 0), replace=False))
    return np.sort(np.concatenate(out))


def stream_pass(root: Path | str, *, n_queries: int, seed: int, b: int, m: int, g_max: int | None = None,
                n_samples: int | None = None, labels: dict[str, str] | None = None,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    labels = load_labels(root) if labels is None else labels
    t0 = time.perf_counter()
    n_rows = 0
    for rec in StreamService(root, data_types=["expression"]):
        if rec.kind == "schema":
            barcodes = rec.fields["columns"]
            cols_idx, y, info = select_samples(barcodes, labels, n_samples or len(barcodes), seed)
            y = np.asarray(y)
            N = cols_idx.size
            q = stratified_queries(y, n_queries, seed)
            base = np.setdiff1d(np.arange(N), q)
            fit_mask = np.zeros(N, dtype=bool)
            fit_mask[base] = True
            acc = StreamingProjectionAccumulator(N, max(b, m), seed, standardize=False)
            exact = StreamingExactKNN(N, q, base, metrics=("cosine",), standardize=False)
            log(f"  {N:,} pacientes elegibles, {q.size} consultas, base {base.size:,}")
            continue
        if rec.kind != "feature_row":
            continue
        if g_max is not None and n_rows >= g_max:
            break
        n_rows += 1
        z = _standardize(log2p1(rec.values[cols_idx].astype(np.float64)), fit_mask)
        if z is None:
            continue
        acc.update(rec.key, z)
        exact.update(rec.key, z)
        if n_rows % 2000 == 0:
            log(f"  {n_rows:,} genes, {time.perf_counter() - t0:,.0f} s")
    pass_s = time.perf_counter() - t0
    S = exact.scores("cosine")  # |Q| × |B|, coseno exacto
    return {
        "P": acc.projections(m),
        "bits": acc.sign_bits(b),
        "S": S,
        "y": y,
        "barcodes": [barcodes[i] for i in cols_idx],
        "q": q,
        "base": base,
        "pasada": {
            "genes_leidos": n_rows,
            "genes_usados": exact.n_features,
            "pacientes": int(N),
            "consultas": int(q.size),
            "base": int(base.size),
            "segundos": pass_s,
            "genes_por_segundo": n_rows / pass_s if pass_s else 0.0,
            "memoria_pico_proceso_MiB": _peak_rss_mib(),
            "bytes_sketches": int(acc.Y.nbytes),
            "bytes_verdad_QxB": int(exact.G.nbytes),
            "bytes_matriz_completa_evitada": int(N) * int(exact.n_features) * 4,
            "seleccion": info,
        },
    }


def _err_stats(est: np.ndarray, real: np.ndarray) -> dict[str, float]:
    e = est - real
    r = np.corrcoef(est.ravel(), real.ravel())[0, 1] if e.size > 1 else float("nan")
    return {"n_pares": int(e.size), "sesgo": float(e.mean()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "error_p95_abs": float(np.percentile(np.abs(e), 95)),
            "correlacion": float(r)}


def _bootstrap_delta(yq: np.ndarray, pa: np.ndarray, pe: np.ndarray, labels: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(reps):
        i = rng.integers(0, yq.size, yq.size)
        ra = classification_report(yq[i], pa[i], labels=labels)
        re = classification_report(yq[i], pe[i], labels=labels)
        acc.append(ra["accuracy"] - re["accuracy"])
        f1.append(ra["f1_macro"] - re["f1_macro"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"delta_exactitud_IC95": q(acc), "delta_f1_macro_IC95": q(f1)}


def evaluate(data: dict[str, Any], *, b: int, m: int, f: int, k: int = 10, sizes: Sequence[int] = (),
             bootstrap: int = 1000, seed: int = 2026, log: Callable[[str], None] = print) -> dict[str, Any]:
    P, bits, S, y, q, base = data["P"], data["bits"], data["S"], data["y"], data["q"], data["base"]
    yq, yb = y[q], y[base]
    Bp, Qp = pack_bits(bits[base, :b]), pack_bits(bits[q, :b])
    res: dict[str, Any] = {"metodo": f"SimHash-{b} + re-ranking RP-{m} (f = {f})"}

    # --- búsqueda sobre la base completa (consulta a consulta) ---
    t0 = time.perf_counter()
    idx_m = SimHashKNN.from_signatures(Bp, b, base=P[base], rerank_factor=f)
    build_s = time.perf_counter() - t0
    for i in range(min(5, q.size)):
        idx_m.query_packed(Qp[i : i + 1], k, Q_base=P[q[i : i + 1]])
    lat, rows = [], []
    for i in range(q.size):
        t1 = time.perf_counter()
        ix, _, _ = idx_m.query_packed(Qp[i : i + 1], k, Q_base=P[q[i : i + 1]])
        lat.append((time.perf_counter() - t1) * 1e3)
        rows.append(ix[0])
    A = np.stack(rows)  # vecinos aproximados (posiciones en la base)
    E, Esim = topk(S, k)  # vecinos exactos y su coseno
    true_of_A = np.take_along_axis(S, np.where(A >= 0, A, 0), axis=1)
    true_of_A[A < 0] = np.nan
    res["busqueda"] = {
        **{f"recall@{kk}": recall_at_k(A, E, kk) for kk in sorted({1, 5, k}) if kk <= k},
        "coseno_medio_topk_exacto": float(np.nanmean(Esim)),
        "coseno_medio_topk_aproximado": float(np.nanmean(true_of_A)),
        "perdida_similitud_topk": float(np.nanmean(Esim) - np.nanmean(true_of_A)),
        "razon_aprox_distancia_coseno": float(np.nanmean((1 - true_of_A) / np.maximum(1 - Esim, 1e-12))),
        "t_consulta_ms_mediana": float(np.median(lat)),
        "t_consulta_ms_p95": float(np.percentile(lat, 95)),
        "t_construccion_s": build_s,
        "bytes_por_paciente": float(Bp.nbytes + P[base].nbytes) / base.size,
        "bytes_por_paciente_exacto_equivalente": data["pasada"]["genes_usados"] * 4,
    }
    log(f"  recall@{k} = {res['busqueda'][f'recall@{k}']:.3f}, {np.median(lat):.3f} ms/consulta")

    # --- error de estimación: todos los pares y los k vecinos verdaderos ---
    H = hamming_matrix(Qp, Bp)
    cos_sh = cosine_estimate(H, b)
    cos_rp = P[q].astype(np.float64) @ P[base].astype(np.float64).T
    near = np.zeros_like(S, dtype=bool)
    np.put_along_axis(near, E, True, axis=1)
    theta = np.arccos(np.clip(S, -1, 1))
    p = theta / np.pi
    res["error_estimacion"] = {
        "simhash_todos": _err_stats(cos_sh, S),
        "simhash_vecinos": _err_stats(cos_sh[near], S[near]),
        "rp_todos": _err_stats(cos_rp, S),
        "rp_vecinos": _err_stats(cos_rp[near], S[near]),
        "simhash_error_angular_rad": {
            "rmse_observado": float(np.sqrt(((np.pi * H / b - theta) ** 2).mean())),
            "rmse_teorico": float(np.pi * np.sqrt((p * (1 - p) / b).mean())),
        },
    }

    # --- clasificación del tipo de cáncer (objetivo del proyecto): predicho vs real ---
    labels = np.unique(y)
    keys = ("accuracy", "f1_macro", "balanced_accuracy")

    def summary(rep: dict) -> dict:
        out = {key: rep[key] for key in keys}
        out["tasa_de_error"] = 1.0 - rep["accuracy"]
        out["f1_ponderado"] = float(np.sum(rep["f1"] * rep["support"]) / max(rep["support"].sum(), 1))
        return out

    # Referencias mínimas: clase mayoritaria de la base y azar estratificado (esperado)
    maj = classification_report(yq, np.full(yq.shape, max(set(yb.tolist()), key=yb.tolist().count)), labels=labels)
    prior = np.array([np.mean(yb == c) for c in labels])
    cls = {"referencias": {"clase_mayoritaria": summary(maj),
                           "azar_estratificado_exactitud_esperada": float(np.sum(prior * np.array([np.mean(yq == c) for c in labels])))}}
    preds = {}
    for kk in (1, 5, 10):
        if kk > k:
            continue
        pe = knn_vote(E[:, :kk], yb)
        pa = knn_vote(A[:, :kk], yb)
        preds[kk] = (pe, pa)
        re, ra = classification_report(yq, pe, labels=labels), classification_report(yq, pa, labels=labels)
        cls[f"k={kk}"] = {"exacto": summary(re), "aproximado": summary(ra), "acuerdo_de_voto": vote_agreement(pa, pe)}
        if kk == min(5, k):
            cls[f"k={kk}"].update(_bootstrap_delta(yq, pa, pe, labels, bootstrap, seed))
            res["k_principal"] = kk
            res["por_clase"] = [
                {"tipo": str(t), "consultas": int(re["support"][i]),
                 "precision_exacto": float(re["precision"][i]), "recall_exacto": float(re["recall"][i]),
                 "f1_exacto": float(re["f1"][i]),
                 "precision_aproximado": float(ra["precision"][i]), "recall_aproximado": float(ra["recall"][i]),
                 "f1_aproximado": float(ra["f1"][i])} for i, t in enumerate(labels)]
            res["confusiones"] = {
                "exacto": [list(map(str, c[:2])) + [c[2]] for c in top_confusions(re["confusion"], labels, 10)],
                "aproximado": [list(map(str, c[:2])) + [c[2]] for c in top_confusions(ra["confusion"], labels, 10)],
            }
            res["_matrices"] = {"labels": labels, "confusion_exacto": re["confusion"], "confusion_aproximado": ra["confusion"],
                                "y_real": yq, "pred_exacto": pe, "pred_aproximado": pa}
    res["clasificacion"] = cls

    # --- escalamiento en bases anidadas (verdad = columnas de S) ---
    order = np.random.default_rng(seed + 2).permutation(base.size)
    esc = []
    for n in [s for s in sizes if s < base.size] + [base.size]:
        sub = np.sort(order[:n])
        Sn = S[:, sub]
        En = topk(Sn, k)[0]
        ix_n = SimHashKNN.from_signatures(Bp[sub], b, base=P[base][sub], rerank_factor=f)
        lat_n, rows_n = [], []
        for i in range(q.size):
            t1 = time.perf_counter()
            r_, _, _ = ix_n.query_packed(Qp[i : i + 1], k, Q_base=P[q[i : i + 1]])
            lat_n.append((time.perf_counter() - t1) * 1e3)
            rows_n.append(r_[0])
        esc.append({"n": int(n), f"recall@{k}": recall_at_k(np.stack(rows_n), En, k),
                    "t_consulta_ms_mediana": float(np.median(lat_n))})
    res["escalamiento"] = esc
    return res


def run(root: Path | str, out: Path | str, *, n_queries: int = 500, b: int = 256, m: int = 256, f: int = 10,
        k: int = 10, seed: int = 2026, sizes: Sequence[int] = (250, 500, 1000, 2000, 4000, 8000),
        g_max: int | None = None, n_samples: int | None = None, bootstrap: int = 1000,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(root)
    log("Pasada del stream de expresión (todos los genes)")
    data = stream_pass(root, n_queries=n_queries, seed=seed, b=b, m=m, g_max=g_max, n_samples=n_samples,
                       labels=labels, log=log)
    log("Evaluación")
    res = evaluate(data, b=b, m=m, f=f, k=k, sizes=sizes, bootstrap=bootstrap, seed=seed, log=log)
    res["parametros"] = {"n_queries": n_queries, "b": b, "m": m, "f": f, "k": k, "seed": seed, "g_max": g_max}
    res["pasada"] = data["pasada"]
    mats = res.pop("_matrices")
    np.savez_compressed(out / "confusion.npz", **{key: np.asarray(v) for key, v in mats.items()})
    with open(out / "predicciones.csv", "w", encoding="utf-8") as fh:
        fh.write("tipo_real,pred_exacto,pred_aproximado\n")
        for t, pe, pa in zip(mats["y_real"], mats["pred_exacto"], mats["pred_aproximado"]):
            fh.write(f"{t},{pe},{pa}\n")
    # La clasificación del tipo de cáncer es el objetivo: va primero en el JSON
    order = ["metodo", "k_principal", "clasificacion", "por_clase", "confusiones", "busqueda", "error_estimacion",
             "escalamiento", "parametros", "pasada"]
    res = {key: res[key] for key in order if key in res}
    (out / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    np.savez_compressed(out / "estimado_vs_real.npz",
                        real=data["S"].astype(np.float32),
                        simhash=cosine_estimate(hamming_matrix(pack_bits(data["bits"][data["q"], :b]),
                                                               pack_bits(data["bits"][data["base"], :b])), b).astype(np.float32),
                        rp=(data["P"][data["q"]] @ data["P"][data["base"]].T).astype(np.float32))
    return res
