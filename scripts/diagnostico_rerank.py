"""Diagnóstico SOLO VALIDACIÓN: ¿el recall se pierde al generar candidatos o al re-rankear?

Para cada configuración: reproduce el recall@k del protocolo (re-ranking en el espacio
proyectado, como en validation.csv) y calcula el recall si los MISMOS candidatos se
ordenaran con la distancia exacta (Gram/L1 precalculados de validación). La diferencia
entre ambos es la pérdida atribuible al re-ranking; 1 - recall_exacto es la pérdida
atribuible a los candidatos. No toca test, no modifica resultados del protocolo.

Es un diagnóstico exploratorio (no forma parte del protocolo validar -> congelar -> test).
Configuraciones: las 4 de LSH y algunas de E2LSH (incluidas las congeladas y las de mayor
recall) para n=1000 y n=6459.

Uso (desde la raíz del repo):
    python scripts/diagnostico_rerank.py data/processed results/escalabilidad/diagnostico_rerank_val.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from tcgaknn import experimento_streaming as ex  # noqa: E402
from tcgaknn.algoritmos.evaluacion import knn_vote, recall_at_k  # noqa: E402

data_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
K = 10
data = ex._phase(data_dir, "val", verify_hash=False)
assert len(data["query"]) == 1384
q = data["query"]
sel_w = {"e2lsh_p2": None, "e2lsh_p1": None}

# Configuraciones a diagnosticar: (n, método, L, r, w_factor); w se toma del JSON congelado.
import json  # noqa: E402
frozen = json.load(open("results/escalabilidad/configuraciones_congeladas.json"))
wref = {b["n"]: b["w_reference"] for b in frozen["selected_by_size"]}
CONFIGS = []
for n in (1000, 6459):
    for L, r in ex.COS_GRID:
        CONFIGS.append((n, "lsh", L, r, None))
    CONFIGS += [(n, "e2lsh_p2", 8, 2, 1.0), (n, "e2lsh_p2", 16, 2, 1.0), (n, "e2lsh_p2", 32, 4, 1.0),
                (n, "e2lsh_p1", 16, 2, 1.0), (n, "e2lsh_p1", 16, 2, 0.75)]

rows = []
for n, method, L, r, wf in CONFIGS:
    base = data["order"][:n]
    ya, yq = data["labels"][base], data["labels"][q]
    metric = "l1" if method == "e2lsh_p1" else "cosine"
    truth, _, _ = ex._truth(data, n, K, metric)
    if method == "lsh":
        cfg = {"method": "lsh", "L": L, "r": r, "m": 512, "fallback": False, "seed": 2026, "replica": 0}
    else:
        p = 1 if method == "e2lsh_p1" else 2
        cfg = {"method": method, "p": p, "L": L, "r": r, "w": wref[n][method] * wf, "w_factor": wf,
               "rerank": "proj", "rerank_cols": 256, "seed": 2026, "replica": 0}
    query_fn, _, _, idx = ex._build(cfg, data, n, 1, with_index=True)
    proj_ids, cands_n, _ = ex._timed_queries(query_fn, len(q), K, warmup=0)
    if method == "lsh":
        Qbits = data["Y"][q, :L * r] >= 0
        cand_lists = idx.candidates_bits(Qbits)
    else:
        Qp = ex._stable_projected(data, q, cfg["p"], L * r, 0, 1)
        cand_lists = [np.unique(c) for c in idx.candidates_from_projections(Qp)]
    exact_ids = np.full((len(q), K), -1, dtype=np.int64)
    base_norm = np.sqrt(data["sq"][base])
    for i, c in enumerate(cand_lists):
        if len(c) == 0:
            continue
        if metric == "cosine":
            s = np.asarray(data["G"][i, :n], dtype=np.float64)[c] / (np.sqrt(data["sq"][q[i]]) * base_norm[c])
            top = c[np.argsort(-s, kind="stable")[:K]]
        else:
            d = np.asarray(data["L1"][i, :n], dtype=np.float64)[c]
            top = c[np.argsort(d, kind="stable")[:K]]
        exact_ids[i, :len(top)] = top
    acc = lambda ids: float(np.mean(knn_vote(ids, ya) == yq))  # noqa: E731
    row = {"n": n, "method": method, "L": L, "r": r, "w_factor": wf,
           "fraction_candidates": float(cands_n.mean() / n),
           "recall_rerank_proyectado": float(recall_at_k(proj_ids, truth, K)),
           "recall_rerank_exacto_mismos_candidatos": float(recall_at_k(exact_ids, truth, K)),
           "accuracy_rerank_proyectado": acc(proj_ids),
           "accuracy_rerank_exacto_mismos_candidatos": acc(exact_ids),
           "accuracy_exacto_total": acc(truth)}
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

with out_csv.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("OK", out_csv)
