"""Benchmark MEDIDO del exacto desde A1 en streaming, sin matriz densa.

A diferencia de recuperar el top-k desde Gram/L1 precomputados, este benchmark
relee A1 (gen por gen) para cada (n, métrica, fase) y computa desde cero las
comparaciones Q×B_n. Es deliberadamente opcional: puede costar muchos minutos
con miles de consultas, bases grandes y decenas de miles de genes.

No confundir una comparación de esta búsqueda de novo con búsquedas ANN sobre
sketches precalculados: consultar el ANN no incluye el costo de generar los
sketches. Se reporta la razón exploratoria separadamente.
"""
from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .algoritmos.base import topk
from .algoritmos.exacto.exact import StreamingExactKNN
from .cohorte_validada import build_cohort
from .preparacion_streaming import _single_source, _transform, peak_rss_mib


def benchmark_exact_streaming(data_dir: Path | str, raw_dir: Path | str,
                              phase: str, sizes: list[int], k: int,
                              *, batch_size: int = 256, log=print) -> list[dict]:
    """Tiempos por (n,métrica) usando A1; requiere los mismos datos de preparación.

    Reproduce genes aceptados y parámetros por gen calculados con train.
    Cada par se acumula en archivos temporales mmap y se borra al terminar.
    """
    data_dir, raw_dir = Path(data_dir), Path(raw_dir)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if phase not in {"val", "test"}:
        raise ValueError("phase debe ser val o test")
    spl = np.load(data_dir / "splits.npz", allow_pickle=False)
    sample_ids = spl["sample_ids"].astype(str)
    n_samples = len(sample_ids)
    order = spl["base_order"]
    queries = spl[f"query_{phase}"]
    stats = {str(g): (float(mu), float(sd)) for g, mu, sd in
             zip(spl["feature_ids"], spl["gene_mean_train"], spl["gene_std_train"])}
    a1 = _single_source(raw_dir, "expression")
    reader = a1.records()
    schema = next(reader)
    reader.close()
    cohort = build_cohort(schema.fields["columns"], raw_dir,
                          check_annotations=meta["check_annotations"])
    if cohort.sample_ids != sample_ids.tolist() or cohort.patient_ids != spl["patient_ids"].tolist():
        raise ValueError("Los archivos A1/A2/A3 ya no coinciden con la cohorte preparada")
    if not meta["zscore"]:
        raise ValueError("Benchmark de repetición requiere zscore=True para reproducir los datos")
    rows = []
    for n in sizes:
        if n < k or n > len(order):
            raise ValueError(f"Tamaño de base fuera de rango: {n}")
        for metric in ("cosine", "l1"):
            log(f"[benchmark exacto streaming] {phase}, n={n}, métrica={metric}")
            with tempfile.TemporaryDirectory(prefix="tcgaknn-exact-") as tmp:
                path = Path(tmp) / ("scores.npy")
                ex = StreamingExactKNN(n_samples, queries, order[:n],
                                       metrics=(metric,), standardize=False,
                                       batch_size=batch_size,
                                       gram_path=path if metric == "cosine" else None,
                                       l1_path=path if metric == "l1" else None)
                start = time.perf_counter()
                compute_seconds = 0.0
                seen = set()
                for rec in a1.records():
                    if rec.kind != "feature_row" or str(rec.key) not in stats:
                        continue
                    gid = str(rec.key)
                    if gid in seen:
                        raise ValueError(f"Gen repetido en benchmark: {gid}")
                    seen.add(gid)
                    mu, sd = stats[gid]
                    val = _transform(rec.values[cohort.columns], meta["transform"])
                    z = np.where(np.isfinite(val), (val-mu)/sd, 0).astype(np.float32)
                    t0 = time.perf_counter()
                    ex.update(gid, z)
                    compute_seconds += time.perf_counter()-t0
                if len(seen) != len(stats):
                    raise ValueError(f"Benchmark: faltan {len(stats)-len(seen)} genes de la preparación")
                t0 = time.perf_counter()
                ex._flush()
                compute_seconds += time.perf_counter()-t0
                scores = ex.G if metric == "cosine" else ex.L1
                scores.flush()
                base_norm = np.sqrt(ex.sq_b) if metric == "cosine" else None
                topk_start = time.perf_counter()
                # Comprobar los resultados del benchmark contra la verdad mmap.
                reference = np.load(data_dir / ("gram.npy" if metric == "cosine" else "l1.npy"),
                                    mmap_mode="r")
                qval_len = len(spl["query_val"])
                start_row = 0 if phase == "val" else qval_len
                maximum_diff = 0.0
                for i in range(len(queries)):
                    row = np.asarray(scores[i, :], dtype=np.float64)
                    expected = np.asarray(reference[start_row+i, :n], dtype=np.float64)
                    maximum_diff = max(maximum_diff, float(np.max(np.abs(row-expected))))
                    if metric == "cosine":
                        denom = np.sqrt(ex.sq_q[i]) * base_norm
                        row = np.divide(row, denom, out=np.zeros_like(row), where=denom > 1e-24)
                    topk(row[None, :], k, largest=(metric == "cosine"))
                topk_seconds = time.perf_counter()-topk_start
                total_seconds = time.perf_counter()-start
                if maximum_diff > 1e-3 * max(1, meta["genes_used"]):
                    raise ValueError(f"Las distancias reconstruidas no coinciden: diferencia={maximum_diff:g}")
                rows.append({"phase": phase, "n": n, "metric": metric,
                             "n_queries": len(queries), "n_genes": len(stats),
                             "total_seconds_with_io": float(total_seconds),
                             "accumulate_seconds": float(compute_seconds),
                             "topk_seconds": float(topk_seconds),
                             "amortized_compute_ms_per_query": float(
                                 (compute_seconds + topk_seconds) * 1000 / len(queries)),
                             "amortized_total_ms_per_query": float(
                                 total_seconds * 1000 / len(queries)),
                             "max_abs_score_diff_vs_prepared": maximum_diff,
                             "process_peak_rss_mib_cumulative": peak_rss_mib(),
                             "stage": "new_streaming_A1_full_gene_exact_search"})
    return rows


def save_exact_benchmark(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
