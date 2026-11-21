"""Gráficos a partir de validation.csv o test_final.csv, sin recalcular ni ajustar modelos.

Uso: python scripts/graficar_escalabilidad.py --csv results/escalabilidad/test_final.csv
Por motivos metodológicos, los tiempos del exacto con scores precalculados
NO se superponen con búsquedas ANN sobre sketches como si fueran equivalentes.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="validation.csv o test_final.csv")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pareado", type=Path, default=None,
                   help="CSV opcional generado por benchmark_pareado.py: comparación justa por consulta")
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Instala las dependencias para gráficas: pip install -e '.[plot]'") from exc
    out = args.out or args.csv.parent / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("CSV vacío")
    graphs = [
        ("accuracy", "Accuracy de clasificación", "accuracy_vs_n.png"),
        ("vote_concordance_vs_exact", "Concordancia del voto vs exacto", "concordancia_vs_n.png"),
        ("recall_at_k", "Recall@k", "recall_vs_n.png"),
        ("fraction_candidates", "Fracción de la base examinada", "candidatos_vs_n.png"),
        ("latency_ms_median", "Búsqueda ANN: latencia sobre sketch precalculado (ms)", "latencia_ann_vs_n.png"),
        ("build_seconds", "Construcción del índice a partir del sketch (s)", "construccion_vs_n.png"),
        ("resident_bytes_estimate", "Bytes residentes estimados del índice", "memoria_indice_vs_n.png"),
        ("exact_stream_ms_per_query_amortized", "Exacto streaming: tiempo amortizado (ms/consulta)", "exacto_streaming_vs_n.png"),
    ]
    for key, title, filename in graphs:
        groups = defaultdict(list)
        for row in records:
            method = row["method"]
            # No combinar latencias con distintas etapas de cómputo en una curva.
            if key in {"latency_ms_median", "build_seconds", "resident_bytes_estimate"} and method.startswith("exact"):
                continue
            if key == "exact_stream_ms_per_query_amortized" and not method.startswith("exact"):
                continue
            if key in {"fraction_candidates", "recall_at_k", "vote_concordance_vs_exact"} and method.startswith("exact"):
                continue
            if not row.get(key):
                continue
            try:
                groups[(method, row.get("replica", ""))].append((int(row["n"]), float(row[key])))
            except (TypeError, ValueError):
                continue
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, replica), pairs in sorted(groups.items()):
            pairs = sorted(pairs)
            label = method + (f" (réplica {replica})" if replica else "")
            ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=label)
        ax.set(title=title, xlabel="Pacientes en base train (n)", ylabel=key)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
        print(out / filename)
    # El benchmark exacto de novo tiene un CSV aparte: no mezclarlo
    # automáticamente con el tiempo del índice que recibe Q ya proyectada.
    if args.pareado is not None:
        with args.pareado.open(newline="", encoding="utf-8") as handle:
            paired = list(csv.DictReader(handle))
        for name, column, ylabel in (
            ("tiempo_pareado_vs_n.png", None, "Milisegundos por consulta"),
            ("speedup_pareado_vs_n.png", "speedup_vs_dense_exact", "Exacto denso / ANN+proyección"),
        ):
            groups = defaultdict(list)
            for row in paired:
                method = row["method"]
                if column is None:
                    measure = ("exact_single_query_median_ms" if method.startswith("exact_dense_")
                               else "ann_projection_plus_search_median_ms")
                else:
                    if method.startswith("exact_dense_"):
                        continue
                    measure = column
                if not row.get(measure):
                    continue
                groups[(method, row.get("replica", ""))].append((int(row["n"]), float(row[measure])))
            if not groups:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for (method, replica), pairs in sorted(groups.items()):
                pairs = sorted(pairs)
                ax.plot([n for n, _ in pairs], [v for _, v in pairs], marker="o",
                        label=method + (f" (rep {replica})" if replica else ""))
            if column is not None:
                ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set(title="Benchmark pareado: consulta individual",
                   xlabel="Pacientes en base train (n)", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / name, dpi=160)
            plt.close(fig)
            print(out / name)


if __name__ == "__main__":
    main()
