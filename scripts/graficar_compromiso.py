"""Curvas de compromiso calidad/costo a partir de validation.csv (sin recalcular nada).

Para cada tamaño de base n dibuja dos paneles con todas las configuraciones
evaluadas de la grilla (L, r, w):

  1. recall@k vs fracción de la base examinada (candidatos / n)
  2. recall@k vs latencia mediana de consulta ANN (ms, sketch precalculado)

Es la curva "recall@k frente a tiempo" comprometida en la propuesta. Los
métodos exactos no aparecen: su recall es 1 por definición y su latencia en el
CSV principal es top-k sobre distancias precalculadas (otra etapa de cómputo).
Si se entrega --selection, la configuración congelada de cada método se marca
con un anillo.

Uso:
    python scripts/graficar_compromiso.py --csv results/escalabilidad/validation.csv \
        --selection results/escalabilidad/configuraciones_congeladas.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Paleta categórica de referencia (3 primeros slots: válidos para dispersión).
COLORS = {"lsh": "#2a78d6", "e2lsh_p2": "#eb6834", "e2lsh_p1": "#1baf7a"}
LABELS = {"lsh": "LSH coseno (SimHash por bandas)",
          "e2lsh_p2": "E2LSH p=2 (Gauss, coseno)",
          "e2lsh_p1": "E2LSH p=1 (Cauchy, L1)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _selected(path: Path | None) -> set[tuple]:
    if path is None:
        return set()
    frozen = json.loads(path.read_text(encoding="utf-8"))
    keys = set()
    for size in frozen["selected_by_size"]:
        for entry in size["selected"]:
            c = entry["config"]
            if c["method"].startswith("exact"):
                continue
            keys.add((int(size["n"]), c["method"], str(c.get("replica", 0)), str(c.get("L")),
                      str(c.get("r")), _round(c.get("w"))))
    return keys


def _round(w) -> str:
    return "" if w in (None, "") else f"{float(w):.6g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True, help="validation.csv (todas las configuraciones)")
    ap.add_argument("--selection", type=Path, default=None, help="configuraciones_congeladas.json (opcional)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="n a graficar (por defecto todos)")
    ap.add_argument("--min-recall", type=float, default=0.90, help="Línea de referencia de la política")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Instala las dependencias para gráficas: pip install -e '.[plot]'") from exc

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if not r["method"].startswith("exact") and not r.get("error")]
    if not rows:
        raise SystemExit("El CSV no tiene configuraciones ANN evaluadas")
    chosen = _selected(args.selection)
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[int(r["n"])].append(r)
    sizes = sorted(by_n) if args.sizes is None else [n for n in args.sizes if n in by_n]
    out = args.out or args.csv.parent / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    k = rows[0].get("k", "k")

    for n in sizes:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        panels = (("fraction_candidates", "Fracción de la base examinada", False),
                  ("latency_ms_median", "Latencia mediana por consulta ANN (ms, escala log)", True))
        for ax, (xkey, xlabel, logx) in zip(axes, panels):
            for method in ("lsh", "e2lsh_p2", "e2lsh_p1"):
                pts = [(_float(r[xkey]), _float(r["recall_at_k"]), r) for r in by_n[n] if r["method"] == method]
                pts = [(x, y, r) for x, y, r in pts if x is not None and y is not None and (x > 0 or not logx)]
                if not pts:
                    continue
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=COLORS[method],
                           edgecolor="white", linewidth=1.2, label=LABELS[method], zorder=3, alpha=0.9)
                for x, y, r in pts:
                    key = (n, method, r.get("replica") or "0", r.get("L"), r.get("r"), _round(r.get("w")))
                    if key in chosen:
                        ax.scatter([x], [y], s=150, facecolor="none", edgecolor=COLORS[method],
                                   linewidth=2, zorder=4)
            ax.axhline(args.min_recall, color=MUTED, linestyle="--", linewidth=1, zorder=1)
            ax.text(ax.get_xlim()[0], args.min_recall, f"  objetivo recall ≥ {args.min_recall:.2f}",
                    color=MUTED, fontsize=8, va="bottom")
            if logx:
                from matplotlib.ticker import FuncFormatter
                ax.set_xscale("log")
                plain = FuncFormatter(lambda v, _: f"{v:g}")
                ax.xaxis.set_major_formatter(plain)
                ax.xaxis.set_minor_formatter(plain)
            ax.set_xlabel(xlabel, color=TEXT)
            ax.grid(color=GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED)
        axes[0].set_ylabel(f"Recall@{k} vs exacto", color=TEXT)
        axes[0].set_ylim(-0.02, 1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        if chosen:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=11, markerfacecolor="none",
                                  markeredgecolor=MUTED, markeredgewidth=2))
            labels.append("configuración congelada")
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
        fig.suptitle(f"Compromiso calidad/costo en validación · base n = {n} pacientes", color=TEXT)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        path = out / f"compromiso_recall_n{n}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
