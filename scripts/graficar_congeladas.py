"""Curvas vs n SOLO con la configuración congelada de cada método (sin recalcular nada).

`graficar_escalabilidad.py` está pensado para `test_final.csv`, que tiene una sola
configuración por método. Sobre `validation.csv` (toda la grilla L, r, w) dibuja
cientos de puntos unidos por líneas verticales. Este script filtra `validation.csv`
con `configuraciones_congeladas.json` y produce tres figuras legibles:

  accuracy_congelada_vs_n.png   accuracy vs n; panel coseno (exacto, LSH, E2LSH p=2)
                                y panel L1 (exacto L1, E2LSH p=1), cada ANN junto a su exacto
  recall_congelada_vs_n.png     recall@k vs n, con la meta de la política (0,90)
  candidatos_congelada_vs_n.png fracción de la base examinada vs n, con el límite del
                                protocolo (50 %) y el del plan original de main (20 %)

Uso:
    python scripts/graficar_congeladas.py --csv results/escalabilidad/validation.csv \
        --selection results/escalabilidad/configuraciones_congeladas.json \
        --out results/escalabilidad/figuras_validacion
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Misma paleta que graficar_compromiso.py; los exactos van en tinta neutra (referencia).
COLORS = {"lsh": "#2a78d6", "e2lsh_p2": "#eb6834", "e2lsh_p1": "#1baf7a",
          "exact": "#52514e", "exact_l1": "#52514e"}
MARKERS = {"lsh": "o", "e2lsh_p2": "s", "e2lsh_p1": "D", "exact": "o", "exact_l1": "o"}
LABELS = {"lsh": "LSH coseno (SimHash por bandas)",
          "e2lsh_p2": "E2LSH p=2 (Gauss, coseno)",
          "e2lsh_p1": "E2LSH p=1 (Cauchy, L1)",
          "exact": "Exacto coseno (baseline)",
          "exact_l1": "Exacto L1 (baseline)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def frozen_rows(csv_path: Path, selection_path: Path) -> dict[str, list[tuple[int, dict]]]:
    """{método: [(n, fila_csv), ...]} con la fila de la configuración congelada."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if not r.get("error")]
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[int, dict]]] = {}
    for block in frozen["selected_by_size"]:
        n = int(block["n"])
        for entry in block["selected"]:
            c = entry["config"]
            match = [r for r in rows if int(r["n"]) == n and r["method"] == c["method"]
                     and (c["method"].startswith("exact")
                          or (int(float(r["L"])) == c["L"] and int(float(r["r"])) == c["r"]
                              and ("w_factor" not in c or abs(float(r["w_factor"]) - c["w_factor"]) < 1e-9)))]
            if len(match) != 1:
                raise SystemExit(f"No se encontró una fila única para n={n}, {c}: {len(match)} filas")
            if abs(float(match[0]["accuracy"]) - entry["validation_accuracy"]) > 1e-9:
                raise SystemExit(f"accuracy del CSV y del JSON no coinciden en n={n}, {c['method']}")
            out.setdefault(c["method"], []).append((n, match[0]))
    for method in out:
        out[method].sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True, help="validation.csv")
    ap.add_argument("--selection", type=Path, required=True, help="configuraciones_congeladas.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-recall", type=float, default=0.90)
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    except ImportError as exc:
        raise SystemExit("Instala las dependencias para gráficas: pip install -e '.[plot]'") from exc

    data = frozen_rows(args.csv, args.selection)
    out = args.out or args.csv.parent / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({n for series in data.values() for n, _ in series})
    k = next(iter(data.values()))[0][1].get("k", "k")

    def style(ax, ylabel=None):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
        ax.set_xlabel("Pacientes en la base TRAIN (n, escala log)", color=TEXT)
        if ylabel:
            ax.set_ylabel(ylabel, color=TEXT)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    end_labels: dict = {}

    def line(ax, method, key, *, label_end=True, fmt="{:.3f}"):
        xs = [n for n, _ in data[method]]
        ys = [float(r[key]) for _, r in data[method]]
        dashed = method == "exact_l1"
        ax.plot(xs, ys, color=COLORS[method], linewidth=2, linestyle="--" if dashed else "-",
                marker=MARKERS[method], markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                label=LABELS[method], zorder=3)
        if label_end:  # etiqueta directa del último punto (alivia el bajo contraste del verde)
            end_labels.setdefault(ax, []).append((xs[-1], ys[-1], fmt.format(ys[-1])))

    def place_end_labels(fig, ax, min_gap_pt=10.0):
        """Etiquetas al final de cada línea, separadas verticalmente para que no se tapen."""
        items = []
        for x, y, text in end_labels.pop(ax, []):
            y_pt = ax.transData.transform((x, y))[1] / fig.dpi * 72
            items.append([y_pt, y_pt, x, y, text])
        items.sort(key=lambda t: t[0])
        for i in range(1, len(items)):
            items[i][1] = max(items[i][1], items[i - 1][1] + min_gap_pt)
        for orig, placed, x, y, text in items:
            ax.annotate(text, (x, y), xytext=(7, placed - orig), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT)

    def finish(fig, axes, title, name, ncol=3):
        handles, labels = [], []
        for ax in axes:
            h, lab = ax.get_legend_handles_labels()
            for hh, ll in zip(h, lab):
                if ll not in labels and not ll.startswith("_"):
                    handles.append(hh)
                    labels.append(ll)
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9)
        fig.suptitle(title, color=TEXT)
        fig.tight_layout(rect=(0, 0.1, 0.97, 1))
        for ax in axes:
            place_end_labels(fig, ax)
        path = out / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(path)

    subtitle = "validación · configuración congelada por método"
    # 1) Accuracy: cada método aproximado junto a SU exacto.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for method in ("exact", "lsh", "e2lsh_p2"):
        line(axes[0], method, "accuracy")
    for method in ("exact_l1", "e2lsh_p1"):
        line(axes[1], method, "accuracy")
    axes[0].set_title("Distancia coseno", color=TEXT, fontsize=10)
    axes[1].set_title("Distancia L1", color=TEXT, fontsize=10)
    style(axes[0], "Accuracy (voto k-NN)")
    style(axes[1])
    finish(fig, axes, f"Accuracy de clasificación vs tamaño de base · {subtitle}",
           "accuracy_congelada_vs_n.png")

    # 2) Recall@k vs n.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in ("lsh", "e2lsh_p2", "e2lsh_p1"):
        line(ax, method, "recall_at_k")
    ax.axhline(args.min_recall, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.text(sizes[0], args.min_recall, f" meta recall ≥ {args.min_recall:.2f}", color=MUTED,
            fontsize=8, va="bottom")
    ax.set_ylim(0, 1.0)
    style(ax, f"Recall@{k} vs su exacto")
    finish(fig, [ax], f"Recall@{k} vs tamaño de base\n{subtitle}", "recall_congelada_vs_n.png")

    # 3) Fracción de candidatos vs n.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in ("lsh", "e2lsh_p2", "e2lsh_p1"):
        line(ax, method, "fraction_candidates", fmt="{:.1%}")
    for y, label in ((0.50, " límite protocolo correcciones (50 %)"), (0.20, " meta plan original main (20 %)")):
        ax.axhline(y, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.text(sizes[0], y, label, color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    style(ax, "Fracción de la base examinada")
    finish(fig, [ax], f"Candidatos examinados vs tamaño de base\n{subtitle}",
           "candidatos_congelada_vs_n.png")


if __name__ == "__main__":
    main()
