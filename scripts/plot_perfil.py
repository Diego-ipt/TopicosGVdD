"""Gráficos del perfilado de datos (results/perfil/*.json, *_rasgos.npz).

    python scripts/plot_perfil.py [--perfil results/perfil] [--out results/figures]

Genera:
- perfil_pureza.png: pureza@5 del k-NN por medida × preprocesamiento, una ómica por panel.
- perfil_hubness.png: hubness (asimetría de k-ocurrencias, k = 10), mismo formato.
- perfil_distribuciones.png: distribución de la media y la varianza por rasgo (tras la transformación).
- perfil_mutaciones.png: genes mutados por paciente (mediana por tipo) y genes más frecuentes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OMICAS = ["expression", "mirna", "rppa", "paradigm", "methylation_450k", "methylation_27k_450k", "mutations"]
TITULOS = {
    "expression": "Expresión (RNA-Seq)",
    "mirna": "miRNA",
    "rppa": "RPPA (proteínas)",
    "paradigm": "PARADIGM (vías)",
    "methylation_450k": "Metilación 450K*",
    "methylation_27k_450k": "Metilación 27K+450K*",
    "mutations": "Mutaciones (conjuntos)",
}


def load(perfil: Path) -> dict[str, dict]:
    out = {}
    for t in OMICAS:
        p = perfil / f"{t}.json"
        if p.exists():
            out[t] = json.loads(p.read_text(encoding="utf-8"))
    return out


def metric_table(d: dict, key: str, sub: str | None = None) -> tuple[list[str], list[str], np.ndarray]:
    """Matriz medida × variante con el valor ``key`` (o ``key[sub]``)."""
    variantes = list(d["analisis"]["variantes"])
    metricas: list[str] = []
    for v in variantes:
        for m in d["analisis"]["variantes"][v]["metricas"]:
            if m not in metricas:
                metricas.append(m)
    M = np.full((len(metricas), len(variantes)), np.nan)
    for j, v in enumerate(variantes):
        for i, m in enumerate(metricas):
            r = d["analisis"]["variantes"][v]["metricas"].get(m)
            if r is not None:
                val = r[key] if sub is None else r[key][sub]
                M[i, j] = val
    return metricas, variantes, M


def heatmap_grid(data: dict[str, dict], key: str, sub: str | None, title: str, cmap: str,
                 out: Path, fmt: str, vmin: float, vmax: float, note: str) -> None:
    tipos = [t for t in OMICAS if t in data]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.subplots_adjust(wspace=0.45, hspace=0.45)
    for ax, t in zip(axes.flat, tipos):
        metricas, variantes, M = metric_table(data[t], key, sub)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(variantes)), [v.replace("transformado", "transf.") for v in variantes],
                      rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(metricas)), metricas, fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    frac = (M[i, j] - vmin) / (vmax - vmin)
                    ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=7.5,
                            color="white" if frac > 0.6 or frac < 0.15 and cmap == "viridis" else "black")
        ax.set_title(TITULOS[t], fontsize=11)
    for ax in axes.flat[len(tipos):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(title)
    fig.suptitle(title, fontsize=14)
    fig.text(0.01, 0.01, note, fontsize=8.5)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribuciones(perfil: Path, data: dict[str, dict], out: Path) -> None:
    tipos = [t for t in OMICAS if t != "mutations" and (perfil / f"{t}_rasgos.npz").exists()]
    fig, axes = plt.subplots(2, len(tipos), figsize=(3.6 * len(tipos), 7))
    for c, t in enumerate(tipos):
        z = np.load(perfil / f"{t}_rasgos.npz")
        mean = z["media_t"] if "media_t" in z else z["media"]
        var = z["varianza_t"] if "varianza_t" in z else z["varianza"]
        mean = mean[np.isfinite(mean)]
        var = var[np.isfinite(var) & (var > 0)]
        transf = data[t].get("transformacion") or "ninguna"
        axes[0, c].hist(mean, bins=60, color="#3b6ea5")
        axes[0, c].set_title(f"{TITULOS[t]}\n(transf.: {transf})", fontsize=9.5)
        axes[0, c].set_xlabel("media por rasgo")
        axes[1, c].hist(np.log10(var), bins=60, color="#c0504d")
        axes[1, c].set_xlabel("log10 varianza por rasgo")
        if c == 0:
            axes[0, c].set_ylabel("nº de rasgos")
            axes[1, c].set_ylabel("nº de rasgos")
    fig.suptitle("Cómo son los datos: distribución por rasgo (gen / sonda / miRNA / proteína / vía)", fontsize=13)
    fig.text(0.01, -0.01, "* Metilación: solo el prefijo del archivo (25.000 y 18.000 sondas). "
             "Las medias de metilación son M-values (logit de beta).", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mutaciones(d: dict, out: Path) -> None:
    tam = d["tamanos_todos_los_pacientes"]
    med = tam["mediana_genes_ns_por_tipo"]
    tipos = sorted(med, key=med.get)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.barh(tipos, [med[t] for t in tipos], color="#4f81bd")
    a1.set_xscale("log")
    a1.set_xlabel("mediana de genes con mutación no silenciosa por paciente (escala log)")
    a1.set_title("Carga mutacional por tipo de cáncer")
    q = tam["genes_no_silenciosas"]
    a1.text(0.98, 0.02,
            f"Todos los pacientes (n = {tam['n_pacientes']:,}):\n"
            f"mediana {q['q50']:.0f}, p95 {q['q95']:.0f}, máx. {q['q100']:.0f}\n"
            f"{100 * tam['frac_ge_500_genes_ns']:.1f} % con ≥ 500 genes\n"
            f"η² log(tamaño) vs tipo = {tam['eta2_log_tamano_vs_tipo']:.2f}".replace(",", "."),
            transform=a1.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    genes = d["genes_mas_frecuentes_ns"][:25]
    names = [g["gen"] if isinstance(g, dict) and "gen" in g else (g[0] if isinstance(g, (list, tuple)) else str(g)) for g in genes]
    vals = [g.get("pacientes", g.get("n", 0)) if isinstance(g, dict) else (g[1] if isinstance(g, (list, tuple)) else 0) for g in genes]
    a2.barh(names[::-1], vals[::-1], color="#9bbb59")
    a2.set_xlabel("pacientes con mutación no silenciosa")
    a2.set_title("Genes más mutados (top 25)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfil", type=Path, default=Path("results/perfil"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = load(a.perfil)
    azar = np.mean([d["analisis"]["pureza_azar"] for d in data.values()])
    note = (f"S = 400 pacientes estratificados por tipo (33 tipos); pureza al azar ≈ {azar:.3f}. "
            "* Metilación: solo el prefijo del archivo. Mutaciones: variantes = todas / no silenciosas / genes frecuentes.")
    heatmap_grid(data, "pureza@5", None, "Pureza@5 del k-NN (fracción de los 5 vecinos con el mismo tipo de cáncer)",
                 "viridis", a.out / "perfil_pureza.png", ".2f", 0.0, 0.85, note)
    heatmap_grid(data, "hubness", "asimetria_Nk", "Hubness (asimetría de k-ocurrencias, k = 10; más alto = peor)",
                 "magma_r", a.out / "perfil_hubness.png", ".1f", 0.0, 8.0, note)
    plot_distribuciones(a.perfil, data, a.out / "perfil_distribuciones.png")
    if "mutations" in data:
        plot_mutaciones(data["mutations"], a.out / "perfil_mutaciones.png")
    for p in sorted(a.out.glob("perfil_*.png")):
        print(p)


if __name__ == "__main__":
    main()
