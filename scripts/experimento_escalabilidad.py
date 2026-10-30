"""Experimento de escalabilidad sobre el stream con una parte de los datos.

    python scripts/experimento_escalabilidad.py <carpeta_gdc> [--g-max 5000] [--n-max 4000]
        [--queries 200] [--sizes 250 500 1000 2000 4000] [--threads 1] [--sin-mutaciones]
        [--out results/escalabilidad] [--fig results/figures]

Por defecto fija 1 hilo de BLAS (antes de importar NumPy) para comparar algoritmos
y no paralelismo. Ver ``tcgaknn.escalabilidad``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--g-max", type=int, default=5000)
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sin-mutaciones", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/escalabilidad"))
    ap.add_argument("--fig", type=Path, default=Path("results/figures"))
    a = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tcgaknn.escalabilidad import run

    res = run(a.root, a.out, g_max=a.g_max, n_max=a.n_max, n_queries=a.queries, sizes=a.sizes,
              seed=a.seed, do_mutations=not a.sin_mutaciones, log=lambda m: print(m, flush=True))
    plot(res, a.fig)
    print_table(res)


def print_table(res: dict) -> None:
    for key in ("expresion", "mutaciones"):
        rows = res.get(key) or []
        if not rows:
            continue
        print(f"\n== {key} ==")
        print(f"{'método':<30}{'n':>6}{'recall@10':>11}{'cand.':>8}{'ms/consulta':>13}{'B/paciente':>12}{'pureza@5':>10}")
        for r in rows:
            print(f"{r['metodo']:<30}{r['n']:>6}{r['recall_at_k']:>11.3f}{r['frac_candidatos']:>8.2f}"
                  f"{r['t_consulta_ms_mediana']:>13.3f}{r['bytes_por_paciente']:>12.0f}{r['pureza_at_5']:>10.3f}")


def plot(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    p = res["parametros"]
    pasada = res.get("expresion_pasada", {})
    for key, titulo in (("expresion", "Expresión (coseno; E2LSH-L1 contra L1)"), ("mutaciones", "Mutaciones (Jaccard)")):
        rows = res.get(key) or []
        if not rows:
            continue
        methods = list(dict.fromkeys(r["metodo"] for r in rows))
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
        for m in methods:
            rr = [r for r in rows if r["metodo"] == m]
            n = [r["n"] for r in rr]
            style = dict(marker="o", lw=2.2 if m.startswith(("EX", "Jaccard exacto")) else 1.4,
                         ls="--" if m.startswith(("EX", "Jaccard exacto")) else "-")
            ax[0].plot(n, [r["t_consulta_ms_mediana"] for r in rr], label=m, **style)
            ax[1].plot(n, [r["recall_at_k"] for r in rr], label=m, **style)
            ax[2].plot(n, [r["frac_candidatos"] for r in rr], label=m, **style)
        ax[0].set(xscale="log", yscale="log", xlabel="n (pacientes en la base)", ylabel="ms por consulta (mediana)",
                  title="Tiempo por consulta vs n")
        ax[1].set(xscale="log", xlabel="n (pacientes en la base)", ylabel="recall@10 frente al exacto",
                  title="Calidad vs n", ylim=(0, 1.02))
        ax[2].set(xscale="log", xlabel="n (pacientes en la base)", ylabel="fracción de la base examinada",
                  title="Candidatos examinados", ylim=(0, 1.05))
        for a_ in ax:
            a_.grid(alpha=0.3, which="both")
        ax[1].legend(fontsize=8, loc="lower left")
        nota = (f"{titulo}. Datos parciales: {pasada.get('genes_usados', '?')} genes de 20.531 (expresión), "
                f"{p['n_queries']} consultas fijas, 1 hilo de BLAS, consulta a consulta.")
        fig.suptitle(f"Escalabilidad sobre el stream — {titulo}", fontsize=13)
        fig.text(0.01, -0.02, nota, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"escalabilidad_{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    rows = res.get("expresion") or []
    if rows:
        nmax = max(r["n"] for r in rows)
        rr = [r for r in rows if r["n"] == nmax]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        for r in rr:
            ax[0].scatter(r["t_consulta_ms_mediana"], r["recall_at_k"], s=60)
            ax[0].annotate(r["metodo"], (r["t_consulta_ms_mediana"], r["recall_at_k"]), fontsize=8,
                           xytext=(4, 4), textcoords="offset points")
        ax[0].set(xscale="log", xlabel="ms por consulta (mediana)", ylabel="recall@10",
                  title=f"Compromiso calidad / tiempo (n = {nmax:,})".replace(",", "."))
        ax[0].grid(alpha=0.3, which="both")
        names = [r["metodo"] for r in rr]
        ax[1].barh(names, [r["bytes_por_paciente"] for r in rr], color="#4f81bd")
        ax[1].set_xscale("log")
        ax[1].set(xlabel="bytes del índice por paciente (escala log)", title="Memoria del índice")
        for i, r in enumerate(rr):
            ax[1].text(r["bytes_por_paciente"], i, f" {r['bytes_por_paciente']:,.0f} B".replace(",", "."), va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "escalabilidad_compromiso.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
