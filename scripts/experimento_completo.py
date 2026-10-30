"""Expresión completa por stream con SimHash + re-ranking RP: F1 y error estimado vs real.

    python scripts/experimento_completo.py <carpeta_gdc> [--queries 500] [--b 256] [--m 256] [--f 10]
        [--threads 1] [--out results/completo_expresion] [--fig results/figures]

Ver ``tcgaknn.completo``. Fija 1 hilo de BLAS antes de importar NumPy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--b", type=int, default=256)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--f", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/completo_expresion"))
    ap.add_argument("--fig", type=Path, default=Path("results/figures"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows (cp1252) no imprime Δ ni −
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tcgaknn.completo import run

    res = run(a.root, a.out, n_queries=a.queries, b=a.b, m=a.m, f=a.f, seed=a.seed,
              log=lambda s: print(s, flush=True))
    plot(a.out, a.fig, res)
    report(res)


def report(res: dict) -> None:
    p, s = res["pasada"], res["busqueda"]
    kp = res["k_principal"]
    print(f"\n{res['metodo']} — predicción del tipo de cáncer por voto k-NN")
    print(f"Pasada: {p['genes_leidos']:,} genes, {p['pacientes']:,} pacientes ({p['consultas']} consultas), "
          f"{p['segundos']:,.0f} s; pico {p['memoria_pico_proceso_MiB']:,.0f} MiB")
    ref = res["clasificacion"]["referencias"]
    print(f"Referencias: clase mayoritaria exactitud {ref['clase_mayoritaria']['accuracy']:.3f}; "
          f"azar estratificado {ref['azar_estratificado_exactitud_esperada']:.3f}")
    print(f"{'k':>4}{'':>3}{'exactitud':>10}{'error':>8}{'F1-macro':>10}{'F1-pond.':>10}{'exact.bal.':>11}   acuerdo")
    for key, v in res["clasificacion"].items():
        if key == "referencias":
            continue
        for who in ("exacto", "aproximado"):
            r = v[who]
            print(f"{key:>4} {who[:5]:>5}{r['accuracy']:>9.3f}{r['tasa_de_error']:>8.3f}{r['f1_macro']:>10.3f}"
                  f"{r['f1_ponderado']:>10.3f}{r['balanced_accuracy']:>11.3f}" + (f"   {v['acuerdo_de_voto']:.3f}" if who == "aproximado" else ""))
        if "delta_f1_macro_IC95" in v:
            print(f"     Δ (aprox − exacto) IC95: exactitud {v['delta_exactitud_IC95']}, F1-macro {v['delta_f1_macro_IC95']}")
    peor = sorted(res["por_clase"], key=lambda r: r["f1_aproximado"])[:6]
    print(f"Tipos con peor F1 (k = {kp}): " + ", ".join(f"{r['tipo']} {r['f1_aproximado']:.2f} (exacto {r['f1_exacto']:.2f})" for r in peor))
    print("Confusiones principales (aprox.): " + ", ".join(f"{a}→{b} {c:.2f}" for a, b, c in res["confusiones"]["aproximado"][:6]))
    print("\nDiagnóstico de la búsqueda (subproducto): " + ", ".join(f"{k} {v:.3f}" for k, v in s.items() if k.startswith("recall"))
          + f"; {s['t_consulta_ms_mediana']:.2f} ms/consulta; {s['bytes_por_paciente']:,.0f} B/paciente")
    for k, v in res["error_estimacion"].items():
        print(f"  error coseno {k}: {v}")
    print("Escalamiento:", [(e["n"], round(e[f"recall@{res['parametros']['k']}"], 3), round(e["t_consulta_ms_mediana"], 3)) for e in res["escalamiento"]])


def plot(out: Path, fig_dir: Path, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(out / "estimado_vs_real.npz")
    real = z["real"].ravel()
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    for a_, key, name in ((ax[0], "simhash", f"SimHash-{res['parametros']['b']}: cos(π·Ham/b)"),
                          (ax[1], "rp", f"RP-{res['parametros']['m']}: producto punto")):
        est = z[key].ravel()
        hb = a_.hexbin(real, est, gridsize=80, bins="log", cmap="viridis", mincnt=1)
        lo, hi = float(min(real.min(), est.min())), float(max(real.max(), est.max()))
        a_.plot([lo, hi], [lo, hi], "r--", lw=1)
        e = res["error_estimacion"][f"{key}_todos"]
        a_.set(xlabel="coseno real", ylabel="coseno estimado", title=name)
        a_.text(0.02, 0.97, f"MAE {e['mae']:.3f}\nRMSE {e['rmse']:.3f}\nsesgo {e['sesgo']:+.4f}\nr = {e['correlacion']:.3f}".replace(".", ","),
                transform=a_.transAxes, va="top", fontsize=9, bbox=dict(fc="white", ec="0.7"))
        fig.colorbar(hb, ax=a_, label="pares (log)")
    for key, color in (("simhash", "#3b6ea5"), ("rp", "#c0504d")):
        ax[2].hist((z[key].ravel() - real), bins=120, histtype="step", lw=1.6, color=color, density=True,
                   label=key.replace("simhash", "SimHash").replace("rp", "RP"))
    ax[2].set(xlabel="error = estimado − real", ylabel="densidad", title="Distribución del error (todos los pares Q × B)")
    ax[2].legend()
    fig.suptitle("Error de estimación del coseno: sketch vs real (expresión completa)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "completo_error_estimacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pc = sorted(res["por_clase"], key=lambda r: r["f1_exacto"])
    fig, ax = plt.subplots(figsize=(10, 9))
    yy = np.arange(len(pc))
    ax.barh(yy - 0.2, [r["f1_exacto"] for r in pc], height=0.4, label="exacto", color="#7f7f7f")
    ax.barh(yy + 0.2, [r["f1_aproximado"] for r in pc], height=0.4, label=res["metodo"], color="#3b6ea5")
    ax.set_yticks(yy, [f"{r['tipo']} ({r['consultas']})" for r in pc], fontsize=8)
    c5 = res["clasificacion"][f"k={res['k_principal']}"]
    ax.set(xlabel=f"F1 por clase (voto k = {res['k_principal']})", xlim=(0, 1.02),
           title=f"F1 por tipo de cáncer — F1-macro exacto {c5['exacto']['f1_macro']:.3f} vs aproximado "
                 f"{c5['aproximado']['f1_macro']:.3f}".replace(".", ","))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "completo_f1_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cz = np.load(out / "confusion.npz", allow_pickle=True)
    labs = [str(x) for x in cz["labels"]]
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    for a_, key, name in ((ax[0], "confusion_exacto", "k-NN exacto"), (ax[1], "confusion_aproximado", res["metodo"])):
        cm = cz[key].astype(float)
        cmn = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
        im = a_.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        a_.set_xticks(range(len(labs)), labs, rotation=90, fontsize=7)
        a_.set_yticks(range(len(labs)), labs, fontsize=7)
        a_.set(xlabel="tipo predicho", ylabel="tipo real", title=f"Matriz de confusión normalizada por fila — {name}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="fracción de la clase real")
    fig.savefig(fig_dir / "completo_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
