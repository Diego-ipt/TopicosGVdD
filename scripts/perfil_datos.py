"""Perfila los datos en streaming para decidir qué medidas de similitud aplican.

    python scripts/perfil_datos.py <carpeta> --types expression mirna ... \
        --samples 400 --max-features 20000 --max-rows N --seed 2026 --out results/perfil

Escribe ``<out>/<tipo>.json`` (y ``<tipo>_rasgos.npz`` con los estadísticos por
rasgo) y regenera ``<out>/RESUMEN.md`` con todos los JSON presentes, de modo que
varios procesos pueden correr tipos distintos en paralelo y el último regenera
el resumen completo (``--only-summary`` lo regenera sin leer datos).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcgaknn import perfil  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="carpeta con los archivos de PanCanAtlas")
    ap.add_argument("--types", nargs="+", default=list(perfil.ALL_TYPES), choices=list(perfil.ALL_TYPES))
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--max-rows", type=int, default=None, help="límite de filas (rasgos/eventos) por tipo")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/perfil")
    ap.add_argument("--progress", type=int, default=20000, help="cada cuántas filas informar (0 = nunca)")
    ap.add_argument("--only-summary", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.only_summary:
        if not a.root:
            ap.error("falta <carpeta>")
        t0 = time.perf_counter()
        labels = perfil.load_labels(a.root)
        print(f"etiquetas CDR: {len(labels):,} pacientes, {len(set(labels.values()))} tipos "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
        for t in a.types:
            t1 = time.perf_counter()
            print(f"== {t}", flush=True)
            if t == "mutations":
                info = perfil.profile_mutations(a.root, labels, n_samples=a.samples, seed=a.seed,
                                                max_rows=a.max_rows, progress=a.progress * 10)
            else:
                info, st = perfil.profile_matrix(a.root, t, labels, n_samples=a.samples,
                                                 max_features=a.max_features, max_rows=a.max_rows,
                                                 seed=a.seed, progress=a.progress)
                np.savez_compressed(out / f"{t}_rasgos.npz", **st)
            info["seg_total"] = round(time.perf_counter() - t1, 2)
            perfil.write_json(info, out / f"{t}.json")
            print(f"   {t}: listo en {info['seg_total']:.0f} s -> {out / (t + '.json')}", flush=True)
    p = perfil.write_summary(out)
    print(f"resumen: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
