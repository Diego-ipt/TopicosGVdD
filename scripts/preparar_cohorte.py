"""Prepara una cohorte A1+A2+A3 en una pasada sobre la matriz RNA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tcgaknn.preparacion_streaming import DEFAULT_TRANSFORM, prepare_streaming


def main() -> None:
    ap = argparse.ArgumentParser(description="Cohorte validada TCGA para experimentos streaming")
    ap.add_argument("root", type=Path, help="Carpeta con A1, A2 y A3")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--transform", choices=["log2clip", "as_is"], default=DEFAULT_TRANSFORM)
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--strict-annotations", action="store_true")
    ap.add_argument("--cols", type=int, default=512, help="Proyecciones para LSH coseno")
    ap.add_argument("--lp-cols", type=int, default=256, help="Proyecciones por distribución p-estable")
    ap.add_argument("--queries-per-phase", type=int, default=0,
                    help="0 = todas (predeterminado); N > 0 limita consultas solo para desarrollo")
    ap.add_argument("--max-genes", type=int, default=None, help="Solo desarrollo: genes iniciales")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    meta = prepare_streaming(args.root, args.out, seed=args.seed, transform=args.transform,
                             zscore=not args.no_zscore, cols=args.cols, lp_cols=args.lp_cols,
                             queries_per_phase=args.queries_per_phase, max_genes=args.max_genes,
                             batch_size=args.batch_size, check_annotations=args.strict_annotations,
                             log=lambda s: print(s, flush=True))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
