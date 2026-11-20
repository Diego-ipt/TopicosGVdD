"""python scripts/preparar_cohorte.py /ruta/gdc --out data/processed"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcgaknn.preparacion import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara cohorte validada A1+A2+A3 y splits por paciente")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--transform", choices=["as_is", "log2clip"], default="as_is")
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--strict-annotations", action="store_true")
    ap.add_argument("--max-genes", type=int, default=None,
                    help="SOLO depuración: limita genes en orden A1; no usar como selección de rasgos científica")
    args = ap.parse_args()
    result = prepare(args.root, args.out, seed=args.seed, transform=args.transform,
                     zscore=not args.no_zscore,
                     check_annotations=args.strict_annotations, max_genes=args.max_genes)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
