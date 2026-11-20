"""Validar y congelar; luego evaluar el TEST final UNA sola vez."""
from __future__ import annotations

import argparse
from pathlib import Path

from tcgaknn.experimento_validado import test, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["validar", "test"])
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results/protocolo_validado"))
    ap.add_argument("--metric", choices=["cosine", "l2", "l1"], default="cosine")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Ejemplo: --methods rp simhash simhash_rp lsh; 'exact' se incluye siempre")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--verify-hash", action="store_true")
    args = ap.parse_args()
    if args.fase == "validar":
        result = validate(args.data, args.out, metric=args.metric, k=args.k,
                          methods=set(args.methods) if args.methods else None,
                          seed=args.seed, verify_hash=args.verify_hash)
    else:
        selected = args.selection or (args.out / "configuraciones_congeladas.json")
        result = test(args.data, selected, args.out, verify_hash=args.verify_hash)
    print(f"OK: fase={result['phase']}, resultados={len(result['results'])}; carpeta={args.out}")


if __name__ == "__main__":
    main()
