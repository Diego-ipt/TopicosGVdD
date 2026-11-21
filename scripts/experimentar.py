"""Validación con ajuste de w; test final con configuraciones congeladas."""
from __future__ import annotations
import argparse
from pathlib import Path
from tcgaknn.experimento_streaming import final_test_streaming, validate_streaming


def main() -> None:
    ap = argparse.ArgumentParser(description="LSH/E2LSH/kNN exacto con bases anidadas en streaming")
    ap.add_argument("fase", choices=["validar", "test"])
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results/escalabilidad"))
    ap.add_argument("--k", type=int, default=10, help="Solo validación (se congela para test)")
    ap.add_argument("--replicas", type=int, default=1, help="Bloques independientes de proyecciones; solo validación")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="n crecientes (train)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="lsh e2lsh_p1 e2lsh_p2 exact; exact incluido siempre")
    ap.add_argument("--seed", type=int, default=None, help="Debe coincidir con preparación")
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--verify-hash", action="store_true")
    ap.add_argument("--exact-benchmark-raw", type=Path, default=None,
                    help="Opcional: relee A1 en streaming por n y métrica para medir exacto real; costoso")
    args = ap.parse_args()
    if args.fase == "validar":
        data = validate_streaming(args.data, args.out, k=args.k, sizes=args.sizes,
                                  methods=set(args.methods) if args.methods else None,
                                  seed=args.seed, verify_hash=args.verify_hash, replicas=args.replicas,
                                  exact_benchmark_raw=args.exact_benchmark_raw)
    else:
        data = final_test_streaming(args.data,
                                    args.selection or args.out / "configuraciones_congeladas.json",
                                    args.out, verify_hash=args.verify_hash,
                                    exact_benchmark_raw=args.exact_benchmark_raw)
    print(f"OK: {data['phase']} · {len(data['results'])} tamaños · {args.out}")


if __name__ == "__main__":
    main()
