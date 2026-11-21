"""Benchmark por consulta densa vs ANN con proyección, sin volver a ajustar nada."""
from __future__ import annotations

import argparse
from pathlib import Path
from tcgaknn.benchmark_pareado import benchmark_pareado


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/processed'))
    ap.add_argument('--raw', type=Path, required=True, help='Carpeta con A1/A2/A3 originales')
    ap.add_argument('--selection', type=Path, default=Path('results/escalabilidad/configuraciones_congeladas.json'))
    ap.add_argument('--out', type=Path, default=Path('results/escalabilidad/pareado_test.csv'))
    ap.add_argument('--phase', choices=['val', 'test'], default='test')
    ap.add_argument('--queries', type=int, default=0, help='0=todas; usar 32 para prueba rápida')
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--max-dense-mib', type=float, default=1024.0,
                    help='Límite estimado RAM; el benchmark denso es OPCIONAL')
    args = ap.parse_args()
    rows = benchmark_pareado(args.data, args.raw, args.selection, args.out,
                            phase=args.phase, queries=args.queries, seed=args.seed,
                            max_dense_mib=args.max_dense_mib)
    print(f'OK: {len(rows)} filas, {args.out}')


if __name__ == '__main__':
    main()
