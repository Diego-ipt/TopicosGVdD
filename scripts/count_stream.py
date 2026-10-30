"""Consume el stream completo de uno o más tipos de dato y solo cuenta.

    python scripts/count_stream.py <carpeta> [--types expression mirna ...] [--progress 50000]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcgaknn.stream import StreamService  # noqa: E402


def peak_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20


def count_type(root: Path, data_type: str, progress: int) -> dict:
    kinds: Counter = Counter()
    values = na = ncols = 0
    t0 = time.perf_counter()
    for rec in StreamService(root, data_types=[data_type]):
        kinds[rec.kind] += 1
        if rec.kind == "schema":
            ncols = len(rec.fields["columns"])
        elif rec.values is not None:
            values += rec.values.size
            na += int(np.isnan(rec.values).sum())
        n = sum(v for k, v in kinds.items() if k not in ("schema", "end"))
        if progress and n and n % progress == 0 and rec.kind not in ("schema", "end"):
            print(f"  .. {data_type}: {n:,} registros, {time.perf_counter()-t0:,.0f} s", flush=True)
    dt = time.perf_counter() - t0
    n = sum(v for k, v in kinds.items() if k not in ("schema", "end"))
    return dict(tipo=data_type, registros=n, columnas=ncols, valores=values, na=na,
                seg=dt, reg_s=n / dt if dt else 0, mib=peak_mib())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--progress", type=int, default=0)
    a = ap.parse_args()
    types = a.types or [s.data_type for s in StreamService(a.root).sources]
    for t in types:
        r = count_type(a.root, t, a.progress)
        pct = 100 * r["na"] / r["valores"] if r["valores"] else 0
        print(f"{r['tipo']:<22} registros={r['registros']:>10,}  columnas={r['columnas']:>6,}  "
              f"valores={r['valores']:>15,}  NA={r['na']:>12,} ({pct:.2f} %)  "
              f"{r['seg']:>7,.1f} s  {r['reg_s']:>9,.0f} reg/s  RAM pico {r['mib']:,.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
