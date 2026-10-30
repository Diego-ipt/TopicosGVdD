"""Hash estable de 64 bits compartido por los sketches.

Los enteros pasan por el finalizador de SplitMix64 (vectorizable con NumPy) y el
resto (str, bytes) por BLAKE2b con la semilla como clave. No se usa ``hash()`` de
Python porque cambia entre procesos (PYTHONHASHSEED).
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Iterable

import numpy as np


M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


def splitmix64(x: int) -> int:
    """Finalizador de SplitMix64 (Steele, Lea y Flood 2014) sobre un entero de 64 bits."""
    z = (x + _GOLDEN) & M64
    z = ((z ^ (z >> 30)) * _MIX1) & M64
    z = ((z ^ (z >> 27)) * _MIX2) & M64
    return z ^ (z >> 31)


def splitmix64_array(x: np.ndarray) -> np.ndarray:
    """``splitmix64`` vectorizado (uint64; la aritmética de NumPy envuelve módulo 2^64)."""
    z = np.asarray(x, dtype=np.uint64) + np.uint64(_GOLDEN)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_MIX1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_MIX2)
    return z ^ (z >> np.uint64(31))


def _seed_key(seed: int) -> int:
    return splitmix64(seed & M64)


def hash64(item: Hashable, seed: int = 0) -> int:
    """Hash de 64 bits estable entre procesos.

    - Enteros (``int`` o ``np.integer``): ``splitmix64((x mod 2^64) XOR s)`` con
      s = splitmix64(seed). Coincide con ``hash64_many`` sobre arreglos enteros.
    - ``str`` (UTF-8) y ``bytes``: BLAKE2b de 8 bytes con la semilla como clave.
    - Otros: BLAKE2b de ``repr(item)`` (estable para tuplas de str/int/float).
    - Escalares de NumPy se llevan antes a su equivalente de Python (``.item()``)
      para que ``np.float64(1.5)`` y ``1.5`` den el mismo hash (``repr`` difiere).
    """
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, (int, np.integer)):
        return splitmix64((int(item) & M64) ^ _seed_key(seed))
    if isinstance(item, str):
        data = item.encode("utf-8")
    elif isinstance(item, (bytes, bytearray, memoryview)):
        data = bytes(item)
    else:
        data = repr(item).encode("utf-8")
    key = (seed & M64).to_bytes(8, "little")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, key=key).digest(), "little")


def hash64_many(items: Iterable[Hashable] | np.ndarray, seed: int = 0) -> np.ndarray:
    """Hash de 64 bits de muchos ítems (uint64). Vectorizado si llega un arreglo entero."""
    if isinstance(items, np.ndarray) and np.issubdtype(items.dtype, np.integer):
        x = items.ravel().astype(np.int64, copy=False).view(np.uint64)
        return splitmix64_array(x ^ np.uint64(_seed_key(seed)))
    return np.fromiter((hash64(it, seed) for it in items), dtype=np.uint64)


def _bit_length_u64(v: np.ndarray) -> np.ndarray:
    """Largo en bits exacto de cada uint64 (0 para 0), por búsqueda binaria de desplazamientos."""
    v = v.copy()
    n = np.zeros(v.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = v >= (np.uint64(1) << np.uint64(s))
        v[big] >>= np.uint64(s)
        n[big] += s
    return n + (v > 0)
