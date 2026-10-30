"""Count-Min sketch: frecuencias aproximadas que nunca subestiman (Cormode y Muthukrishnan 2005)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

from ._hash import M64, hash64, hash64_many


class CountMinSketch:
    """Count-Min (Cormode y Muthukrishnan 2005, *J. Algorithms* 55(1):58–75).

    Matriz de ``depth`` × ``width`` contadores. Cada fila i tiene su función
    h_i: add(x, c) suma c en C[i, h_i(x)] para toda fila; ``estimate(x)`` =
    min_i C[i, h_i(x)]. Con conteos no negativos, f(x) ≤ estimado y, con
    w = ⌈e/ε⌉ y d = ⌈ln(1/δ)⌉, P[estimado > f(x) + ε·N] ≤ δ, donde N es la
    suma de todos los conteos (``from_error``).

    Funciones por fila: el ítem se lleva a una clave de 64 bits con ``hash64``
    y luego h_i(x) = ⌊(((a_i·key + b_i) mod 2^64) >> 32) · w / 2^32⌋, con a_i
    impar y b_i aleatorios (multiply-add-shift de Dietzfelbinger, más reducción
    al rango [0, w) por multiplicación). La cota usa solo independencia por
    pares; esta familia es un sustituto práctico estándar.

    ``conservative=True`` activa la actualización conservadora (Estan y Varghese
    2002): solo sube cada celda hasta max(C[i, h_i(x)], estimado + c). Sigue sin
    subestimar y el error baja, pero exige c ≥ 0 y no se puede vectorizar.
    """

    def __init__(self, width: int, depth: int, seed: int = 0, *, conservative: bool = False):
        if width < 1 or depth < 1:
            raise ValueError("width y depth deben ser ≥ 1")
        if width >= 1 << 32:
            raise ValueError("width debe ser < 2^32")
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        self.conservative = conservative
        rng = np.random.default_rng([0xC0, seed & M64])
        self._a = rng.integers(0, 1 << 63, size=depth, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self._b = rng.integers(0, 1 << 63, size=depth, dtype=np.uint64)
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total = 0
        self._rows = np.arange(depth)

    @classmethod
    def from_error(cls, eps: float, delta: float, seed: int = 0, *, conservative: bool = False) -> "CountMinSketch":
        """Dimensiona con w = ⌈e/ε⌉ y d = ⌈ln(1/δ)⌉ (Cormode y Muthukrishnan 2005, Thm. 1)."""
        if not (0 < eps < 1 and 0 < delta < 1):
            raise ValueError("eps y delta deben estar en (0, 1)")
        return cls(math.ceil(math.e / eps), math.ceil(math.log(1.0 / delta)), seed, conservative=conservative)

    @property
    def eps(self) -> float:
        return math.e / self.width

    @property
    def delta(self) -> float:
        return math.exp(-self.depth)

    def _cols(self, keys: np.ndarray) -> np.ndarray:
        """Columnas (depth × len(keys)) para claves de 64 bits."""
        k = np.asarray(keys, dtype=np.uint64)[None, :]
        hi = (self._a[:, None] * k + self._b[:, None]) >> np.uint64(32)
        return ((hi * np.uint64(self.width)) >> np.uint64(32)).astype(np.int64)

    def _key(self, item: Hashable) -> np.ndarray:
        return np.array([hash64(item, self.seed)], dtype=np.uint64)

    def add(self, item: Hashable, count: int = 1) -> None:
        cols = self._cols(self._key(item))[:, 0]
        if self.conservative:
            if count < 0:
                raise ValueError("la actualización conservadora exige count ≥ 0")
            cur = self.table[self._rows, cols]
            self.table[self._rows, cols] = np.maximum(cur, cur.min() + count)
        else:
            self.table[self._rows, cols] += count
        self.total += int(count)

    def add_many(self, items: Iterable[Hashable] | np.ndarray, counts: Sequence[int] | np.ndarray | None = None) -> None:
        """Agrega muchos ítems; vectorizado (``np.add.at``) salvo en modo conservador."""
        keys = hash64_many(items if isinstance(items, np.ndarray) else list(items), self.seed)
        c = np.ones(keys.size, dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
        if c.shape != keys.shape:
            raise ValueError("counts debe tener el mismo largo que items")
        if self.conservative:
            if (c < 0).any():
                raise ValueError("la actualización conservadora exige count ≥ 0")
            cols = self._cols(keys)
            for j in range(keys.size):  # el orden importa: secuencial
                cur = self.table[self._rows, cols[:, j]]
                self.table[self._rows, cols[:, j]] = np.maximum(cur, cur.min() + c[j])
        else:
            for i, col in enumerate(self._cols(keys)):
                np.add.at(self.table[i], col, c)
        self.total += int(c.sum())

    def estimate(self, item: Hashable) -> int:
        cols = self._cols(self._key(item))[:, 0]
        return int(self.table[self._rows, cols].min())

    def estimate_many(self, items: Iterable[Hashable] | np.ndarray) -> np.ndarray:
        keys = hash64_many(items if isinstance(items, np.ndarray) else list(items), self.seed)
        cols = self._cols(keys)
        return np.take_along_axis(self.table, cols, axis=1).min(axis=0)

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Suma celda a celda (mismo ancho, profundidad y semilla). Modifica ``self``."""
        if (self.width, self.depth, self.seed) != (other.width, other.depth, other.seed):
            raise ValueError("solo se pueden combinar sketches con igual width, depth y seed")
        self.table += other.table
        self.total += other.total
        return self

    def heavy_hitters(
        self,
        candidates: Iterable[Hashable],
        *,
        phi: float | None = None,
        threshold: float | None = None,
    ) -> list[tuple[Hashable, int]]:
        """Candidatos con estimado ≥ umbral (``threshold`` absoluto o ``phi``·N).

        Count-Min no guarda los ítems: hay que darle la lista de candidatos (p.
        ej. todos los genes del esquema). Como nunca subestima, todo ítem con
        f ≥ umbral aparece (sin falsos negativos); los falsos positivos tienen
        f ≥ umbral − ε·N con probabilidad ≥ 1 − δ. Orden: estimado descendente.
        """
        if (phi is None) == (threshold is None):
            raise ValueError("indique exactamente uno de phi o threshold")
        thr = phi * self.total if phi is not None else threshold
        cand = list(dict.fromkeys(candidates))
        if not cand:
            return []
        est = self.estimate_many(cand)
        out = [(c, int(e)) for c, e in zip(cand, est) if e >= thr]
        out.sort(key=lambda t: -t[1])
        return out
