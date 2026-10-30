"""KLL: cuantiles aproximados en streaming (Karnin, Lang y Liberty 2016)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ._hash import M64


class KLLSketch:
    """Cuantiles en streaming (Karnin, Lang y Liberty 2016, FOCS; arXiv:1603.05346).

    Pila de compactores; un ítem en el nivel h pesa 2^h. La capacidad del nivel
    h con H niveles es ⌈k·c^(H−h−1)⌉ + 1 con c = 2/3 (los niveles altos tienen
    capacidad ≈ k y los bajos decaen geométricamente). Cuando el total supera
    la suma de capacidades se compacta el primer nivel lleno: se ordena, se
    elige al azar desplazamiento 0 o 1 y se promueven al nivel h+1 los ítems en
    posiciones desplazamiento, desplazamiento+2, … (si hay un número impar, el
    mayor se queda). Con ``alternate`` las compactaciones impares de cada nivel
    usan el desplazamiento opuesto al anterior, lo que reduce la varianza.
    Sigue la implementación de referencia de E. Liberty
    (github.com/edoliberty/streaming-quantiles, ``kll.py``, Apache-2.0).

    Error de rango normalizado O(1/k); en DataSketches, k = 200 da ≈ 1,33 %
    (un lado) y ≈ 1,65 % (dos lados). Memoria O(k) más O(log(N/k)) niveles.

    ``rank(x)`` = fracción estimada de datos ≤ x; ``quantile(q)`` = menor ítem
    guardado cuyo rango acumulado es ≥ q (q = 0 y q = 1 dan el mínimo y el
    máximo exactos). Los NaN se ignoran.
    """

    def __init__(self, k: int = 200, seed: int = 0, *, c: float = 2.0 / 3.0, alternate: bool = True):
        if k < 2:
            raise ValueError("k debe ser ≥ 2")
        self.k = int(k)
        self.c = float(c)
        self.alternate = alternate
        self._rng = np.random.default_rng([0x4B11, seed & M64])
        self.levels: list[list[float]] = []
        self._offset: list[int] = []
        self._ncomp: list[int] = []
        self.size = 0
        self.max_size = 0
        self.n = 0
        self.min = math.inf
        self.max = -math.inf
        self._cache: tuple[np.ndarray, np.ndarray] | None = None
        self._grow()

    @property
    def H(self) -> int:
        return len(self.levels)

    def capacity(self, h: int) -> int:
        return int(math.ceil(self.k * self.c ** (self.H - h - 1))) + 1

    def _grow(self) -> None:
        self.levels.append([])
        self._offset.append(0)
        self._ncomp.append(0)
        self.max_size = sum(self.capacity(h) for h in range(self.H))

    def _compact(self, h: int) -> list[float]:
        if self.alternate and self._ncomp[h] % 2 == 1:
            off = 1 - self._offset[h]
        else:
            off = int(self._rng.integers(2))
        self._offset[h] = off
        self._ncomp[h] += 1
        arr = sorted(self.levels[h])
        keep = [arr.pop()] if len(arr) % 2 == 1 else []
        self.levels[h] = keep
        return arr[off::2]

    def _compress(self) -> None:
        """Compacta el primer nivel lleno (versión perezosa: uno por llamada)."""
        for h in range(self.H):
            if len(self.levels[h]) >= self.capacity(h):
                if h + 1 >= self.H:
                    self._grow()
                self.levels[h + 1].extend(self._compact(h))
                self.size = sum(len(lv) for lv in self.levels)
                return

    def update(self, x: float) -> None:
        x = float(x)
        if math.isnan(x):
            return
        self._cache = None
        self.levels[0].append(x)
        self.size += 1
        self.n += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        if self.size >= self.max_size:
            self._compress()

    def update_many(self, xs: Iterable[float] | np.ndarray) -> None:
        """Igual que llamar ``update`` con cada valor (mismo resultado), pero por bloques."""
        a = np.asarray(xs if isinstance(xs, np.ndarray) else list(xs), dtype=np.float64).ravel()
        a = a[~np.isnan(a)]
        if a.size == 0:
            return
        self._cache = None
        self.n += a.size
        self.min = min(self.min, float(a.min()))
        self.max = max(self.max, float(a.max()))
        i = 0
        while i < a.size:
            take = min(self.max_size - self.size, a.size - i)
            if take <= 0:
                self._compress()
                continue
            self.levels[0].extend(a[i : i + take].tolist())
            self.size += take
            i += take
            if self.size >= self.max_size:
                self._compress()

    def merge(self, other: "KLLSketch") -> "KLLSketch":
        if self.k != other.k:
            raise ValueError("solo se combinan KLL con el mismo k")
        while self.H < other.H:
            self._grow()
        for h in range(other.H):
            self.levels[h].extend(other.levels[h])
        self.size = sum(len(lv) for lv in self.levels)
        self.n += other.n
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self._cache = None
        while self.size >= self.max_size:
            self._compress()
        return self

    def _sorted(self) -> tuple[np.ndarray, np.ndarray]:
        """Ítems guardados ordenados y su peso acumulado (float64)."""
        if self._cache is None:
            items = np.concatenate([np.asarray(lv, dtype=np.float64) for lv in self.levels])
            w = np.concatenate([np.full(len(lv), 2.0**h) for h, lv in enumerate(self.levels)])
            order = np.argsort(items, kind="stable")
            self._cache = (items[order], np.cumsum(w[order]))
        return self._cache

    def total_weight(self) -> float:
        return float(self._sorted()[1][-1]) if self.size else 0.0

    def rank(self, x: float | np.ndarray) -> float | np.ndarray:
        """Fracción estimada de datos ≤ x."""
        if self.size == 0:
            raise ValueError("sketch vacío")
        items, cw = self._sorted()
        pos = np.searchsorted(items, np.asarray(x, dtype=np.float64), side="right")
        r = np.where(pos > 0, cw[np.maximum(pos - 1, 0)], 0.0) / cw[-1]
        return float(r) if np.ndim(r) == 0 else r

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Menor ítem con rango acumulado ≥ q; q = 0 → mínimo y q = 1 → máximo exactos."""
        if self.size == 0:
            raise ValueError("sketch vacío")
        qa = np.asarray(q, dtype=np.float64)
        if ((qa < 0) | (qa > 1)).any():
            raise ValueError("q debe estar en [0, 1]")
        items, cw = self._sorted()
        pos = np.minimum(np.searchsorted(cw, qa * cw[-1], side="left"), items.size - 1)
        out = items[pos]
        out = np.where(qa <= 0, self.min, np.where(qa >= 1, self.max, out))
        return float(out) if out.ndim == 0 else out
