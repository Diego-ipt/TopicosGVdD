"""HyperLogLog: número aproximado de elementos distintos (Flajolet et al. 2007)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable

import numpy as np

from ._hash import hash64, hash64_many, _bit_length_u64


def _hll_alpha(m: int) -> float:
    """Constante α_m de Flajolet et al. (2007), Fig. 3."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLog:
    """Conteo de distintos (Flajolet, Fusy, Gandouet y Meunier 2007, AofA).

    Hash de 64 bits: los p bits altos eligen el registro j ∈ [0, 2^p) y en los
    64 − p restantes ρ = posición del primer 1 (desde la izquierda, empezando en
    1; 65 − p si son todos 0). M[j] = max ρ. Estimador bruto

        E = α_m · m² / Σ_j 2^(−M[j]),   m = 2^p.

    Rango pequeño: si E ≤ 2,5·m y hay V > 0 registros en cero, se usa *linear
    counting* E = m·ln(m/V). Con hash de 64 bits no hace falta la corrección de
    rango grande del artículo original (Heule, Nunkesser y Hall 2013,
    *HyperLogLog in practice*, EDBT). Error estándar relativo ≈ 1,04/√m
    (p = 14 → 0,81 %). ``merge`` = máximo registro a registro (sin pérdida).
    """

    def __init__(self, p: int = 14, seed: int = 0):
        if not 4 <= p <= 18:
            raise ValueError("p debe estar en [4, 18]")
        self.p = int(p)
        self.m = 1 << p
        self.seed = int(seed)
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = _hll_alpha(self.m)

    @property
    def std_error(self) -> float:
        return 1.04 / math.sqrt(self.m)

    def _add_hashes(self, h: np.ndarray) -> None:
        q = 64 - self.p
        idx = (h >> np.uint64(q)).astype(np.int64)
        rest = h & np.uint64((1 << q) - 1)
        rho = (q - _bit_length_u64(rest) + 1).astype(np.uint8)
        np.maximum.at(self.registers, idx, rho)

    def add(self, item: Hashable) -> None:
        h = hash64(item, self.seed)
        q = 64 - self.p
        j = h >> q
        rho = q - (h & ((1 << q) - 1)).bit_length() + 1
        if rho > self.registers[j]:
            self.registers[j] = rho

    def add_many(self, items: Iterable[Hashable] | np.ndarray) -> None:
        """Vectorizado para arreglos enteros; en otro caso hashea ítem a ítem."""
        h = hash64_many(items if isinstance(items, np.ndarray) else list(items), self.seed)
        if h.size:
            self._add_hashes(h)

    def raw_estimate(self) -> float:
        return self.alpha * self.m * self.m / float(np.sum(np.ldexp(1.0, -self.registers.astype(np.int64))))

    def estimate(self) -> float:
        e = self.raw_estimate()
        if e <= 2.5 * self.m:
            v = int(np.count_nonzero(self.registers == 0))
            if v > 0:
                return self.m * math.log(self.m / v)
        return e

    def __len__(self) -> int:
        return int(round(self.estimate()))

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        if (self.p, self.seed) != (other.p, other.seed):
            raise ValueError("solo se combinan HLL con igual p y seed")
        np.maximum(self.registers, other.registers, out=self.registers)
        return self
