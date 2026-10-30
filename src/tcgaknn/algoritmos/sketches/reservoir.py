"""Reservoir sampling: muestra uniforme de tamaño k de un stream de largo desconocido (Algorithm R y L)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ._hash import M64


class ReservoirSampler:
    """Muestra uniforme sin reemplazo de tamaño k de un stream de largo desconocido.

    - ``method="R"`` (Algorithm R, Vitter 1985, *ACM TOMS* 11(1):37–57): el
      ítem t (0-based, t ≥ k) reemplaza una posición al azar con prob. k/(t+1).
    - ``method="L"`` (Li 1994, *ACM TOMS* 20(4):481–493): en vez de un número
      aleatorio por ítem, salta directamente al próximo ítem que entra:
      W ← W·exp(ln(u)/k), salto = ⌊ln(u′)/ln(1 − W)⌋ + 1. Espera O(k·(1 +
      log(N/k))) números aleatorios; ``extend`` con una secuencia aprovecha los
      saltos sin recorrer los ítems ignorados.

    En ambos casos, tras N ≥ k ítems, cada subconjunto de k ítems es igualmente
    probable.
    """

    def __init__(self, k: int, seed: int = 0, *, method: str = "R"):
        if k < 1:
            raise ValueError("k debe ser ≥ 1")
        if method not in ("R", "L"):
            raise ValueError("method debe ser 'R' o 'L'")
        self.k = int(k)
        self.method = method
        self._rng = np.random.default_rng([0x5A, seed & M64])
        self.reservoir: list[Any] = []
        self.n_seen = 0
        self._w = 1.0
        self._next = -1  # índice (0-based) del próximo ítem que entra (Algorithm L)

    def _u(self) -> float:
        """Uniforme en (0, 1) (excluye 0 para poder tomar logaritmo)."""
        u = self._rng.random()
        while u == 0.0:
            u = self._rng.random()
        return u

    def _schedule(self) -> None:
        self._w *= math.exp(math.log(self._u()) / self.k)
        if self._w >= 1.0:  # salvaguarda numérica
            self._next = self.n_seen
            return
        self._next = self.n_seen + int(math.floor(math.log(self._u()) / math.log1p(-self._w)))

    def update(self, item: Any) -> None:
        t = self.n_seen
        if t < self.k:
            self.reservoir.append(item)
            self.n_seen += 1
            if self.method == "L" and self.n_seen == self.k:
                self._w = 1.0
                self._schedule()
            return
        if self.method == "R":
            j = int(self._rng.integers(0, t + 1))
            if j < self.k:
                self.reservoir[j] = item
            self.n_seen += 1
            return
        if t == self._next:
            self.reservoir[int(self._rng.integers(0, self.k))] = item
            self.n_seen += 1
            self._schedule()
        else:
            self.n_seen += 1

    def extend(self, items: Iterable[Any]) -> None:
        """Agrega muchos ítems. Con Algorithm L y una secuencia indexable, salta los ignorados."""
        if self.method == "L" and isinstance(items, (Sequence, np.ndarray)):
            n = len(items)
            i = 0
            while i < n and self.n_seen < self.k:
                self.update(items[i])
                i += 1
            if i >= n:
                return
            base = self.n_seen - i  # n_seen = base + i para el ítem i
            while True:
                j = self._next - base
                if j >= n:
                    self.n_seen = base + n
                    return
                self.n_seen = self._next
                self.reservoir[int(self._rng.integers(0, self.k))] = items[j]
                self.n_seen += 1
                self._schedule()
        for it in items:
            self.update(it)

    def sample(self) -> list[Any]:
        """Copia de la muestra actual (min(k, N) ítems)."""
        return list(self.reservoir)
