"""Misra-Gries: ítems frecuentes con k contadores y cota determinista de subestimación."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


class MisraGries:
    """Ítems frecuentes con k contadores (Misra y Gries 1982, *Sci. Comput. Program.* 2(2):143–152).

    Si el ítem tiene contador, se incrementa; si hay un contador libre, se crea;
    si no, se decrementan los k contadores y el ítem nuevo en 1 (se borran los
    que llegan a 0). Cada decremento descuenta k + 1 unidades del total, así que
    hay a lo más N/(k+1) y, para todo ítem,

        f(x) − N/(k+1) ≤ estimado(x) ≤ f(x).

    Pesos enteros c > 0: el ítem entra con c y, si hay k + 1 contadores, se
    resta el mínimo a todos (equivale a c decrementos unitarios agrupados).

    ``merge`` (Agarwal et al. 2012, *Mergeable summaries*, PODS): se suman los
    contadores y, si quedan más de k, se resta el (k+1)-ésimo mayor a todos y se
    conservan los positivos; la cota pasa a (N₁ + N₂)/(k+1).
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k debe ser ≥ 1")
        self.k = int(k)
        self.counters: dict[Hashable, int] = {}
        self.n = 0

    def update(self, item: Hashable, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count debe ser > 0")
        self.n += count
        ctr = self.counters
        if item in ctr:
            ctr[item] += count
        elif len(ctr) < self.k:
            ctr[item] = count
        elif count == 1:
            for key in list(ctr):  # O(k), a lo más N/(k+1) veces: O(N) amortizado
                if ctr[key] == 1:
                    del ctr[key]
                else:
                    ctr[key] -= 1
        else:
            ctr[item] = count
            self._reduce()

    def extend(self, items: Iterable[Hashable]) -> None:
        for it in items:
            self.update(it)

    def _reduce(self) -> None:
        """Si hay más de k contadores, resta el (k+1)-ésimo mayor y deja los positivos."""
        if len(self.counters) <= self.k:
            return
        vals = sorted(self.counters.values(), reverse=True)
        cut = vals[self.k]
        self.counters = {key: v - cut for key, v in self.counters.items() if v > cut}

    def estimate(self, item: Hashable) -> int:
        """Cota inferior de f(item); el error es ≤ ``error_bound``."""
        return self.counters.get(item, 0)

    @property
    def error_bound(self) -> float:
        return self.n / (self.k + 1)

    def frequent(self, phi: float | None = None) -> list[tuple[Hashable, int]]:
        """Contadores ordenados por valor; con ``phi``, solo los candidatos a f > phi·N.

        Se devuelven los ítems con estimado > phi·N − N/(k+1): incluye a todo
        ítem con f > phi·N (sin falsos negativos si phi > 1/(k+1)).
        """
        items = sorted(self.counters.items(), key=lambda t: -t[1])
        if phi is None:
            return items
        thr = phi * self.n - self.error_bound
        return [(key, v) for key, v in items if v > thr]

    def merge(self, other: "MisraGries") -> "MisraGries":
        if self.k != other.k:
            raise ValueError("solo se combinan resúmenes con el mismo k")
        for key, v in other.counters.items():
            self.counters[key] = self.counters.get(key, 0) + v
        self.n += other.n
        self._reduce()
        return self
