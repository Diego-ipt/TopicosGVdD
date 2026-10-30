"""Normas Lp con distribuciones p-estables (Datar et al. 2004).

- ``e2lsh``: h(v) = ⌊(a·v + β)/w⌋ con a gaussiana (p = 2, euclidiana) o de
  Cauchy (p = 1, Manhattan). Es la única opción del repo para L1: no existe
  reducción JL lineal para L1 (Brinkman y Charikar 2005).
"""

from .e2lsh import E2LSH, candidate_probability, collision_probability, rho, suggest_w

__all__ = ["E2LSH", "candidate_probability", "collision_probability", "rho", "suggest_w"]
