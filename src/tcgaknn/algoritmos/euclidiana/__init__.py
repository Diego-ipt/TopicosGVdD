"""Distancia euclidiana (L2): reducción de dimensionalidad que la preserva.

- ``projection``: proyecciones aleatorias de Johnson–Lindenstrauss (gaussiana y
  Achlioptas), con la cota de Dasgupta–Gupta. JL solo existe para L2.
- ``pca_ref``: PCA aleatorizada como referencia dependiente de los datos.

Para LSH euclidiano directo ver ``lp`` (E2LSH con p = 2).
"""

from .pca_ref import PCAKNN
from .projection import (
    ProjectedKNN,
    RandomProjection,
    achlioptas_matrix,
    achlioptas_row,
    distortion_stats,
    jl_eps_for,
    jl_min_dim,
    project,
    random_matrix,
)

__all__ = [
    "PCAKNN",
    "ProjectedKNN",
    "RandomProjection",
    "achlioptas_matrix",
    "achlioptas_row",
    "distortion_stats",
    "jl_eps_for",
    "jl_min_dim",
    "project",
    "random_matrix",
]
