"""Implementaciones de búsqueda por similitud y sketches del proyecto.

Estructura:

- ``base``: interfaz ``KNNIndex``, ``topk``, Ω por ID de rasgo y acumulador en streaming.
- ``evaluacion``: recall@k, voto k-NN y métricas de clasificación.

Los algoritmos de búsqueda están agrupados por la **medida** que aproximan:

==================  ===========================================  ==========================
Carpeta             Algoritmos                                   Datos donde aplica
==================  ===========================================  ==========================
``exacto/``         k-NN exacto (coseno, L2, L1, correlación)    todos (verdad de terreno)
``coseno/``         SimHash, LSH por bandas                      ómicas continuas (Pearson
                                                                 = coseno tras centrar)
``euclidiana/``     proyección JL, PCA de referencia             ómicas continuas (L2)
``lp/``             E2LSH p-estable (L2 gaussiano, L1 Cauchy)    L2 o L1
``jaccard/``        MinHash, MinHashLSH, ICWS                    conjuntos (mutaciones)
==================  ===========================================  ==========================

- ``sketches/``: Count-Min, Misra-Gries, HyperLogLog, KLL y reservoir sampling
  (resúmenes del stream, no medidas de similitud).
- ``clustering/``: mini-batch k-means con k-means++.

Las clases principales se pueden importar directo desde aquí::

    from tcgaknn.algoritmos import ExactKNN, SimHashKNN, BandedLSH
"""

from .base import KNNIndex, StreamingProjectionAccumulator, l2_normalize, omega_matrix, omega_row, topk
from .clustering import MiniBatchKMeans
from .exacto import ExactKNN, StreamingExactKNN
from .coseno import BandedLSH, SimHashKNN
from .euclidiana import PCAKNN, ProjectedKNN, RandomProjection
from .jaccard import MinHash, MinHashLSH, StreamingMinHash
from .lp import E2LSH
from .sketches import CountMinSketch, HyperLogLog, KLLSketch, MisraGries, ReservoirSampler

__all__ = [
    "PCAKNN",
    "BandedLSH",
    "CountMinSketch",
    "E2LSH",
    "ExactKNN",
    "HyperLogLog",
    "KLLSketch",
    "KNNIndex",
    "MiniBatchKMeans",
    "MinHash",
    "MinHashLSH",
    "MisraGries",
    "ProjectedKNN",
    "RandomProjection",
    "ReservoirSampler",
    "SimHashKNN",
    "StreamingExactKNN",
    "StreamingMinHash",
    "StreamingProjectionAccumulator",
    "l2_normalize",
    "omega_matrix",
    "omega_row",
    "topk",
]
