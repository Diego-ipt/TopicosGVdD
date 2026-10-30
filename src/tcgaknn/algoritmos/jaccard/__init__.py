"""Similitud de Jaccard entre conjuntos (p. ej. genes mutados por paciente).

- ``minhash``: MinHash (Broder 1997) en lote y en streaming por eventos, LSH por
  bandas (``MinHashLSH``) y Jaccard ponderada con ICWS (Ioffe 2010).
"""

from .minhash import (
    MinHash,
    MinHashLSH,
    StreamingMinHash,
    StreamingWeightedMinHash,
    WeightedMinHash,
    exact_jaccard,
    jaccard_estimate,
    lsh_threshold,
    s_curve,
    token_hash,
    token_hashes,
    weighted_jaccard,
    weighted_jaccard_estimate,
)

__all__ = [
    "MinHash",
    "MinHashLSH",
    "StreamingMinHash",
    "StreamingWeightedMinHash",
    "WeightedMinHash",
    "exact_jaccard",
    "jaccard_estimate",
    "lsh_threshold",
    "s_curve",
    "token_hash",
    "token_hashes",
    "weighted_jaccard",
    "weighted_jaccard_estimate",
]
