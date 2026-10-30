"""k-NN exacto (verdad de terreno): coseno, L2, L1 y correlación, en memoria y en streaming."""

from .exact import (
    l1_accumulate,
    ExactKNN,
    standardize_feature,
    StreamingExactKNN,
)

__all__ = [
    "ExactKNN",
    "StreamingExactKNN",
    "l1_accumulate",
    "standardize_feature",
]
