"""Medida angular: similitud coseno (y Pearson, aplicando coseno a vectores centrados por muestra).

Es la medida recomendada para las ómicas continuas tras log2 / M-value, z-score por
rasgo y L2 por muestra (ver ``docs/analisis_medidas_similitud.md``); en ese espacio
coincide con L2.

- ``simhash``: firmas de hiperplanos aleatorios, P[bit igual] = 1 − θ/π (Charikar 2002).
- ``lsh``: LSH por bandas sobre los bits SimHash, con re-ranking.
"""

from .lsh import BandedLSH, band_codes, s_curve, threshold_angle
from .simhash import (
    SimHashKNN,
    angle_estimate,
    cosine_estimate,
    hamming,
    hamming_matrix,
    pack_bits,
    sign_bits,
    signatures,
    unpack_bits,
)

__all__ = [
    "BandedLSH",
    "SimHashKNN",
    "angle_estimate",
    "band_codes",
    "cosine_estimate",
    "hamming",
    "hamming_matrix",
    "pack_bits",
    "s_curve",
    "sign_bits",
    "signatures",
    "threshold_angle",
    "unpack_bits",
]
