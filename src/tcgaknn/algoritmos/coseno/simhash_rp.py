"""SimHash con re-ranking en proyección RP, incluido el coste de firmar consultas.

Evita el coste n*d del re-ranking exacto: mantiene únicamente n*m números
float32 más n*b/8 bytes de firmas y la matriz de proyección d*max(b,m).
La búsqueda de Hamming sigue siendo exhaustiva sobre las n firmas.
"""
from __future__ import annotations

import numpy as np

from ..base import as_float32_2d, l2_normalize
from ..euclidiana.projection import RandomProjection
from .simhash import SimHashKNN, pack_bits


class SimHashRPKNN:
    name = "SH+RP"
    score_kind = "similarity"

    def __init__(self, b: int = 256, m: int = 256, rerank_factor: int = 10, seed: int = 2026):
        if b % 64 or b <= 0 or m < 1 or rerank_factor < 1:
            raise ValueError("b debe ser múltiplo de 64, m>=1 y rerank_factor>=1")
        self.b, self.m, self.f, self.seed = b, m, rerank_factor, seed
        self.rp = RandomProjection(m=max(b, m), seed=seed)
        self.sh: SimHashKNN | None = None

    def _encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = l2_normalize(as_float32_2d(X))
        Y = self.rp.transform(Xn)
        # La Ω compartida se divide por sqrt(max(b,m)), lo que no altera signos.
        bits = Y[:, :self.b] >= 0
        return pack_bits(bits), np.ascontiguousarray(Y[:, :self.m])

    def fit(self, X: np.ndarray):
        X = as_float32_2d(X)
        self.rp.fit(X.shape[1])
        packed, proj = self._encode(X)
        self.sh = SimHashKNN.from_signatures(packed, self.b, base=proj, rerank_factor=self.f)
        return self

    def query(self, Q: np.ndarray, k: int):
        if self.sh is None:
            raise RuntimeError("Se necesita fit antes de query")
        packed, proj = self._encode(Q)
        idx, score, stats = self.sh.query_packed(packed, k, Q_base=proj)
        stats["index_bytes"] = self.index_bytes()
        return idx, score, stats

    def index_bytes(self) -> int:
        if self.sh is None:
            return 0
        return int(self.sh.packed.nbytes + self.sh.base.nbytes + self.rp.nbytes)
