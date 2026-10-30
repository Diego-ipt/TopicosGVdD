"""Sketches para resumir el stream en una pasada y con memoria acotada.

Son contenido del curso (unidad de sketches) y sirven en el preprocesamiento en
streaming, no en la búsqueda k-NN en sí:

- ``CountMinSketch``: frecuencias aproximadas (p. ej. cuántas mutaciones tiene
  cada gen en el stream de eventos MAF) que **nunca subestiman**.
- ``MisraGries``: ítems frecuentes con k contadores (genes más mutados) y cota
  determinista de subestimación.
- ``HyperLogLog``: número de elementos distintos (genes, pacientes, muestras)
  con 2^p registros de 6 bits.
- ``KLLSketch``: cuantiles en streaming; p. ej. el umbral de varianza que deja
  los G genes más variables sin guardar todas las varianzas.
- ``ReservoirSampler``: muestra uniforme de tamaño k de un stream de largo
  desconocido (Algorithm R y Algorithm L).

Hash estable de 64 bits (``hash64``): los enteros pasan por el finalizador de
SplitMix64 (vectorizable con NumPy) y el resto (str, bytes) por BLAKE2b con la
semilla como clave. No se usa ``hash()`` de Python porque cambia entre procesos
(PYTHONHASHSEED). Todas las semillas son explícitas.

Uso típico sobre el stream::

    hll = HyperLogLog(p=14, seed=0)
    cm = CountMinSketch.from_error(eps=1e-4, delta=1e-3, seed=0)
    for rec in StreamService(raiz, data_types=["mutations"]):
        if rec.kind == "event":
            gen = rec.fields["Hugo_Symbol"]
            hll.add(rec.fields["Tumor_Sample_Barcode"])
            cm.add(gen)

    kll = KLLSketch(k=200, seed=0)
    for rec in StreamService(raiz, data_types=["expression"]):
        if rec.kind == "feature_row":
            kll.update(np.nanvar(rec.values))
    umbral = kll.quantile(1 - G / kll.n)   # varianza mínima del top-G aproximado
"""

from ._hash import hash64, hash64_many, splitmix64, splitmix64_array
from .count_min import CountMinSketch
from .hyperloglog import HyperLogLog
from .kll import KLLSketch
from .misra_gries import MisraGries
from .reservoir import ReservoirSampler

__all__ = [
    "CountMinSketch",
    "HyperLogLog",
    "KLLSketch",
    "MisraGries",
    "ReservoirSampler",
    "hash64",
    "hash64_many",
    "splitmix64",
    "splitmix64_array",
]
