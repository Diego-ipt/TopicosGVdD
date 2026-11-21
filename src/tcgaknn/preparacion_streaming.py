"""Prepara A1∩A2∩A3 con una sola lectura gen-a-gen y sin matriz de expresión densa.

Se usa la misma partición train/val/test para los tres ANN. Se acumulan:
- Y_cos = Z·Ω para LSH angular y su reranking con proyección.
- Y_p2 = Z·A_gauss y Y_p1 = Z·A_cauchy (E2LSH).
- Gram coseno y distancias L1 exactas Q×train en archivos .npy mmap.

Las matrices de verdad exacta se guardan en disco, no en un objeto RAM o ZIP;
son O(|Q|×|train|) y NO equivalen a conservar la matriz genes×pacientes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .algoritmos.base import StreamingProjectionAccumulator
from .algoritmos.exacto.exact import StreamingExactKNN
from .algoritmos.lp.e2lsh import E2LSH
from .cohorte_validada import Cohort, build_cohort, save_audit
from .splits import make_split, stratified_nested_order
from .stream import discover

DEFAULT_TRANSFORM = "log2clip"
SKETCH_FILE = "sketches.npz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _transform(x: np.ndarray, mode: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if mode == "log2clip":
        return np.log2(np.maximum(v, 0) + 1)
    if mode == "as_is":
        return v
    raise ValueError("transform debe ser log2clip o as_is")


def _single_source(root: Path, kind: str):
    sources, _ = discover(root)
    files = [s for s in sources if s.data_type == kind]
    if len(files) != 1:
        raise ValueError(f"Se esperaba exactamente un archivo {kind}, hay {len(files)}")
    return files[0]


def peak_rss_mib() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


class RowAccumulator:
    """Portado de main: acumula Z·A por lotes, regenerando A desde ID de gen."""

    def __init__(self, n_samples: int, cols: int, row_fn: Callable[[str], np.ndarray], batch_size: int = 256):
        self.Y = np.zeros((n_samples, cols), dtype=np.float64)
        self.row_fn = row_fn
        self._z = np.empty((batch_size, n_samples), dtype=np.float32)
        self._a = np.empty((batch_size, cols), dtype=np.float32)
        self._fill = 0

    def update(self, feature_id: str, z: np.ndarray) -> None:
        self._z[self._fill] = z
        self._a[self._fill] = self.row_fn(feature_id)
        self._fill += 1
        if self._fill == self._z.shape[0]:
            self._flush()

    def _flush(self) -> None:
        if self._fill:
            self.Y += self._z[:self._fill].T.astype(np.float64) @ self._a[:self._fill].astype(np.float64)
            self._fill = 0

    def result(self) -> np.ndarray:
        self._flush()
        return self.Y


def _queries(part: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    """Subconjunto fijo de consultas por fase, independiente del test y del tamaño n."""
    if limit is None or limit <= 0 or len(part) <= limit:
        return np.asarray(part, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(part, size=limit, replace=False)).astype(np.int64)


def prepare_streaming(
    root: Path | str, out: Path | str, *, seed: int = 2026,
    transform: str = DEFAULT_TRANSFORM, zscore: bool = True, cols: int = 512,
    lp_cols: int = 256, queries_per_phase: int | None = 0,
    check_annotations: bool = False, max_genes: int | None = None,
    batch_size: int = 256, log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """1 pasada sobre A1; la verdad de cada B_n se extrae de prefijos de las matrices Q×B.

    ``queries_per_phase=0`` (por defecto) usa toda validación y test.
    Un límite positivo es una opción explícita para desarrollo rápido.
    """
    root, out = Path(root), Path(out)
    if root.resolve() == out.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError("La salida debe estar fuera de la carpeta de entrada")
    if cols < 1 or lp_cols < 1 or batch_size < 1:
        raise ValueError("cols, lp_cols y batch_size deben ser positivos")
    if max_genes is not None and max_genes < 1:
        raise ValueError("max_genes debe ser positivo")
    if queries_per_phase is not None and queries_per_phase < 0:
        raise ValueError("queries_per_phase no puede ser negativo")
    out.mkdir(parents=True, exist_ok=True)
    a1 = _single_source(root, "expression")
    _single_source(root, "clinical_cdr")
    _single_source(root, "sample_quality")
    reader = a1.records()
    schema = next(reader)
    reader.close()
    if schema.kind != "schema":
        raise ValueError("Cabecera A1 inválida")
    cohort: Cohort = build_cohort(schema.fields["columns"], root, check_annotations=check_annotations)
    parts = make_split(cohort.patient_ids, cohort.labels, seed=seed)
    base_order = stratified_nested_order(parts.train, cohort.labels, seed=seed + 1)
    q_val = _queries(parts.val, queries_per_phase, seed + 2)
    q_test = _queries(parts.test, queries_per_phase, seed + 3)
    query_idx = np.concatenate((q_val, q_test))
    n = len(cohort.labels)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[parts.train] = True
    save_audit(cohort.audit, out / "auditoria_muestras.csv")
    cos = StreamingProjectionAccumulator(n, cols, seed, standardize=False, batch_size=batch_size)
    # A debe compartir semilla y generación de filas con E2LSH.fit_projections.
    gaussian = E2LSH(p=2, r=1, L=lp_cols, seed=seed, rerank="proj")
    cauchy = E2LSH(p=1, r=1, L=lp_cols, seed=seed, rerank="proj")
    p2 = RowAccumulator(n, lp_cols, gaussian.a_row, batch_size)
    p1 = RowAccumulator(n, lp_cols, cauchy.a_row, batch_size)
    ex = StreamingExactKNN(n, query_idx, base_order, metrics=("cosine", "l1"),
                           standardize=False, batch_size=batch_size,
                           gram_path=out / "gram.npy", l1_path=out / "l1.npy")
    seen: set[str] = set()
    genes: list[str] = []
    means: list[float] = []
    sds: list[float] = []
    rejected = Counter()
    negatives = missing = read = 0
    start = time.perf_counter()
    for rec in a1.records():
        if rec.kind != "feature_row":
            continue
        if max_genes is not None and read >= max_genes:
            break
        read += 1
        gid = str(rec.key)
        if gid in seen:
            raise ValueError(f"Gen repetido en A1: {gid}")
        seen.add(gid)
        raw = rec.values[cohort.columns]
        negatives += int(np.count_nonzero(raw < 0))
        missing += int(np.count_nonzero(~np.isfinite(raw)))
        v = _transform(raw, transform)
        reference = v[train_mask]
        reference = reference[np.isfinite(reference)]
        if len(reference) < 2:
            rejected["train_insuficiente"] += 1
            continue
        mu, sd = float(reference.mean()), float(reference.std())
        if not np.isfinite(sd) or sd <= 1e-12:
            rejected["sin_variacion_en_train"] += 1
            continue
        z = ((v - mu) / sd) if zscore else v.copy()
        # Si se deshabilita zscore no se reemplaza NA por la media cruda de train de forma silenciosa.
        z = np.where(np.isfinite(z), z, 0 if zscore else mu).astype(np.float32)
        cos.update(gid, z)
        p2.update(gid, z)
        p1.update(gid, z)
        ex.update(gid, z)
        genes.append(gid)
        means.append(mu)
        sds.append(sd)
        if read % 2000 == 0:
            log(f"  genes leídos: {read:,}; tiempo {time.perf_counter()-start:.0f}s")
    if not genes:
        raise ValueError("No hay genes utilizables tras filtrar con train")
    Y = cos.raw()
    Y2, Y1 = p2.result(), p1.result()
    sq = cos.sqnorm.copy()
    ex._flush()
    ex.G.flush()
    ex.L1.flush()
    np.savez(out / SKETCH_FILE, Y=Y, Y_p2=Y2, Y_p1=Y1, sqnorm=sq)
    np.savez_compressed(out / "splits.npz", train=parts.train, val=parts.val, test=parts.test,
                        base_order=base_order, query_val=q_val, query_test=q_test,
                        labels=cohort.labels, patient_ids=np.asarray(cohort.patient_ids),
                        sample_ids=np.asarray(cohort.sample_ids), feature_ids=np.asarray(genes),
                        gene_mean_train=np.asarray(means), gene_std_train=np.asarray(sds))
    with (out / "pacientes.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "patient_id", "sample_id", "type", "split"])
        group = {int(i): phase for phase, ix in (("train", parts.train), ("val", parts.val),
                                                  ("test", parts.test)) for i in ix}
        for i, (pid, sid, label) in enumerate(zip(cohort.patient_ids, cohort.sample_ids, cohort.labels)):
            wr.writerow([i, pid, sid, label, group[i]])
    files = (SKETCH_FILE, "splits.npz", "gram.npy", "l1.npy")
    meta: dict[str, Any] = {
        "mode": "streaming", "seed": seed, "transform": transform, "zscore": bool(zscore),
        "cols": cols, "lp_cols": lp_cols, "queries_per_phase": queries_per_phase,
        "base_order_policy": "stratified_nested_first_each_class_then_deficit",
        "check_annotations": check_annotations, "cohort_total": n,
        "n_train": len(parts.train), "n_val": len(parts.val), "n_test": len(parts.test),
        "n_queries_val": len(q_val), "n_queries_test": len(q_test),
        "genes_read": read, "genes_used": len(genes), "genes_discarded": dict(rejected),
        "n_negative_values": negatives, "n_missing_values": missing,
        "count_by_class": cohort.counts(), "count_audit_reasons": dict(Counter(x["reason"] for x in cohort.audit)),
        "zero_norm_patients": [cohort.patient_ids[i] for i in np.flatnonzero(sq <= 1e-12)],
        "hashes": {name: sha256_file(out / name) for name in files},
        "metric_spaces": {"lsh": "cosine(Z)", "e2lsh_p2": "L2(Z/||Z||2)",
                          "e2lsh_p1": "L1(Z)"},
    }
    meta["dataset_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    meta["streaming_run"] = {
        "pass_seconds": time.perf_counter()-start, "peak_rss_mib": peak_rss_mib(),
        "bytes_gram_on_disk": (out / "gram.npy").stat().st_size,
        "bytes_l1_on_disk": (out / "l1.npy").stat().st_size,
        "bytes_dense_matrix_avoided": n * len(genes) * 4,
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_sketches(path: Path | str, *, phase: str, verify_hash: bool = False):
    """Carga sketches y *solo las filas* de verdad exacta de la fase solicitada.

    Devuelve (Y, Y_p2, Y_p1, sqnorm, G_fase, L1_fase, split_npz, metadata).
    G y L1 son memmaps leídos por columnas según la base anidada.
    """
    path = Path(path)
    if phase not in {"val", "test"}:
        raise ValueError("phase debe ser val o test")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta["mode"] != "streaming":
        raise ValueError("La carpeta no corresponde al protocolo streaming")
    if verify_hash:
        for name, digest in meta["hashes"].items():
            if sha256_file(path / name) != digest:
                raise ValueError(f"Archivo alterado: {name}")
    parts = np.load(path / "splits.npz", allow_pickle=False)
    with np.load(path / SKETCH_FILE, allow_pickle=False) as stored:
        Y, Y2, Y1, sq = (stored[x] for x in ("Y", "Y_p2", "Y_p1", "sqnorm"))
    qval = len(parts["query_val"])
    rows = slice(0, qval) if phase == "val" else slice(qval, qval + len(parts["query_test"]))
    gram = np.load(path / "gram.npy", mmap_mode="r")[rows]
    l1 = np.load(path / "l1.npy", mmap_mode="r")[rows]
    return Y, Y2, Y1, sq, gram, l1, parts, meta
