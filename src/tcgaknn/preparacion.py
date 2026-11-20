"""Preparación auditada TCGA RNA, con parámetros por gen aprendidos solo en training."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from .cohorte_validada import Cohort, build_cohort, save_audit
from .splits import make_split
from .stream import discover


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _transform(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "as_is":
        return x
    if mode == "log2clip":
        # SOLO para estudio exploratorio: A1 ya está ajustada y puede tener negativos.
        return np.log2(np.maximum(x, 0.0) + 1.0)
    raise ValueError("transform debe ser 'as_is' o 'log2clip'")


def _single_source(root: Path, kind: str):
    sources, _ = discover(root)
    files = [s for s in sources if s.data_type == kind]
    if len(files) != 1:
        raise ValueError(f"Se esperaba exactamente un archivo {kind} (hay {len(files)}) en {root}")
    return files[0]


def prepare(root: Path | str, out: Path | str, *, seed: int = 2026,
            transform: str = "as_is", zscore: bool = True,
            check_annotations: bool = False, max_genes: int | None = None) -> dict[str, Any]:
    """Procesa A1 por filas sin cargar el archivo entero a RAM.

    Archivo X_gene_major.npy: (genes, muestras), se consulta mediante .T.
    Descartes de genes, parámetros e imputación se aprenden SOLO de training.
    No se toca el conjunto de test para seleccionar rasgos ni hiperparámetros.
    """
    root, out = Path(root), Path(out)
    if out.resolve() == root.resolve():
        raise ValueError("La carpeta de salida debe ser distinta a la de entrada")
    out.mkdir(parents=True, exist_ok=True)
    a1 = _single_source(root, "expression")
    _single_source(root, "clinical_cdr")
    _single_source(root, "sample_quality")
    # Solo lee esquema de A1: no consume valores de expresión para definir cohortes/splits.
    reader = a1.records()
    schema = next(reader)
    reader.close()
    if schema.kind != "schema":
        raise ValueError("A1 no tiene cabecera válida")
    barcodes = schema.fields["columns"]
    cohort: Cohort = build_cohort(barcodes, root, check_annotations=check_annotations)
    split = make_split(cohort.patient_ids, cohort.labels, seed=seed)
    save_audit(cohort.audit, out / "auditoria_muestras.csv")

    # Primera pasada: medias y desviaciones exclusivamente con filas de training.
    gene_ids: list[str] = []
    mu: list[float] = []
    sd: list[float] = []
    discarded: Counter[str] = Counter()
    all_seen: set[str] = set()
    n_read = 0
    for rec in a1.records():
        if rec.kind != "feature_row":
            continue
        if max_genes is not None and n_read >= max_genes:
            break
        n_read += 1
        gid = str(rec.key)
        if gid in all_seen:
            raise ValueError(f"A1 tiene un identificador de gen duplicado: {gid!r}")
        all_seen.add(gid)
        v = _transform(rec.values[cohort.columns], transform)
        ref = v[split.train]
        ok = np.isfinite(ref)
        if ok.sum() < 2:
            discarded["menos_de_2_valores_entrenamiento"] += 1
            continue
        avg = float(ref[ok].mean(dtype=np.float64))
        scale = float(ref[ok].std(dtype=np.float64))
        if not np.isfinite(scale) or scale <= 1e-12:
            discarded["sin_variacion_en_entrenamiento"] += 1
            continue
        gene_ids.append(gid)
        mu.append(avg)
        sd.append(scale)
    if not gene_ids:
        raise ValueError("No quedaron genes con variación suficiente en training")

    # Segunda pasada: escritura por gen, contigua en disco; no vuelve a ajustar parámetros.
    path = out / "X_gene_major.npy"
    matrix = open_memmap(path, mode="w+", dtype=np.float32,
                         shape=(len(gene_ids), len(cohort.patient_ids)))
    keep = {gid: i for i, gid in enumerate(gene_ids)}
    j = 0
    for rec in a1.records():
        if rec.kind != "feature_row":
            continue
        if max_genes is not None and j >= max_genes:
            break
        j += 1
        ix = keep.get(str(rec.key))
        if ix is None:
            continue
        v = _transform(rec.values[cohort.columns], transform)
        if zscore:
            v = (v - mu[ix]) / sd[ix]
            v[~np.isfinite(v)] = 0.0  # equivale a imputar media de TRAIN
        else:
            v[~np.isfinite(v)] = mu[ix]
        matrix[ix] = v.astype(np.float32)
    matrix.flush()
    del matrix

    with (out / "pacientes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "patient_id", "sample_id", "type", "split"])
        sets = {i: part for part, pos in (("train", split.train), ("val", split.val), ("test", split.test))
                for i in pos.tolist()}
        for i, (p, s, lab) in enumerate(zip(cohort.patient_ids, cohort.sample_ids, cohort.labels.tolist())):
            w.writerow([i, p, s, lab, sets[i]])
    np.savez_compressed(out / "splits.npz", train=split.train, val=split.val,
                        test=split.test, labels=cohort.labels,
                        patient_ids=np.asarray(cohort.patient_ids), sample_ids=np.asarray(cohort.sample_ids),
                        feature_ids=np.asarray(gene_ids), gene_mean_train=np.asarray(mu), gene_std_train=np.asarray(sd))
    meta = {
        "seed": seed, "transform": transform, "zscore": bool(zscore),
        "quality_check_annotations": bool(check_annotations),
        "cohort_total": len(cohort.patient_ids), "genes_read": n_read, "genes_used": len(gene_ids),
        "genes_discarded": dict(discarded), "shape_gene_major": [len(gene_ids), len(cohort.patient_ids)],
        "n_train": int(len(split.train)), "n_val": int(len(split.val)), "n_test": int(len(split.test)),
        "count_by_class": cohort.counts(),
        "count_audit_reasons": dict(Counter(r["reason"] for r in cohort.audit)),
        "X_sha256": sha256_file(path),
        "splits_sha256": sha256_file(out / "splits.npz"),
    }
    fingerprint = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    meta["dataset_id"] = hashlib.sha256(fingerprint).hexdigest()
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_prepared(path: Path | str, *, verify_hash: bool = False):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if verify_hash:
        if sha256_file(path / "X_gene_major.npy") != meta["X_sha256"]:
            raise ValueError("El contenido de la matriz cambió después de preparar la cohorte")
        if sha256_file(path / "splits.npz") != meta["splits_sha256"]:
            raise ValueError("Los splits cambiaron después de preparar la cohorte")
    data = np.load(path / "splits.npz")
    arr = np.load(path / "X_gene_major.npy", mmap_mode="r")
    if tuple(arr.shape) != tuple(meta["shape_gene_major"]):
        raise ValueError("La matriz no tiene las dimensiones registradas")
    return arr.T, data, meta
