"""Cohorte TCGA para RNA: enlace A1/A2/A3 con trazabilidad, sin fuga de datos.

A1 es la matriz de expresión; A2 proporciona etiquetas por paciente;
A3 contiene anotaciones por alícuota. NO se equipara un paciente con una
alícuota: el emparejamiento es exacto por alícuota o único por muestra TCGA.
Se excluye por defecto cualquier muestra sin correspondencia inequívoca.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .stream import StreamService


def _id(v: object) -> str:
    return str(v).strip().upper()


TUMOR_CODES = ("01", "03", "06")
CANONICAL_CODE = {"LAML": "03", "SKCM": "06"}


def patient_barcode(barcode: str) -> str:
    return barcode[:12]


def sample_type_code(barcode: str) -> str | None:
    if len(barcode) >= 15 and barcode[12] == "-" and barcode[13:15].isdigit():
        return barcode[13:15]
    return None


def load_labels(root: Path | str) -> dict[str, str]:
    """A2: tipo de cáncer por barcode de paciente, leído como stream."""
    labels = {}
    for rec in StreamService(root, data_types=["clinical_cdr"]):
        if rec.kind == "table_row" and rec.key:
            label = rec.fields.get("type")
            if label is not None and str(label).strip():
                labels[str(rec.key).strip().upper()] = str(label).strip()
    return labels


def _flag(v: object) -> bool:
    """Excluye solo flags verdaderos (texto True o 1); los demás no marcan exclusión."""
    return str(v).strip().lower() in {"true", "1", "1.0"}


def _is_rna_platform(v: object) -> bool:
    """A1 proviene específicamente de IlluminaHiSeq RNASeqV2, no GA."""
    text = str(v or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    return "illuminahiseqrnaseqv2" in text


def _note_issue(v: object) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in {"", "nan", "none", "na", "no comments", "no comment", "0", "false"}


def load_quality(root: Path | str, *, check_annotations: bool = False) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Indexa filas RNA de A3 por alícuota completa y por muestra (16 caracteres).

    No se asume que una anotación de otra plataforma sirva para RNA.
    Si A3 usa una codificación de plataforma inesperada, se lanza un error:
    nunca se acepta silenciosamente un cruce de otra tecnología.
    Los flags A3 son booleanos: solo True/1 excluye, sin políticas adicionales.
    """
    exact: dict[str, list[dict]] = defaultdict(list)
    sample: dict[str, list[dict]] = defaultdict(list)
    seen = total = 0
    for rec in StreamService(root, data_types=["sample_quality"]):
        if rec.kind != "table_row":
            continue
        total += 1
        fields = rec.fields
        if not _is_rna_platform(fields.get("platform")):
            continue
        seen += 1
        aid = _id(fields.get("aliquot_barcode", ""))
        if not aid.startswith("TCGA-") or len(aid) < 16:
            continue
        reasons = []
        for field in ("Do_not_use", "AWG_excluded_because_of_pathology"):
            if field not in fields:
                raise ValueError(f"A3 requiere la columna {field!r}")
            if _flag(fields[field]):
                reasons.append(field)
        if check_annotations:
            for field in ("patient_annotation", "sample_annotation", "aliquot_annotation"):
                if _note_issue(fields.get(field)):
                    reasons.append(field)
        item = {"aliquot": aid, "sample": aid[:16], "reasons": sorted(set(reasons)), "fields": fields}
        exact[aid].append(item)
        sample[aid[:16]].append(item)
    if total == 0:
        raise FileNotFoundError("No se encontraron filas de A3 (merged_sample_quality_annotations.tsv)")
    if seen == 0:
        raise ValueError("A3 no contiene plataformas identificadas como RNASeq; revisar los valores de 'platform'")
    return dict(exact), dict(sample)


def _quality_for(barcode: str, exact: dict, sample: dict) -> tuple[bool, str]:
    """Coincidencia exacta preferida; si no, aceptar prefijo de muestra solo
    si todas las alícuotas RNA de esa muestra tienen la misma decisión.
    """
    barcode = _id(barcode)
    matches = exact.get(barcode)
    how = "alicuota"
    if not matches:
        matches = sample.get(barcode[:16], []) if len(barcode) >= 16 else []
        how = "muestra_16"
    if not matches:
        return False, "sin_correspondencia_A3_RNA"
    signatures = {tuple(m["reasons"]) for m in matches}
    if len(signatures) != 1:
        return False, "A3_ambiguo_distintas_anotaciones"
    reasons = next(iter(signatures))
    return (False, "+".join(reasons)) if reasons else (True, f"aprobada_{how}")


@dataclass
class Cohort:
    columns: np.ndarray                 # índice de columna A1
    patient_ids: list[str]
    sample_ids: list[str]
    labels: np.ndarray
    audit: list[dict]

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.labels.tolist()).items()))


def build_cohort(barcodes: Iterable[str], root: Path | str, *,
                 check_annotations: bool = False) -> Cohort:
    """Una alícuota tumoral aprobada por paciente y con etiqueta clínica A2.

    Si dos alícuotas del mismo paciente son válidas, se prioriza tipo canónico
    (01, excepto LAML=03 y SKCM=06) y luego el menor barcode alfabético.
    La política de tipos de muestra se mantiene compatible con el proyecto.
    """
    labels = load_labels(root)
    if not labels:
        raise ValueError("A2 no entregó etiquetas de 'type'; revisar hoja TCGA-CDR")
    exact, sample = load_quality(root, check_annotations=check_annotations)
    audit: list[dict] = []
    by_patient: dict[str, list[int]] = defaultdict(list)
    original = list(barcodes)
    for i, raw in enumerate(original):
        sample_id = _id(raw)
        patient_id = patient_barcode(sample_id)
        lab = labels.get(patient_id)
        code = sample_type_code(sample_id)
        accepted, why = _quality_for(sample_id, exact, sample)
        if not lab:
            reason = "sin_etiqueta_A2"
        elif code not in TUMOR_CODES:
            reason = f"tipo_muestra_no_tumoral_{code}"
        elif not accepted:
            reason = why
        else:
            reason = "candidata"
        audit.append({"column_index": i, "patient_id": patient_id, "sample_id": sample_id,
                      "type": lab or "", "sample_code": code or "", "quality_rule": why,
                      "included": False, "reason": reason})
        if reason == "candidata":
            by_patient[patient_id].append(i)
    chosen: list[int] = []
    for patient_id, candidates in sorted(by_patient.items()):
        lab = labels[patient_id]
        preferred = CANONICAL_CODE.get(lab, "01")
        best = min(candidates, key=lambda i: (sample_type_code(_id(original[i])) != preferred,
                                              _id(original[i]), i))
        chosen.append(best)
        audit[best]["included"] = True
        audit[best]["reason"] = "incluida"
        for i in candidates:
            if i != best:
                audit[i]["reason"] = "otra_alicuota_del_mismo_paciente"
    chosen.sort()
    if not chosen:
        reasons = dict(Counter(row["reason"] for row in audit))
        raise ValueError(f"No quedaron muestras elegibles. Razones: {reasons}. Revisar IDs/plataforma de A3")
    return Cohort(columns=np.asarray(chosen, dtype=np.int64),
                  patient_ids=[audit[i]["patient_id"] for i in chosen],
                  sample_ids=[audit[i]["sample_id"] for i in chosen],
                  labels=np.asarray([audit[i]["type"] for i in chosen]), audit=audit)


def save_audit(audit: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["column_index", "patient_id", "sample_id", "type", "sample_code",
                                          "quality_rule", "included", "reason"])
        w.writeheader()
        w.writerows(audit)
