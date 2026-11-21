"""La intersección A1/A2/A3 y la regla mínima de flags True/False."""
import shutil
from pathlib import Path

import numpy as np

from tcgaknn.cohorte_validada import build_cohort, load_quality
from tcgaknn.preparacion_streaming import DEFAULT_TRANSFORM

RAW = Path(__file__).parents[1] / "demo" / "raw"
A1 = "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
A3 = "merged_sample_quality_annotations.tsv"


def test_quality_flags_and_a1_a2_a3(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    barcodes = (raw / A1).read_text().splitlines()[0].split("\t")[1:]
    original = build_cohort(barcodes, raw)
    assert len(original.patient_ids) == 30
    a3 = raw / A3
    lines = a3.read_text().splitlines()
    assert "Do_not_use" in lines[0]
    cells = lines[1].split("\t")
    cells[4] = "TrUe"  # excluir incluso con mayúsculas variables
    lines[1] = "\t".join(cells)
    cells = lines[2].split("\t")
    cells[3] = "1"  # AWG True
    lines[2] = "\t".join(cells)
    cells = lines[3].split("\t")
    cells[4] = "False"
    lines[3] = "\t".join(cells)
    a3.write_text("\n".join(lines) + "\n")
    exact, sample = load_quality(raw)
    assert exact[barcodes[0].upper()][0]["reasons"] == ["Do_not_use"]
    assert sample[barcodes[1].upper()[:16]][0]["reasons"] == ["AWG_excluded_because_of_pathology"]
    cohort = build_cohort(barcodes, raw)
    assert len(cohort.patient_ids) == 28
    assert barcodes[0][:12] not in cohort.patient_ids
    assert barcodes[1][:12] not in cohort.patient_ids
    assert any(x["reason"] == "Do_not_use" for x in cohort.audit)
    assert any(x["reason"] == "AWG_excluded_because_of_pathology" for x in cohort.audit)
    assert DEFAULT_TRANSFORM == "log2clip"


def test_filter_platform_and_unique_patient(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    barcodes = (raw / A1).read_text().splitlines()[0].split("\t")[1:]
    a3 = raw / A3
    a3.write_text(a3.read_text().replace(
        barcodes[0] + "\tIlluminaHiSeq_RNASeqV2", barcodes[0] + "\tIlluminaGA_RNASeqV2"))
    cohort = build_cohort(barcodes + [barcodes[2]], raw)
    assert barcodes[0][:12] not in cohort.patient_ids
    assert len(set(cohort.patient_ids)) == len(cohort.patient_ids)
    assert np.all(np.diff(cohort.columns) > 0)
