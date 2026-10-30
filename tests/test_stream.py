"""Tests del servicio de stream.

- Sintéticos: una carpeta pequeña con un archivo de cada formato, en subcarpetas
  por UUID como las de gdc-client. Corren en cualquier equipo.
- Datos reales: pocas instancias por tipo de dato desde la carpeta de descarga.
  Se saltan si la carpeta no existe (ruta en la variable TCGA_RAW_DIR).
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from tcgaknn.stream import SOURCES, StreamService, discover

SAMPLES = ["TCGA-AA-0001-01A-11R-A000-07", "TCGA-AA-0002-01A-11R-A000-07", "TCGA-AA-0003-11A-11R-A000-07"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "gdc"
    tab = "\t"
    # Expresión: IDs entre comillas, un NA, 4 genes.
    header = tab.join(['"gene_id"'] + [f'"{s}"' for s in SAMPLES])
    rows = [
        tab.join(['"TP53|7157"', "1.5", "0", "NA"]),
        tab.join(['"?|100130426"', "0", "0.3861", "2"]),
        tab.join(['"BRCA1|672"', "3", "4", "5"]),
        tab.join(['"EGFR|1956"', "7", "8", "9"]),
    ]
    _write(root / "uuid-expr" / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv",
           "\n".join([header, *rows]) + "\n")
    _write(root / "uuid-expr" / "logs" / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv.parcel", "log")
    # Metilación 450K: primera celda del encabezado vacía.
    _write(root / "uuid-met" / "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv",
           tab.join([""] + SAMPLES) + "\ncg00000029\t0.1\t0.2\tNA\ncg00000108\t0.9\t0.8\t0.7\n")
    # miRNA: CSV con columna extra "Correction".
    _write(root / "uuid-mir" / "pancanMiRs_EBadj_test.csv",
           "Genes,Correction," + ",".join(SAMPLES) + "\nhsa-let-7a,Corrected,1,2,3\nhsa-mir-21,Uncorrected,4,5,6\n")
    # RPPA: muestras en filas.
    _write(root / "uuid-rppa" / "TCGA-RPPA-pancan-clean.txt",
           "SampleID\tTumorType\tAKT\tBAX\nTCGA-AA-0001-01A\tACC\t0.5\t-1\nTCGA-AA-0002-01A\tBRCA\tNA\t2\n")
    # Mutaciones: MAF comprimido con línea de comentario.
    maf = "#version 2.4\nHugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\nTP53\tTCGA-AA-0001-01A\tMissense\nKRAS\tTCGA-AA-0002-01A\tMissense\n"
    (root / "uuid-maf").mkdir(parents=True)
    with gzip.open(root / "uuid-maf" / "mc3.v0.2.8.PUBLIC.maf.gz", "wt", encoding="utf-8") as fh:
        fh.write(maf)
    # Número de copias.
    _write(root / "uuid-seg" / "broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.seg",
           "Sample\tChromosome\tStart\tEnd\tNum_Probes\tSegment_Mean\nTCGA-AA-0001-01A\t1\t100\t200\t5\t0.1\n")
    # PARADIGM dentro de un .tar.gz.
    paradigm = b"Gene\tTCGA-AA-0001\tTCGA-AA-0002\nUBE2Q1\t0\t-1.5\n"
    (root / "uuid-par").mkdir(parents=True)
    with tarfile.open(root / "uuid-par" / "merge_merged_reals.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("merge_merged_reals.txt")
        info.size = len(paradigm)
        tar.addfile(info, io.BytesIO(paradigm))
    # CDR en Excel.
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TCGA-CDR"
    ws.append([None, "bcr_patient_barcode", "type"])
    ws.append([1, "TCGA-AA-0001", "ACC"])
    ws.append([2, "TCGA-AA-0002", "BRCA"])
    (root / "uuid-cdr").mkdir(parents=True)
    wb.save(root / "uuid-cdr" / "TCGA-CDR-SupplementalTableS1.xlsx")
    # Archivo desconocido.
    _write(root / "uuid-x" / "otra_cosa.txt", "x\n")
    return root


def test_discover_classifies_and_ignores_logs(synthetic_root: Path) -> None:
    found, unknown = discover(synthetic_root)
    types = sorted(s.data_type for s in found)
    assert types == sorted(["expression", "methylation_450k", "mirna", "rppa", "mutations",
                            "copy_number", "paradigm", "clinical_cdr"])
    assert [p.name for p in unknown] == ["otra_cosa.txt"]  # el .parcel de logs/ no aparece


def test_stream_limit_and_parsing(synthetic_root: Path) -> None:
    by_type = defaultdict(list)
    for rec in StreamService(synthetic_root, limit_per_type=2):
        by_type[rec.data_type].append(rec)

    expr = by_type["expression"]
    assert [r.kind for r in expr] == ["schema", "feature_row", "feature_row", "end"]
    assert expr[0].fields["columns"] == SAMPLES  # sin comillas
    assert expr[1].key == "TP53|7157"
    assert expr[1].values.dtype == np.float32
    assert np.isnan(expr[1].values[2])
    assert expr[-1].fields == {"records": 2, "truncated": True}

    mir = by_type["mirna"]
    assert mir[1].fields == {"Correction": "Corrected"}
    np.testing.assert_array_equal(mir[1].values, [1, 2, 3])

    rppa = by_type["rppa"]
    assert rppa[1].kind == "sample_row" and rppa[1].fields == {"TumorType": "ACC"}
    assert rppa[0].fields["columns"] == ["AKT", "BAX"]

    muts = [r for r in by_type["mutations"] if r.kind == "event"]
    assert [r.key for r in muts] == ["TCGA-AA-0001-01A", "TCGA-AA-0002-01A"]  # salta el comentario

    assert by_type["paradigm"][1].key == "UBE2Q1"
    cdr = [r for r in by_type["clinical_cdr"] if r.kind == "table_row"]
    assert [(r.key, r.fields["type"]) for r in cdr] == [("TCGA-AA-0001", "ACC"), ("TCGA-AA-0002", "BRCA")]
    assert by_type["copy_number"][-1].fields == {"records": 1, "truncated": False}


def test_stream_filter_types(synthetic_root: Path) -> None:
    recs = list(StreamService(synthetic_root, data_types=["rppa"]))
    assert {r.data_type for r in recs} == {"rppa"}
    with pytest.raises(ValueError):
        StreamService(synthetic_root, data_types=["methylation_27k_450k"])


def test_early_break_releases_producer(synthetic_root: Path) -> None:
    # Cola de tamaño 1: el productor queda bloqueado; cortar el consumo no debe colgar el test.
    service = StreamService(synthetic_root, queue_size=1)
    stream = service.stream()
    next(stream)
    stream.close()
    assert not any(t.name == "tcga-stream" and t.is_alive() for t in threading.enumerate())


def test_malformed_row_propagates_error(tmp_path: Path) -> None:
    _write(tmp_path / "u" / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv",
           "gene_id\tS1\tS2\nG1\t1\n")
    with pytest.raises(ValueError, match="columnas"):
        list(StreamService(tmp_path))


# --------------------------------------------------------------------------
# Datos reales: unas pocas instancias por tipo de dato
# --------------------------------------------------------------------------

RAW_DIR = Path(os.environ.get("TCGA_RAW_DIR", r"D:\Universidad\10mo semestre\Topicos\gdc-client_2.3_Windows_x64"))
N_REAL = 3


@pytest.fixture(scope="module")
def real_records() -> dict[str, list]:
    if not RAW_DIR.is_dir():
        pytest.skip(f"No está la carpeta de datos reales ({RAW_DIR})")
    by_type = defaultdict(list)
    for rec in StreamService(RAW_DIR, limit_per_type=N_REAL):
        by_type[rec.data_type].append(rec)
    return by_type


@pytest.mark.parametrize("data_type", [s.data_type for s in SOURCES])
def test_real_data_few_instances_per_type(real_records: dict[str, list], data_type: str) -> None:
    recs = real_records[data_type]
    assert recs, f"no llegó ningún registro de {data_type}"
    data = [r for r in recs if r.kind not in ("schema", "end")]
    assert len(data) == N_REAL
    assert recs[-1].kind == "end" and recs[-1].fields["records"] == N_REAL

    schema = recs[0] if recs[0].kind == "schema" else None
    for r in data:
        assert r.key, "registro sin clave"
        if schema is not None:
            assert r.values.shape == (len(schema.fields["columns"]),)
            assert r.values.dtype == np.float32
    if schema is not None and schema.fields["orientation"] == "feature":
        # Las columnas son barcodes TCGA (de alícuota o de paciente en PARADIGM).
        assert all(c.startswith("TCGA-") for c in schema.fields["columns"][:50])
    if data_type in ("rppa", "mutations", "copy_number", "absolute_segments", "absolute_purity",
                     "clinical_cdr", "clinical_followup", "sample_quality", "mirna_samples"):
        assert all(r.key.startswith("TCGA-") for r in data)
