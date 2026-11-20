from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import openpyxl
import pytest
from scipy.spatial.distance import cdist

from tcgaknn.cohorte_validada import build_cohort
from tcgaknn.experimento_validado import make_model, test as final_test, validate
from tcgaknn.preparacion import load_prepared, prepare
from tcgaknn.splits import make_split
from tcgaknn.algoritmos.exacto.exact import ExactKNN
from tcgaknn.algoritmos.euclidiana.pca_ref import PCAKNN


def make_tiny_tcga(root: Path):
    rng = np.random.default_rng(17)
    labels = {}
    sample_ids = []
    quality = []
    for i in range(18):
        pid = f"TCGA-AB-{i+1:04d}"
        cancer = ["BRCA", "LUAD", "KIRC"][i // 6]
        labels[pid] = cancer
        aid = pid + "-01A-01R-ABCD-01"
        sample_ids.append(aid)
        quality.append([pid, aid, "IlluminaHiSeq_RNASeqV2", 0, 0])
    # Se agrega una alícuota repetida de paciente; se debe escoger una.
    extra = sample_ids[0].replace("01A", "01B")
    sample_ids.append(extra)
    quality.append(["TCGA-AB-0001", extra, "IlluminaHiSeq_RNASeqV2", 0, 0])
    # Un paciente excluido por calidad; cada clase sigue teniendo al menos 3.
    quality[7][-1] = 1
    (root / "merged_sample_quality_annotations.tsv").write_text(
        "patient_barcode\taliquot_barcode\tplatform\tAWG_excluded_because_of_pathology\tDo_not_use\n"
        + "\n".join("\t".join(map(str, row)) for row in quality) + "\n", encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TCGA-CDR"
    ws.append(["bcr_patient_barcode", "type"])
    for k, v in labels.items():
        ws.append([k, v])
    wb.save(root / "TCGA-CDR-SupplementalTableS1.xlsx")
    with (root / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv").open("w") as f:
        f.write("gene\t" + "\t".join(sample_ids) + "\n")
        for j in range(40):
            a = np.array([i // 6 for i in range(18)] + [0])
            v = 4 * (a == j % 3) + rng.normal(0, 0.2, size=len(a))
            if j == 0:
                v[2] = np.nan
            f.write(f"GENE{j}\t" + "\t".join(f"{t:.7f}" for t in v) + "\n")
    return sample_ids


def test_end_to_end_no_leakage_and_fixed_test(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sample_ids = make_tiny_tcga(raw)
    cohort = build_cohort(sample_ids, raw)
    assert len(cohort.columns) == 17  # 18 pacientes, 1 excluido por calidad
    assert len(set(cohort.patient_ids)) == 17
    assert any(r["reason"] == "Do_not_use" for r in cohort.audit)
    assert any(r["reason"] == "otra_alicuota_del_mismo_paciente" for r in cohort.audit)
    processed = tmp_path / "processed"
    meta = prepare(raw, processed, seed=12)
    assert meta["cohort_total"] == 17
    X, npz, _ = load_prepared(processed, verify_hash=True)
    assert X.shape == (17, meta["genes_used"])
    assert np.isfinite(X).all()
    tr, val, te = (npz[s] for s in ("train", "val", "test"))
    assert set(tr) .isdisjoint(set(val)) and set(te).isdisjoint(set(tr))
    # Solo test elige los valores de consultas, jamás los agrega a la base.
    out = tmp_path / "results"
    v = validate(processed, out, metric="cosine", k=2, seed=5,
                 methods={"rp", "simhash", "lsh", "simhash_rp"})
    assert (out / "configuraciones_congeladas.json").is_file()
    assert len(v["results"]) >= 2
    t = final_test(processed, out / "configuraciones_congeladas.json", out)
    assert t["phase"] == "test_final"
    assert all(0 <= row["metrics"]["recall_at_k"] <= 1 for row in t["results"])
    sel = json.loads((out / "configuraciones_congeladas.json").read_text())
    sel["dataset_id"] = "cambiado"
    (out / "seleccion_falsa.json").write_text(json.dumps(sel))
    with pytest.raises(ValueError, match="otra cohorte"):
        final_test(processed, out / "seleccion_falsa.json", out)


def test_stratification_rejects_duplicates_and_rare_classes():
    with pytest.raises(ValueError, match="una fila por paciente"):
        make_split(["P1", "P1", "P2"], np.array(["A", "A", "B"]))
    with pytest.raises(ValueError, match="solo 2 pacientes"):
        make_split(["P1", "P2"], np.array(["A", "A"]))


def test_l2_exact_tricky_pairs_against_scipy():
    rng = np.random.default_rng(77)
    X = (2000 + rng.normal(0, 1e-2, size=(50, 30))).astype(np.float32)
    Q = (2000 + rng.normal(0, 1e-2, size=(7, 30))).astype(np.float32)
    ids, distances, _ = ExactKNN("l2").fit(X).query(Q, k=8)
    direct = cdist(Q.astype(np.float64), X.astype(np.float64))
    assert np.array_equal(ids, np.argsort(direct, axis=1, kind="stable")[:, :8])
    np.testing.assert_allclose(distances, np.take_along_axis(direct, ids, axis=1), rtol=1e-6)


def test_pca_resident_memory_includes_keep_data():
    x = np.random.default_rng(1).normal(size=(40, 10)).astype(np.float32)
    p = PCAKNN(m=6, keep_data=True).fit(x)
    assert p.resident_bytes() >= p.index_bytes() + p._Xn.nbytes
    p2 = PCAKNN(m=6, keep_data=False).fit(x)
    assert p2.resident_bytes() < p.resident_bytes()


def test_algorithm_metric_guardrails():
    with pytest.raises(ValueError, match="solo aproxima"):
        make_model({"method": "simhash", "b": 256}, "l2")
    with pytest.raises(ValueError, match="distancia"):
        make_model({"method": "e2lsh", "p": 1, "w": 1, "r": 2, "L": 4}, "cosine")


def test_preprocessing_statistics_use_train_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    samples = make_tiny_tcga(raw)
    first = tmp_path / "first"
    prepare(raw, first, seed=6)
    X1, data1, _ = load_prepared(first)
    train = data1["train"]
    val = data1["val"]
    affected_sample = str(data1["sample_ids"][int(val[0])])
    # Alterar SOLO un valor de validación, conservando headers y el resto.
    a1 = raw / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
    content = a1.read_text().splitlines()
    col = content[0].split("\t").index(affected_sample)
    row = content[1].split("\t")
    row[col] = "999999.0"
    content[1] = "\t".join(row)
    a1.write_text("\n".join(content) + "\n")
    second = tmp_path / "second"
    prepare(raw, second, seed=6)
    X2, data2, _ = load_prepared(second)
    np.testing.assert_array_equal(data1["train"], data2["train"])
    np.testing.assert_allclose(data1["gene_mean_train"], data2["gene_mean_train"])
    np.testing.assert_allclose(data1["gene_std_train"], data2["gene_std_train"])
    np.testing.assert_allclose(np.asarray(X1[train]), np.asarray(X2[train]))
    assert not np.allclose(np.asarray(X1[val]), np.asarray(X2[val]))


def test_quality_only_matches_hiseq_and_ambiguous_samples_are_excluded(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    samples = make_tiny_tcga(raw)
    path = raw / "merged_sample_quality_annotations.tsv"
    text = path.read_text()
    # Solo GA en una alícuota A1: no se debe usar como validación de HiSeq.
    text = text.replace("TCGA-AB-0003-01A-01R-ABCD-01\tIlluminaHiSeq_RNASeqV2",
                        "TCGA-AB-0003-01A-01R-ABCD-01\tIlluminaGA_RNASeqV2")
    path.write_text(text)
    cohort = build_cohort(samples, raw)
    assert "TCGA-AB-0003" not in cohort.patient_ids
    assert any(r["reason"] == "sin_correspondencia_A3_RNA" for r in cohort.audit)


def test_confusion_counts_abstentions():
    from tcgaknn.experimento_validado import _predict_metrics
    y = np.array(["A", "A", "B", "B"])
    p = np.array(["", "A", "", "A"])
    rep = _predict_metrics(y, p, np.array(["A", "B"]))
    assert rep["n_no_prediction"] == 2
    assert rep["coverage"] == 0.5
    assert rep["confusion_columns"] == ["A", "B", "SIN_PREDICCION"]
    assert np.sum(rep["confusion_with_abstention"]) == len(y)
    assert rep["per_class"][1]["fnr"] == 1.0
