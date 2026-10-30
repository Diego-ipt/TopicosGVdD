"""Pruebas de ``tcgaknn.perfil`` con una carpeta sintética de respuesta conocida.

Tres tipos de cáncer (AAA, BBB, CCC). En expresión cada muestra es
``escala * dirección_del_tipo + ruido`` con escala log-uniforme en [1, 100]:
la dirección identifica el tipo y la escala no, así que el coseno debe acertar
casi siempre y la L2 cruda (dominada por la escala) mucho menos.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tcgaknn import perfil

TYPES = ("AAA", "BBB", "CCC")
N_PER = 20
N_GENES = 60


def _patients() -> list[tuple[str, str]]:
    return [(f"TCGA-{t[:2]}-{i:04d}", t) for t in TYPES for i in range(N_PER)]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import openpyxl

    root = tmp_path_factory.mktemp("perfil")
    rng = np.random.default_rng(0)
    pats = _patients()
    # CDR: etiquetas (más un paciente etiquetado sin datos).
    (root / "cdr").mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TCGA-CDR"
    ws.append(["bcr_patient_barcode", "type"])
    for p, t in pats + [("TCGA-ZZ-9999", "AAA")]:
        ws.append([p, t])
    wb.save(root / "cdr" / "TCGA-CDR-SupplementalTableS1.xlsx")

    # Expresión (feature-major): columnas tumorales + una normal (11) y una sin etiqueta.
    base = rng.uniform(0.5, 1, N_GENES)
    dirs = {t: base + 0.4 * rng.uniform(0, 1, N_GENES) for t in TYPES}
    for t in TYPES:
        dirs[t] /= np.linalg.norm(dirs[t])
    cols, vals = [], []
    for p, t in pats:
        s = 10 ** rng.uniform(0, 2)
        cols.append(f"{p}-01A-11R-0001-07")
        vals.append(s * 100 * dirs[t] + rng.uniform(0, 0.05, N_GENES))
    cols += ["TCGA-AA-0000-11A-11R-0001-07", "TCGA-QQ-0001-01A-11R-0001-07"]
    vals += [100 * dirs["AAA"], 100 * dirs["BBB"]]
    M = np.array(vals).T  # genes × muestras
    lines = ["gene_id\t" + "\t".join(cols)]
    for g in range(N_GENES):
        row = [f"{x:.4f}" for x in M[g]]
        if g == 0:
            row[3] = "NA"
        lines.append(f"G{g}|{g}\t" + "\t".join(row))
    (root / "expr").mkdir()
    (root / "expr" / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # RPPA (sample-major) con una columna de meta (TumorType).
    prot = [f"P{j}" for j in range(12)]
    centers = {t: rng.normal(0, 1, len(prot)) for t in TYPES}
    rl = ["SampleID\tTumorType\t" + "\t".join(prot)]
    for p, t in pats:
        v = centers[t] + rng.normal(0, 0.2, len(prot))
        rl.append(f"{p}-01A\t{t}\t" + "\t".join(f"{x:.4f}" for x in v))
    (root / "rppa").mkdir()
    (root / "rppa" / "TCGA-RPPA-pancan-clean.txt").write_text("\n".join(rl) + "\n", encoding="utf-8")

    # MAF: cada tipo muta su propio bloque de genes (+ una variante silenciosa común).
    ml = ["#version 2.4", "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\tFILTER"]
    for k, (p, t) in enumerate(pats):
        block = [f"{t}_G{j}" for j in range(8)]
        for g in rng.choice(block, size=4, replace=False):
            ml.append(f"{g}\t{p}-01A-11D\tMissense_Mutation\tPASS")
        ml.append(f"TTN\t{p}-01A-11D\tSilent\tPASS")
    (root / "maf").mkdir()
    with gzip.open(root / "maf" / "mc3.v0.2.8.PUBLIC.maf.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(ml) + "\n")
    return root


@pytest.fixture(scope="module")
def labels(root: Path) -> dict[str, str]:
    return perfil.load_labels(root)


def test_labels_and_selection(labels: dict[str, str]) -> None:
    assert len(labels) == 3 * N_PER + 1
    cols = [f"{p}-01A-11R" for p, _ in _patients()] + ["TCGA-AA-0000-11A", "TCGA-QQ-0001-01A", "TCGA-AA-0001-06A"]
    idx, y, info = perfil.select_samples(cols, labels, 30, seed=2026)
    assert len(idx) == 30 and sorted(set(y)) == list(TYPES)
    assert info["elegidas_por_tipo"] == {t: 10 for t in TYPES}
    assert info["embudo"]["sin_etiqueta"] == 1
    assert info["embudo"]["excluida_codigo_11"] == 1
    assert info["embudo"]["duplicada_mismo_paciente"] == 1  # 01A preferida a 06A


def test_stratified_allocation_water_filling() -> None:
    assert perfil.stratified_allocation({"a": 2, "b": 50, "c": 50}, 30) == {"a": 2, "b": 14, "c": 14}


def test_reservoir_is_uniform() -> None:
    hits = np.zeros(50)
    for s in range(400):
        for x in perfil.reservoir_sample(range(50), 10, s):
            hits[x] += 1
    assert abs(hits.mean() - 80) < 1e-9 and hits.min() > 45 and hits.max() < 120


def test_metrics_basic() -> None:
    A = np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 1.0, 0]])
    C = perfil.cosine_distance(A)
    assert C[0, 1] == pytest.approx(0, abs=1e-12) and C[0, 2] == pytest.approx(1)
    assert perfil.euclidean_distance(A)[0, 1] == pytest.approx(1)
    B = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)
    assert perfil.jaccard_distance(B)[0, 1] == pytest.approx(2 / 3)
    assert perfil.hamming_distance(B)[0, 1] == pytest.approx(2 / 4) or perfil.hamming_distance(B)[0, 1] == 2
    assert perfil.m_value(np.array([0.5]))[0] == pytest.approx(0)
    assert perfil.log2p1(np.array([-3.0, 1.0])).tolist() == [0.0, 1.0]


def test_profile_expression_cosine_beats_raw_l2(root: Path, labels: dict[str, str]) -> None:
    info, st = perfil.profile_matrix(root, "expression", labels, n_samples=45, max_features=40, seed=2026)
    assert info["orientacion"] == "feature" and info["filas_leidas"] == N_GENES
    assert info["sketch"]["S"] == 45 and info["sketch"]["F"] == 40
    assert len(st["feature_id"]) == N_GENES
    g = info["estadisticos_por_rasgo"]["global"]
    assert g["rasgos_con_na"] == 1 and g["frac_neg"] == 0
    an = info["analisis"]
    raw = an["variantes"]["crudo"]["metricas"]
    assert raw["coseno"]["pureza@1"] >= 0.95
    assert raw["coseno"]["pureza@1"] > raw["L2"]["pureza@1"] + 0.2
    # La norma por muestra no informa del tipo (la escala es aleatoria).
    assert an["variantes"]["crudo"]["normas"]["eta2_norma_vs_tipo"] < 0.3
    assert set(an["variantes"]) == {"crudo", "transformado", "transformado+z", "transformado+z+L2"}


def test_profile_rppa_sample_major(root: Path, labels: dict[str, str]) -> None:
    info, _ = perfil.profile_matrix(root, "rppa", labels, n_samples=30, max_features=100, seed=1)
    assert info["orientacion"] == "sample" and info["sketch"]["S"] == 30 and info["sketch"]["F"] == 12
    assert info["seleccion"]["etiqueta_archivo_vs_cdr"] == {"coincide": 30}
    assert info["analisis"]["variantes"]["crudo"]["metricas"]["L2"]["pureza@5"] > 0.9


def test_profile_mutations(root: Path, labels: dict[str, str]) -> None:
    info = perfil.profile_mutations(root, labels, n_samples=30, seed=2026, top_genes=10)
    assert info["muestras_en_maf"] == 3 * N_PER and info["claves_no_contiguas"] == 0
    tz = info["tamanos_todos_los_pacientes"]
    assert tz["genes_no_silenciosas"]["q50"] == 4 and tz["genes_todas"]["q50"] == 5
    v = info["analisis"]["variantes"]
    assert v["no_silenciosas"]["metricas"]["jaccard"]["pureza@5"] > 0.9
    # Con todas las variantes, TTN es común a todos.
    assert v["todas"]["universo_genes"] == 3 * 8 + 1


def test_script_end_to_end(root: Path, tmp_path: Path) -> None:
    import perfil_datos

    out = tmp_path / "out"
    rc = perfil_datos.main([str(root), "--types", "expression", "rppa", "mutations", "--samples", "30",
                            "--max-features", "50", "--out", str(out), "--progress", "0"])
    assert rc == 0
    for t in ("expression", "rppa", "mutations"):
        assert json.loads((out / f"{t}.json").read_text(encoding="utf-8"))["tipo"] == t
    md = (out / "RESUMEN.md").read_text(encoding="utf-8")
    assert "## expression" in md and "## mutations" in md and "Pureza@5" in md
