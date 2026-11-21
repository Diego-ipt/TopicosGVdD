"""Pruebas integradas: streaming efectivo, verdad L1/coseno y selección congelada."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from tcgaknn.algoritmos.base import topk
from tcgaknn.algoritmos.lp.e2lsh import E2LSH
from tcgaknn.experimento_streaming import _phase, _truth, final_test_streaming, validate_streaming
from tcgaknn.preparacion_streaming import load_sketches, prepare_streaming, _transform
from tcgaknn.stream import StreamService

RAW = Path(__file__).parents[1] / "demo" / "raw"


@pytest.fixture()
def prepared(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    out = tmp_path / "prepared"
    prepare_streaming(raw, out, max_genes=32, cols=256, lp_cols=64,
                      queries_per_phase=0, seed=6, batch_size=7)
    return raw, out


def test_dense_oracle_only_in_test_and_no_dense_project(prepared):
    raw, out = prepared
    assert not (out / "X_gene_major.npy").exists()
    Y, Y2, Y1, sq, G, D1, spl, meta = load_sketches(out, phase="val", verify_hash=True)
    genes = set(spl["feature_ids"].tolist())
    rows = []
    for rec in StreamService(raw, data_types=["expression"]):
        if rec.kind != "feature_row" or rec.key not in genes:
            continue
        raw_row = rec.values.astype(np.float64)
        positions = [int(i) for i in range(len(raw_row))]  # 30 filas, 1 por paciente en demo
        v = _transform(raw_row[positions], "log2clip")
        ix = list(spl["feature_ids"]).index(rec.key)
        z = (v - spl["gene_mean_train"][ix]) / spl["gene_std_train"][ix]
        rows.append(np.where(np.isfinite(z), z, 0).astype(np.float32))
    Z = np.stack(rows, axis=1)
    # Las matrices grandes solo se materializan aquí para probar el resultado numérico.
    q = spl["query_val"]
    base = spl["base_order"]
    np.testing.assert_allclose(np.asarray(G), Z[q].astype(float) @ Z[base].astype(float).T, atol=1e-5)
    np.testing.assert_allclose(np.asarray(D1), cdist(Z[q].astype(float), Z[base].astype(float), "cityblock"), atol=1e-4)
    for p, saved in ((1, Y1), (2, Y2)):
        model = E2LSH(p=p, L=64, r=1, seed=6, rerank="proj")
        projected = Z.astype(np.float64) @ model.a_matrix(list(spl["feature_ids"])).astype(np.float64)
        np.testing.assert_allclose(saved, projected, atol=2e-4, rtol=1e-5)
    data = _phase(out, "val", True)
    for n in (5, len(base)):
        for metric in ("l1", "cosine"):
            result, _, timing = _truth(data, n, 2, metric)
            assert timing["median_ms"] >= 0
            scores = (cdist(Z[q].astype(float), Z[base[:n]].astype(float), "cityblock")
                      if metric == "l1" else
                      (Z[q] / np.linalg.norm(Z[q], axis=1)[:, None]) @
                      (Z[base[:n]] / np.linalg.norm(Z[base[:n]], axis=1)[:, None]).T)
            expected = topk(scores, 2, largest=(metric == "cosine"))[0]
            np.testing.assert_array_equal(result, expected)


def test_train_only_statistics_immune_to_query(prepared, tmp_path):
    raw, base_dir = prepared
    with np.load(base_dir / "splits.npz") as dat:
        qid = str(dat["sample_ids"][int(dat["query_val"][0])])
        mu = dat["gene_mean_train"].copy()
        sd = dat["gene_std_train"].copy()
    a1 = next(raw.glob("*geneExp.tsv"))
    lines = a1.read_text().splitlines()
    col = lines[0].split("\t").index(qid)
    parts = lines[1].split("\t")
    parts[col] = "999999"
    lines[1] = "\t".join(parts)
    a1.write_text("\n".join(lines) + "\n")
    again = tmp_path / "again"
    prepare_streaming(raw, again, max_genes=32, cols=256, lp_cols=64, seed=6, batch_size=7)
    with np.load(again / "splits.npz") as dat:
        np.testing.assert_array_equal(mu, dat["gene_mean_train"])
        np.testing.assert_array_equal(sd, dat["gene_std_train"])


def test_validate_freeze_test_all_methods_and_nested_bases(prepared, tmp_path):
    _, s = prepared
    results = tmp_path / "results"
    report = validate_streaming(s, results, k=2, sizes=[5, 10, 18], verify_hash=True)
    assert report["sizes"] == [5, 10, 18]
    for step in report["results"]:
        assert not [r for r in step["results"] if "error" in r]
        assert {r["config"]["method"] for r in step["results"]} == {
            "exact", "exact_l1", "lsh", "e2lsh_p1", "e2lsh_p2"}
        assert {r["metric"] for r in step["results"]} == {"cosine", "l1"}
    frozen = json.loads((results / "configuraciones_congeladas.json").read_text())
    assert all(len(x["selected"]) == 5 for x in frozen["selected_by_size"])
    done = final_test_streaming(s, results / "configuraciones_congeladas.json", results)
    assert done["phase"] == "test_final" and len(done["results"]) == 3
    assert all(len(r["results"]) == 5 for r in done["results"])
    frozen["dataset_id"] = "corrupto"
    (results / "invalida.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="otra cohorte"):
        final_test_streaming(s, results / "invalida.json", results)
    with pytest.raises(ValueError, match="semilla"):
        validate_streaming(s, results, k=2, sizes=[10], seed=999)


def test_independent_projection_blocks_as_replicates(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    out = tmp_path / "prep"
    prepare_streaming(raw, out, max_genes=12, cols=384, lp_cols=96,
                      queries_per_phase=2, seed=6, batch_size=6)
    report = validate_streaming(out, tmp_path / "results", k=2, sizes=[5], replicas=2)
    configs = [r["config"] for r in report["results"][0]["results"]]
    assert {x.get("replica") for x in configs if x["method"] == "lsh"} == {0, 1}
    selected = json.loads((tmp_path / "results" / "configuraciones_congeladas.json").read_text())
    assert len(selected["selected_by_size"][0]["selected"]) == 8  # 2 exact + 3 métodos × 2
    assert not [r.get("error") for r in report["results"][0]["results"] if "error" in r]
    done = final_test_streaming(out, tmp_path / "results" / "configuraciones_congeladas.json",
                                tmp_path / "results")
    assert len(done["results"][0]["results"]) == 8


def test_selection_considers_candidates_not_only_recall():
    from tcgaknn.experimento_streaming import select_frozen
    def row(r, candidates, recall, accuracy, quality, budget):
        return {"config": {"method": "e2lsh_p2", "replica": 0, "r": r},
                "metrics": {"meets_efficiency_and_quality": quality and budget,
                            "meets_quality_targets": quality,
                            "fraction_candidates": candidates,
                            "recall_at_k": recall, "accuracy": accuracy,
                            "coverage": 1.0, "latency_ms_median": 3.0,
                            "vote_concordance_vs_exact": 0.9}}
    rows = [row(2, 1.0, 0.99, 0.95, True, False),
            row(4, 0.35, 0.55, 0.94, True, True),
            row(8, 0.05, 0.10, 0.50, False, True)]
    chosen = select_frozen(rows)[0]
    assert chosen["config"]["r"] == 4
    assert chosen["qualified"]
    assert chosen["validation_fraction_candidates"] == 0.35
    rows = [row(2, 0.98, 0.98, 0.9, False, False),
            row(4, 0.25, 0.4, 0.89, False, True)]
    chosen = select_frozen(rows)[0]
    assert not chosen["qualified"]
    assert chosen["config"]["r"] == 4


def test_stratified_order_first_represents_each_training_class():
    from tcgaknn.splits import stratified_nested_order
    labels = np.array(list("AAAAAAAABBBBCCD"))
    train = np.arange(len(labels), dtype=int)
    seq = stratified_nested_order(train, labels, seed=42)
    np.testing.assert_array_equal(np.sort(seq), train)
    assert len(set(labels[seq[:len(np.unique(labels))]])) == len(np.unique(labels))
    np.testing.assert_array_equal(seq, stratified_nested_order(train, labels, seed=42))


@pytest.mark.parametrize("p", [1, 2])
def test_e2lsh_uses_all_saved_columns_for_reranking(p):
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(17, 64))
    Q = rng.normal(size=(4, 64))
    if p == 1:
        # Las dimensiones son sketches Cauchy ya generados: la API acepta sus valores.
        Y *= 4
        Q *= 4
    idx = E2LSH(p=p, L=2, r=2, w=1e10, seed=6, rerank="proj").fit_projections(
        Y[:, :4], rerank_projections=Y)
    assert idx._Y.shape[1] == 64
    result, _, stats = idx.query_projections(Q[:, :4], 3, rerank_query_projections=Q)
    assert result.shape == (4, 3)
    assert np.all(stats["n_candidates"] == 17)
    # Cuando todos son candidatos, el índice debe ordenar por las 64 columnas.
    D = np.abs(Q[:, None, :] - Y[None, :, :])
    expected = np.median(D, axis=2) if p == 1 else np.sqrt((D**2).mean(axis=2))
    assert np.array_equal(result, np.argsort(expected, axis=1)[:, :3])
    with pytest.raises(ValueError, match="rerank_query_projections"):
        idx.query_projections(Q[:, :4], 3)


def test_csv_and_full_exact_stream_benchmark(prepared, tmp_path):
    import csv
    raw, data = prepared
    results_dir = tmp_path / "results_full_exact"
    report = validate_streaming(data, results_dir, k=2, sizes=[5], methods={"lsh"},
                                exact_benchmark_raw=raw)
    assert (results_dir / "validation.csv").exists()
    assert (results_dir / "exact_streaming_val.csv").exists()
    with (results_dir / "validation.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(report["results"][0]["results"])
    assert {r["method"] for r in rows} == {"exact", "exact_l1", "lsh"}
    assert any(r["amortized_ratio_vs_exact_stream"] for r in rows if r["method"] == "lsh")
    with (results_dir / "exact_streaming_val.csv").open() as f:
        baseline = list(csv.DictReader(f))
    assert len(baseline) == 2
    assert all(float(r["max_abs_score_diff_vs_prepared"]) < 1e-3 for r in baseline)
    assert all(float(r["amortized_compute_ms_per_query"]) > 0 for r in baseline)


def test_csv_n_is_base_size_not_number_of_queries(prepared, tmp_path):
    """Regresión: _predict_metrics['n'] no puede sobrescribir step['n']."""
    import csv
    from tcgaknn.experimento_streaming import _write_csv
    out = tmp_path / 'correcto.csv'
    report = {
        'phase': 'test_final', 'k': 2, 'n_queries': 428,
        'results': [
            {'n': size, 'results': [
                {'metric': 'cosine', 'config': {'method': 'exact'},
                 'metrics': {'n': 428, 'accuracy': 0.9}}
            ]} for size in (50, 250, 1991)
        ],
    }
    _write_csv(out, report)
    with out.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row['n']) for row in rows] == [50, 250, 1991]
    assert all(int(row['n_queries']) == 428 for row in rows)


def test_quality_policy_original_targets_and_recall():
    from tcgaknn.experimento_streaming import QUALITY_POLICY, annotate_quality
    assert QUALITY_POLICY == {
        'max_accuracy_loss': 0.02, 'max_f1_loss': 0.03,
        'max_class_sensitivity_loss_support_ge_10': 0.05,
        'min_recall_at_k': 0.90, 'min_coverage': 0.90,
        'max_candidate_fraction': 0.50,
    }
    exact = {'accuracy': 0.90, 'f1_macro': 0.84,
             'per_class': [{'type': 'A', 'sensitivity': 0.9}]}
    s = {'accuracy': 0.90, 'f1_macro': 0.84, 'coverage': 1.0,
         'recall_at_k': 0.89, 'fraction_candidates': 0.30,
         'per_class': [{'type': 'A', 'support': 20, 'sensitivity': 0.9}]}
    annotate_quality(s, exact)
    assert not s['meets_quality_targets']
    assert not s['meets_efficiency_and_quality']
    s['recall_at_k'] = 0.90
    annotate_quality(s, exact)
    assert s['meets_quality_targets']
    assert s['meets_efficiency_and_quality']
    s['accuracy'] = 0.879
    annotate_quality(s, exact)
    assert not s['meets_quality_targets']


def test_paired_single_query_benchmark(prepared, tmp_path):
    """El exacto y el ANN consultan el mismo paciente y el mismo n.

    Es un benchmark opcional: el pipeline de entrenamiento no carga matrices densas.
    """
    import csv
    from tcgaknn.benchmark_pareado import benchmark_pareado
    from tcgaknn.experimento_streaming import QUALITY_POLICY
    raw, data = prepared
    results = tmp_path / 'results_paired'
    validate_streaming(data, results, k=2, sizes=[5, 10], methods={'lsh', 'e2lsh_p1', 'e2lsh_p2'})
    path = results / 'pareado_test.csv'
    rows = benchmark_pareado(data, raw, results / 'configuraciones_congeladas.json', path,
                             queries=2, max_dense_mib=256, log=lambda s: None)
    assert path.is_file()
    assert {x['n'] for x in rows} == {5, 10}
    assert all(x['n_queries'] == 2 for x in rows)
    assert {x['method'] for x in rows} == {
        'exact_dense_cosine', 'exact_dense_l1', 'lsh', 'e2lsh_p1', 'e2lsh_p2',
    }
    assert all(x['exact_single_query_median_ms'] > 0 for x in rows)
    assert all(x['speedup_vs_dense_exact'] is not None for x in rows
               if not x['method'].startswith('exact'))
    assert all(x['ann_projection_median_ms'] > 0 for x in rows
               if not x['method'].startswith('exact'))
    with path.open(newline='', encoding='utf-8') as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    frozen = json.loads((results / 'configuraciones_congeladas.json').read_text())
    assert frozen['quality_thresholds'] == QUALITY_POLICY
    frozen['quality_thresholds']['max_accuracy_loss'] = 0.10
    (results / 'policy_changed.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='política de calidad'):
        final_test_streaming(data, results / 'policy_changed.json', results)
