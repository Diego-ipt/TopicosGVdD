"""Prueba rápida de ``tcgaknn.completo`` sobre la carpeta sintética de ``test_perfil``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from test_perfil import root  # noqa: F401  (fixture con la carpeta sintética)
from tcgaknn import completo
from tcgaknn.algoritmos import ExactKNN, l2_normalize


def test_stratified_queries() -> None:
    y = np.array(["A"] * 50 + ["B"] * 10 + ["C"] * 3 + ["D"])
    q = completo.stratified_queries(y, 10, seed=0)
    lab, cnt = np.unique(y[q], return_counts=True)
    assert dict(zip(lab, cnt))["C"] == 2 and "D" not in lab  # mínimo 2; D tiene 1 → queda en la base
    assert len(set(q)) == q.size


def test_run_on_synthetic(root: Path, tmp_path: Path) -> None:  # noqa: F811
    res = completo.run(root, tmp_path, n_queries=12, b=64, m=64, f=5, k=5, sizes=(10, 30),
                       bootstrap=50, log=lambda s: None)
    s = res["busqueda"]
    assert 0.6 < s["recall@5"] <= 1.0 and s["perdida_similitud_topk"] >= -1e-6
    err = res["error_estimacion"]
    assert abs(err["rp_todos"]["sesgo"]) < 0.1 and err["rp_todos"]["correlacion"] > 0.8
    assert err["simhash_todos"]["correlacion"] > 0.7
    # El objetivo: predecir el tipo de cáncer
    c = res["clasificacion"]["k=5"]
    assert c["exacto"]["f1_macro"] > 0.8 and c["aproximado"]["f1_macro"] > 0.8
    assert abs(c["aproximado"]["tasa_de_error"] - (1 - c["aproximado"]["accuracy"])) < 1e-12
    assert 0 <= c["acuerdo_de_voto"] <= 1
    assert res["k_principal"] == 5 and {r["tipo"] for r in res["por_clase"]} == {"AAA", "BBB", "CCC"}
    assert list(res)[:3] == ["metodo", "k_principal", "clasificacion"]
    assert (tmp_path / "predicciones.csv").read_text(encoding="utf-8").count("\n") == 13  # encabezado + 12 consultas
    lo, hi = c["delta_f1_macro_IC95"]
    assert lo <= hi
    assert [e["n"] for e in res["escalamiento"]][-1] == res["pasada"]["base"]
    js = json.loads((tmp_path / "metricas.json").read_text(encoding="utf-8"))
    assert js["parametros"]["b"] == 64


def test_streaming_truth_matches_in_memory(root: Path) -> None:  # noqa: F811
    """El coseno exacto acumulado en el stream coincide con ExactKNN sobre Z materializada."""
    from tcgaknn.escalabilidad import _standardize
    from tcgaknn.perfil import load_labels, log2p1, select_samples
    from tcgaknn.stream import StreamService

    data = completo.stream_pass(root, n_queries=12, seed=2026, b=64, m=64, log=lambda s: None)
    labels = load_labels(root)
    zs = []
    for rec in StreamService(root, data_types=["expression"]):
        if rec.kind == "schema":
            cols, _, _ = select_samples(rec.fields["columns"], labels, len(rec.fields["columns"]), 2026)
            mask = np.zeros(cols.size, dtype=bool)
            mask[data["base"]] = True
        elif rec.kind == "feature_row":
            z = _standardize(log2p1(rec.values[cols].astype(np.float64)), mask)
            if z is not None:
                zs.append(z)
    Z = l2_normalize(np.stack(zs, axis=1))
    ref = Z[data["q"]] @ Z[data["base"]].T
    np.testing.assert_allclose(data["S"], ref, atol=1e-5)
    _, sims, _ = ExactKNN("cosine").fit(Z[data["base"]]).query(Z[data["q"]], 3)
    np.testing.assert_allclose(np.sort(data["S"], axis=1)[:, ::-1][:, :3], sims, atol=1e-5)
