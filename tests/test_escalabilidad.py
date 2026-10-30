"""Prueba rápida de ``tcgaknn.escalabilidad`` sobre la carpeta sintética de ``test_perfil``.

Tres tipos con direcciones distintas en expresión y bloques de genes propios en
mutaciones: los métodos deben recuperar vecinos del mismo tipo y el exacto debe
tener recall 1 contra sí mismo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from test_perfil import root  # noqa: F401  (fixture con la carpeta sintética)
from tcgaknn import escalabilidad


def test_run_on_synthetic(root: Path, tmp_path: Path) -> None:  # noqa: F811
    res = escalabilidad.run(root, tmp_path, g_max=40, n_max=40, n_queries=12, sizes=(10, 30, 48),
                            k=5, log=lambda m: None)
    rows = res["expresion"]
    assert {r["n"] for r in rows} == {10, 30}  # 48 > base disponible: se omite
    ex = [r for r in rows if r["metodo"].startswith("EX")]
    assert all(r["recall_at_k"] == 1.0 for r in ex)
    # Con n = 30 hay 10 pacientes por tipo (con n = 10, solo ~3: la pureza@5 no puede pasar de ~0,6)
    assert all(r["pureza_at_5"] > 0.75 for r in ex if r["n"] == 30)
    # Los sketches acumulados en el stream deben aproximar bien al exacto (incluido E2LSH,
    # cuyas proyecciones se normalizan con las normas del acumulador)
    for r in rows:
        if r["n"] == 30 and not r["metodo"].startswith("EX"):
            assert r["recall_at_k"] > 0.6, r
    methods = {r["metodo"] for r in rows}
    assert {"RP-256", "SimHash-256", "LSH bandas 16×12", "E2LSH-L2", "E2LSH-L1"} <= methods
    for r in rows:
        assert 0.0 <= r["recall_at_k"] <= 1.0 and r["t_consulta_ms_mediana"] > 0
    sk = [r for r in rows if r["metodo"] == "SimHash-256"]
    assert all(r["bytes_por_paciente"] == 32 for r in sk)  # 256 bits

    p = res["expresion_pasada"]
    assert p["genes_leidos"] == 40 and p["pacientes"] == 52 and p["consultas"] == 12

    mrows = res["mutaciones"]
    exj = [r for r in mrows if r["metodo"] == "Jaccard exacto"]
    assert exj and all(r["recall_at_k"] == 1.0 for r in exj)
    mh = [r for r in mrows if r["metodo"] == "MinHash-128 lineal"]
    assert all(r["pureza_at_5"] > 0.9 for r in mh if r["n"] == 30)  # bloques de genes por tipo

    assert (tmp_path / "expresion.csv").exists() and (tmp_path / "mutaciones.csv").exists()
    js = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert js["parametros"]["k"] == 5 and np.isfinite(js["expresion"][0]["recall_at_k"])
