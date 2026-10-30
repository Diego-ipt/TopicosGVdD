"""Prueba de ``scripts/mapa_2d.py`` sobre la carpeta sintética de ``test_perfil``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from test_perfil import root  # noqa: F401  (fixture con la carpeta sintética)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mapa_2d  # noqa: E402


def test_leave_one_out_excludes_self() -> None:
    rng = np.random.default_rng(0)
    # Dos grupos bien separados y un punto "huérfano" de una tercera clase: sin él mismo
    # en la búsqueda, su predicción no puede ser su propia clase.
    P = np.vstack([rng.normal(0, 0.05, (10, 8)) + 3, rng.normal(0, 0.05, (10, 8)) - 3, [[3.0] * 8]]).astype(np.float32)
    y = np.array(["A"] * 10 + ["B"] * 10 + ["C"])
    bits = P @ rng.normal(size=(8, 64)) >= 0
    pred = mapa_2d.leave_one_out_predictions(bits, P, y, b=64, f=5, k=5)
    assert pred[-1] == "A"
    assert (pred[:10] == "A").all() and (pred[10:20] == "B").all()


def test_build_and_figures(root: Path, tmp_path: Path) -> None:  # noqa: F811
    npz = mapa_2d.build(root, tmp_path, seed=2026, b=64, m=64, f=5, k=5, n_queries=12)
    mapa_2d.figures(npz, tmp_path, tmp_path / "fig")
    for name in ("mapa_tipos.png", "mapa_errores.png", "mapa_interactivo.html"):
        assert (tmp_path / "fig" / name).stat().st_size > 1000
    csv = (tmp_path / "mapa.csv").read_text(encoding="utf-8").splitlines()
    assert csv[0].startswith("barcode,tipo") and len(csv) == 61  # 60 pacientes elegibles
    page = (tmp_path / "fig" / "mapa_interactivo.html").read_text(encoding="utf-8")
    assert "__DATA__" not in page and "TCGA-" in page
    z = np.load(npz, allow_pickle=True)
    assert z["es_consulta"].sum() == 12 and (z["pred_exacto"][z["es_consulta"]] != "").all()
