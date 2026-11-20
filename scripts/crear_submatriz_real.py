"""Construye una submatriz REAL de A1, con A2/A3 originales para el pipeline.

Uso:
    python scripts/crear_submatriz_real.py data/raw \
        --out data/submatriz/raw --classes BRCA LUAD KIRC \
        --per-class 20 --genes 500 --gene-mode first --seed 2026

La selección de pacientes se hace DESPUÉS del cruce A1/A2/A3 y los filtros
preliminares de calidad. El pipeline vuelve a aplicar esos filtros. No se
permite repetir pacientes. Los genes se seleccionan sin mirar las etiquetas,
solo para depuración, NO como selección de rasgos definitiva.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from tcgaknn.cohorte_validada import build_cohort
from tcgaknn.stream import discover, _unquote


def single_source(root: Path, kind: str) -> Path:
    sources, _ = discover(root)
    matches = [s.path for s in sources if s.data_type == kind]
    if len(matches) != 1:
        raise ValueError(f"En {root} se esperaba exactamente 1 archivo {kind}; encontrados: {len(matches)}")
    return matches[0]


def crear_submatriz(
    root: Path,
    out: Path,
    *,
    classes: list[str] | None = None,
    per_class: int = 20,
    genes: int = 500,
    gene_mode: str = "first",
    seed: int = 2026,
    strict_annotations: bool = False,
) -> dict:
    root, out = Path(root).resolve(), Path(out).resolve()
    if per_class < 3:
        raise ValueError("--per-class debe ser >= 3 (mínimo train, val y test por clase)")
    if genes < 1:
        raise ValueError("--genes debe ser >= 1")
    if gene_mode not in {"first", "random"}:
        raise ValueError("--gene-mode debe ser 'first' o 'random'")
    if root == out or root in out.parents:
        # Subdirectorio dentro de raw puede hacer aparecer duplicados al repetir.
        raise ValueError("La salida debe estar fuera de la carpeta data/raw")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Carpeta de salida no vacía: {out}. Usa una nueva para no mezclar archivos")
    a1 = single_source(root, "expression")
    a2 = single_source(root, "clinical_cdr")
    a3 = single_source(root, "sample_quality")

    with a1.open("r", encoding="utf-8", newline="") as f:
        head_line = f.readline().rstrip("\r\n")
    header = [_unquote(x) for x in head_line.split("\t")]
    if len(header) < 2:
        raise ValueError("A1: cabecera inválida")
    barcodes = header[1:]
    cohort = build_cohort(barcodes, root, check_annotations=strict_annotations)
    rng = np.random.default_rng(seed)

    available = sorted(set(cohort.labels.tolist()))
    requested = [s.upper() for s in classes] if classes else available
    if len(set(requested)) != len(requested):
        raise ValueError("Hay clases duplicadas en --classes")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Tipos no presentes tras los filtros A3: {unknown}. Disponibles: {available}")

    chosen_cohort_rows: list[int] = []
    counts_selected = {}
    for typ in requested:
        positions = np.flatnonzero(cohort.labels == typ)
        if len(positions) < 3:
            raise ValueError(f"La clase {typ} tiene {len(positions)} pacientes elegibles: mínimo 3")
        k = min(per_class, len(positions))
        picked = rng.choice(positions, size=k, replace=False).tolist()
        chosen_cohort_rows.extend(int(i) for i in picked)
        counts_selected[typ] = k
    selected_positions = sorted(int(cohort.columns[i]) for i in chosen_cohort_rows)
    assert len(set(selected_positions)) == len(selected_positions)
    assert len(selected_positions) == len(chosen_cohort_rows)

    if gene_mode == "random":
        with a1.open("r", encoding="utf-8", newline="") as f:
            f.readline()
            n_total_genes = sum(bool(line.strip()) for line in f)
        n_wanted = min(genes, n_total_genes)
        chosen_gene_rows = set(int(x) for x in rng.choice(n_total_genes, size=n_wanted, replace=False))
    else:
        chosen_gene_rows = None

    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    # Se conservan A2 y A3 intactos; el preparador original vuelve a cruzar
    # sus metadatos con las columnas seleccionadas de A1.
    shutil.copy2(a2, out / a2.name)
    shutil.copy2(a3, out / a3.name)
    outfile = out / a1.name
    written = 0
    observed = 0
    feature_ids = []
    with a1.open("r", encoding="utf-8", newline="") as fin, outfile.open("w", encoding="utf-8", newline="") as fout:
        input_header = [_unquote(x) for x in fin.readline().rstrip("\r\n").split("\t")]
        if input_header != header:
            raise ValueError("A1 cambió durante la selección")
        fout.write("\t".join([input_header[0]] + [input_header[i+1] for i in selected_positions]) + "\n")
        for line in fin:
            if not line.strip():
                continue
            use = (observed < genes) if chosen_gene_rows is None else (observed in chosen_gene_rows)
            observed += 1
            if not use:
                if chosen_gene_rows is None:
                    break
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != len(header):
                raise ValueError(f"A1: gen en línea {observed+1} tiene {len(parts)} columnas, se esperaban {len(header)}")
            fout.write("\t".join([parts[0]] + [parts[i+1] for i in selected_positions]) + "\n")
            feature_ids.append(parts[0])
            written += 1
            if chosen_gene_rows is not None and written == len(chosen_gene_rows):
                break
    if written == 0:
        raise ValueError("La selección produjo cero genes")
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("El subconjunto tiene nombres de genes repetidos")
    meta = {
        "tipo": "submatriz_real_depuracion",
        "seed": seed,
        "gene_mode": gene_mode,
        "genes_solicitados": genes,
        "genes_seleccionados": written,
        "pacientes_seleccionados": len(selected_positions),
        "clases": counts_selected,
        "a1_origen": str(a1),
        "a2_origen": str(a2),
        "a3_origen": str(a3),
        "nota": "Subconjunto real, pero NO representativo para conclusiones científicas. A2 y A3 intactos.",
    }
    (out / "submatriz_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Submatriz REAL A1 + A2/A3 para depurar el proyecto")
    parser.add_argument("root", type=Path, help="Carpeta con A1, A2, A3 originales")
    parser.add_argument("--out", type=Path, default=Path("data/submatriz/raw"))
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Tipos de cáncer, por ejemplo BRCA LUAD KIRC. Por defecto, todos")
    parser.add_argument("--per-class", type=int, default=20,
                        help="Máximo de pacientes por tipo de cáncer")
    parser.add_argument("--genes", type=int, default=500)
    parser.add_argument("--gene-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict-annotations", action="store_true")
    a = parser.parse_args()
    print(json.dumps(crear_submatriz(a.root, a.out, classes=a.classes,
          per_class=a.per_class, genes=a.genes, gene_mode=a.gene_mode,
          seed=a.seed, strict_annotations=a.strict_annotations),
          indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
