"""Crea archivos A1/A2/A3 sintéticos para probar el pipeline sin descargar TCGA."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from openpyxl import Workbook


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=Path("demo/raw"))
    a.add_argument("--genes", type=int, default=90)
    a.add_argument("--pacientes", type=int, default=30)
    args = a.parse_args()
    if args.pacientes < 9 or args.pacientes % 3:
        raise ValueError("El número de pacientes debe ser múltiplo de 3 y al menos 9")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    types = ["BRCA", "KIRC", "LUAD"]
    pids = [f"TCGA-AB-{i:04d}" for i in range(1, args.pacientes + 1)]
    aids = [p + "-01A-01R-ABCD-01" for p in pids]
    labels = [types[i * 3 // args.pacientes] for i in range(args.pacientes)]
    wb = Workbook()
    ws = wb.active
    ws.title = "TCGA-CDR"
    ws.append(["bcr_patient_barcode", "type"])
    for pid, label in zip(pids, labels):
        ws.append([pid, label])
    wb.save(args.out / "TCGA-CDR-SupplementalTableS1.xlsx")
    with (args.out / "merged_sample_quality_annotations.tsv").open("w") as f:
        f.write("patient_barcode\taliquot_barcode\tplatform\tAWG_excluded_because_of_pathology\tDo_not_use\n")
        for pid, aid in zip(pids, aids):
            f.write(f"{pid}\t{aid}\tIlluminaHiSeq_RNASeqV2\t0\t0\n")
    with (args.out / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv").open("w") as f:
        f.write("gene_id\t" + "\t".join(aids) + "\n")
        for gene in range(args.genes):
            v = rng.normal(0.0, 0.3, size=args.pacientes)
            v[np.asarray(labels) == types[gene % 3]] += 4.5
            if gene == 0:
                v[0] = np.nan
            f.write(f"GENE{gene}\t" + "\t".join(f"{x:.6f}" for x in v) + "\n")
    print(f"Datos simulados creados en {args.out} (NO son TCGA real)")


if __name__ == "__main__":
    main()
