"""Servicio de stream sobre la carpeta de descarga de gdc-client.

La carpeta tiene una subcarpeta por UUID de GDC con un archivo de datos y una
carpeta ``logs/``. El servicio reconoce el tipo de dato de cada archivo por su
nombre y lo entrega como un stream de registros (``Record``), leyendo línea a
línea: nunca carga un archivo completo en memoria ni lo copia a disco.

Tipos de registro (``Record.kind``):

- ``schema``: primer registro de un archivo matricial; ``fields`` trae las
  columnas (IDs de muestra o de rasgo) que indexan ``values`` en los registros
  siguientes.
- ``feature_row``: un rasgo (gen, sonda, miRNA, vía) para todas las muestras.
- ``sample_row``: una muestra con todos sus rasgos (RPPA).
- ``event``: una mutación o un segmento de número de copias.
- ``table_row``: una fila de una tabla clínica o de anotaciones.
- ``end``: último registro de cada archivo, con el conteo emitido.

Uso::

    for rec in StreamService(raiz, limit_per_type=3):
        ...

    python -m tcgaknn.stream <carpeta> --limit 3
"""

from __future__ import annotations

import argparse
import gzip
import io
import queue
import re
import tarfile
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

NA_TOKENS = frozenset({"", "NA", "NaN", "nan", "null", "NULL", "None", "."})

DATA_KINDS = frozenset({"feature_row", "sample_row", "event", "table_row"})


@dataclass(frozen=True)
class Record:
    """Sobre común de todo lo que viaja por el stream."""

    data_type: str
    kind: str
    source: str
    key: str | None = None
    values: np.ndarray | None = None
    fields: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Apertura de archivos en modo streaming
# --------------------------------------------------------------------------


class _SequentialReader(io.RawIOBase):
    """Adaptador de solo lectura secuencial para miembros de un tar en modo stream ``r|gz``.

    ``TextIOWrapper`` consulta ``seekable()``, que el objeto de tarfile en modo
    stream no implementa.
    """

    def __init__(self, raw: Any):
        self._raw = raw

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer: Any) -> int:
        data = self._raw.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    """Abre texto plano, ``.gz`` o el primer ``.txt`` de un ``.tar.gz``, sin descomprimir a disco."""
    name = path.name
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(path, mode="r|gz") as tar:  # modo stream: lectura secuencial
            for member in tar:
                if member.isfile() and member.name.endswith(".txt"):
                    raw = io.BufferedReader(_SequentialReader(tar.extractfile(member)))
                    with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as fh:
                        yield fh
                    return
            raise ValueError(f"{path}: el .tar.gz no contiene un .txt")
    elif name.endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            yield fh


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    return token


def _split(line: str, sep: str) -> list[str]:
    return [_unquote(t) for t in line.rstrip("\r\n").split(sep)]


def parse_floats(tokens: list[str]) -> np.ndarray:
    """Convierte tokens a float32; los valores faltantes pasan a NaN."""
    clean = ["nan" if t in NA_TOKENS else t for t in tokens]
    return np.asarray(clean, dtype=np.str_).astype(np.float32)


# --------------------------------------------------------------------------
# Lectores por formato
# --------------------------------------------------------------------------


def read_matrix(
    path: Path,
    data_type: str,
    *,
    orientation: str,
    sep: str = "\t",
    meta_cols: int = 0,
) -> Iterator[Record]:
    """Matriz delimitada: primera columna = ID de fila, luego ``meta_cols`` de texto y los valores.

    ``orientation="feature"``: cada fila es un rasgo (columnas = muestras).
    ``orientation="sample"``: cada fila es una muestra (columnas = rasgos).
    """
    kind = "feature_row" if orientation == "feature" else "sample_row"
    with _open_text(path) as fh:
        header = _split(fh.readline(), sep)
        meta_names = header[1 : 1 + meta_cols]
        columns = header[1 + meta_cols :]
        yield Record(
            data_type,
            "schema",
            path.name,
            fields={
                "orientation": orientation,
                "id_column": header[0],
                "meta_columns": meta_names,
                "columns": columns,
            },
        )
        for lineno, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            parts = _split(line, sep)
            if len(parts) != len(header):
                raise ValueError(
                    f"{path.name}:{lineno}: {len(parts)} columnas, se esperaban {len(header)}"
                )
            yield Record(
                data_type,
                kind,
                path.name,
                key=parts[0],
                values=parse_floats(parts[1 + meta_cols :]),
                fields=dict(zip(meta_names, parts[1 : 1 + meta_cols])),
            )


def read_delimited(
    path: Path,
    data_type: str,
    *,
    kind: str,
    key_column: str,
    sep: str = "\t",
    comment: str | None = None,
) -> Iterator[Record]:
    """Tabla delimitada fila a fila (eventos o tablas); los valores quedan como texto."""
    with _open_text(path) as fh:
        header_line = fh.readline()
        while comment and header_line.startswith(comment):
            header_line = fh.readline()
        header = _split(header_line, sep)
        if key_column not in header:
            raise ValueError(f"{path.name}: no existe la columna clave {key_column!r}")
        key_idx = header.index(key_column)
        for line in fh:
            if not line.strip() or (comment and line.startswith(comment)):
                continue
            parts = _split(line, sep)
            parts += [""] * (len(header) - len(parts))
            yield Record(
                data_type,
                kind,
                path.name,
                key=parts[key_idx],
                fields=dict(zip(header, parts)),
            )


def read_xlsx(path: Path, data_type: str, *, sheet: str, key_column: str) -> Iterator[Record]:
    """Hoja de Excel en modo ``read_only`` (openpyxl la recorre sin cargarla entera)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb[sheet].iter_rows(values_only=True)
        header = [str(h) if h is not None else f"col{i}" for i, h in enumerate(next(rows))]
        key_idx = header.index(key_column)
        for row in rows:
            if row is None or all(v is None for v in row):
                continue
            yield Record(
                data_type,
                "table_row",
                path.name,
                key=None if row[key_idx] is None else str(row[key_idx]),
                fields=dict(zip(header, row)),
            )
    finally:
        wb.close()


# --------------------------------------------------------------------------
# Registro de fuentes conocidas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    data_type: str
    pattern: str
    reader: Callable[..., Iterator[Record]]
    options: dict[str, Any] = field(default_factory=dict)

    def matches(self, filename: str) -> bool:
        return re.search(self.pattern, filename) is not None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("expression", r"^EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2\.geneExp\.tsv$",
               read_matrix, {"orientation": "feature"}),
    SourceSpec("methylation_450k", r"^jhu-usc\.edu_PANCAN_HumanMethylation450\.betaValue_whitelisted\.tsv$",
               read_matrix, {"orientation": "feature"}),
    SourceSpec("methylation_27k_450k", r"^jhu-usc\.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450\.betaValue_whitelisted\.tsv$",
               read_matrix, {"orientation": "feature"}),
    SourceSpec("mirna", r"^pancanMiRs_.*\.csv$",
               read_matrix, {"orientation": "feature", "sep": ",", "meta_cols": 1}),
    SourceSpec("paradigm", r"^merge_merged_reals\.tar\.gz$",
               read_matrix, {"orientation": "feature"}),
    SourceSpec("rppa", r"^TCGA-RPPA-pancan-clean\.txt$",
               read_matrix, {"orientation": "sample", "meta_cols": 1}),
    SourceSpec("mutations", r"^mc3\..*\.maf\.gz$",
               read_delimited, {"kind": "event", "key_column": "Tumor_Sample_Barcode", "comment": "#"}),
    SourceSpec("copy_number", r"^broad\.mit\.edu_PANCAN_Genome_Wide_SNP_6_whitelisted\.seg$",
               read_delimited, {"kind": "event", "key_column": "Sample"}),
    SourceSpec("absolute_segments", r"^TCGA_mastercalls\.abs_segtabs\.fixed\.txt$",
               read_delimited, {"kind": "event", "key_column": "Sample"}),
    SourceSpec("absolute_purity", r"^TCGA_mastercalls\.abs_tables_JSedit\.fixed\.txt$",
               read_delimited, {"kind": "table_row", "key_column": "sample"}),
    SourceSpec("clinical_cdr", r"^TCGA-CDR-SupplementalTableS1\.xlsx$",
               read_xlsx, {"sheet": "TCGA-CDR", "key_column": "bcr_patient_barcode"}),
    SourceSpec("clinical_followup", r"^clinical_PANCAN_patient_with_followup\.tsv$",
               read_delimited, {"kind": "table_row", "key_column": "bcr_patient_barcode"}),
    SourceSpec("sample_quality", r"^merged_sample_quality_annotations\.tsv$",
               read_delimited, {"kind": "table_row", "key_column": "aliquot_barcode"}),
    SourceSpec("mirna_samples", r"^PanCanAtlas_miRNA_sample_information_list\.txt$",
               read_delimited, {"kind": "table_row", "key_column": "id"}),
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    spec: SourceSpec

    @property
    def data_type(self) -> str:
        return self.spec.data_type

    def records(self) -> Iterator[Record]:
        return self.spec.reader(self.path, self.spec.data_type, **self.spec.options)


def discover(root: Path | str, specs: Iterable[SourceSpec] = SOURCES) -> tuple[list[SourceFile], list[Path]]:
    """Recorre ``root`` (todas las subcarpetas, salvo ``logs/``) y clasifica cada archivo.

    Devuelve (fuentes reconocidas, archivos no reconocidos).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {root}")
    specs = tuple(specs)
    found: list[SourceFile] = []
    unknown: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or "logs" in rel.parts[:-1]:
            continue
        spec = next((s for s in specs if s.matches(path.name)), None)
        if spec is None:
            unknown.append(path)
        else:
            found.append(SourceFile(path, spec))
    return found, unknown


# --------------------------------------------------------------------------
# Servicio: productor en un hilo + cola acotada
# --------------------------------------------------------------------------


class _Failure:
    def __init__(self, exc: BaseException):
        self.exc = exc


_DONE = object()


class StreamService:
    """Stream interno de registros a partir de una carpeta.

    Un hilo productor recorre las fuentes en orden y deja cada registro en una
    cola acotada (``queue_size``): si el consumidor es lento, el productor se
    bloquea (*backpressure*) y la memoria queda acotada. ``limit_per_type``
    corta cada tipo de dato tras ese número de registros de datos (sin contar
    ``schema`` ni ``end``).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        data_types: Iterable[str] | None = None,
        limit_per_type: int | None = None,
        queue_size: int = 256,
    ):
        self.root = Path(root)
        self.data_types = None if data_types is None else frozenset(data_types)
        self.limit_per_type = limit_per_type
        self.queue_size = queue_size
        sources, self.unknown = discover(self.root)
        if self.data_types is not None:
            missing = self.data_types - {s.data_type for s in sources}
            if missing:
                raise ValueError(f"Tipos sin archivo en {self.root}: {sorted(missing)}")
            sources = [s for s in sources if s.data_type in self.data_types]
        self.sources = sources

    def __iter__(self) -> Iterator[Record]:
        return self.stream()

    def stream(self) -> Iterator[Record]:
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        stop = threading.Event()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                emitted: dict[str, int] = {}
                for src in self.sources:
                    count = emitted.get(src.data_type, 0)
                    truncated = False
                    reader = src.records()
                    try:
                        for rec in reader:
                            if rec.kind in DATA_KINDS:
                                if self.limit_per_type is not None and count >= self.limit_per_type:
                                    truncated = True
                                    break
                                count += 1
                            if not put(rec):
                                return
                    finally:
                        reader.close()  # libera el archivo aunque se corte antes
                    emitted[src.data_type] = count
                    end = Record(src.data_type, "end", src.path.name,
                                 fields={"records": count, "truncated": truncated})
                    if not put(end):
                        return
            except BaseException as exc:  # se re-lanza en el consumidor
                put(_Failure(exc))
            finally:
                put(_DONE)

        worker = threading.Thread(target=produce, name="tcga-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                if isinstance(item, _Failure):
                    raise item.exc
                yield item
        finally:
            stop.set()
            worker.join(timeout=5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _describe(rec: Record) -> str:
    if rec.kind == "schema":
        cols = rec.fields["columns"]
        return f"schema  {rec.fields['orientation']}  {len(cols)} columnas (p. ej. {cols[:2]})"
    if rec.kind == "end":
        return f"end     {rec.fields['records']} registros (truncado={rec.fields['truncated']})"
    if rec.values is not None:
        v = rec.values
        n_na = int(np.isnan(v).sum())
        return f"{rec.kind:<11} {rec.key}  n={v.size}  NA={n_na}  primeros={np.round(v[:3], 4).tolist()}"
    preview = {k: rec.fields[k] for k in list(rec.fields)[:4]}
    return f"{rec.kind:<11} {rec.key}  {preview}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stream de registros desde la carpeta de datos de GDC.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=3, help="registros de datos por tipo (0 = sin límite)")
    parser.add_argument("--types", nargs="*", help="tipos de dato a incluir (por defecto, todos)")
    args = parser.parse_args(argv)

    service = StreamService(args.root, data_types=args.types, limit_per_type=args.limit or None)
    for src in service.sources:
        print(f"[fuente] {src.data_type:<22} {src.path.relative_to(args.root)}")
    for path in service.unknown:
        print(f"[ignorado] {path.relative_to(args.root)}")
    for rec in service:
        print(f"{rec.data_type:<22} {_describe(rec)}")


if __name__ == "__main__":
    main()
