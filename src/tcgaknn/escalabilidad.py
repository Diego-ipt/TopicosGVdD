"""Experimento de escalabilidad: algoritmos aproximados construidos sobre el stream.

Mide cómo escalan con el número de pacientes n los algoritmos de
``tcgaknn.algoritmos`` cuando sus sketches se arman **en una pasada del stream**
(filas de rasgo que llegan una a una), sobre una parte de los datos.

Expresión (medida coseno; L1 para E2LSH-L1):

1. Se eligen N = n_max + |Q| pacientes (estratificados por tipo de cáncer) y se
   leen las primeras ``g_max`` filas de genes del stream.
2. Por fila: log2(x+1), z-score con μ y σ solo de la base (regla anti-fuga),
   NA → 0. Con esa fila z se actualizan, en la misma pasada:
   - ``StreamingProjectionAccumulator`` (Ω por ID): RP, bits SimHash y bits LSH;
   - dos ``RowAccumulator`` con las filas a de E2LSH (gaussiana y Cauchy).
3. La matriz Z (g × N) se guarda **solo** para la verdad de terreno y para la
   línea base EX: es lo que un sistema en streaming no podría guardar.

Mutaciones (medida Jaccard): stream de eventos MAF hasta juntar N pacientes;
``StreamingMinHash`` construye las firmas y los conjuntos exactos se guardan
solo para evaluar.

Para cada n de una grilla de bases anidadas se miden recall@k frente al exacto,
candidatos examinados, tiempo por consulta (consulta a consulta), construcción,
bytes del índice y pureza / voto del tipo de cáncer.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tcgaknn.algoritmos import (
    BandedLSH,
    E2LSH,
    ExactKNN,
    MinHashLSH,
    SimHashKNN,
    StreamingMinHash,
    StreamingProjectionAccumulator,
    l2_normalize,
    topk,
)
from tcgaknn.algoritmos.coseno import pack_bits
from tcgaknn.algoritmos.evaluacion import recall_at_k
from tcgaknn.algoritmos.lp import suggest_w
from tcgaknn.perfil import TUMOR_CODES, load_labels, log2p1, patient_barcode, sample_type_code, select_samples
from tcgaknn.stream import StreamService

SIZES = (250, 500, 1000, 2000, 4000)
NON_SILENT_EXCLUDED = frozenset({"Silent"})


class RowAccumulator:
    """Acumula Y = Z·A con A generada fila a fila por ID de rasgo (``row_fn``), en lotes."""

    def __init__(self, n_samples: int, cols: int, row_fn: Callable[[str], np.ndarray], batch_size: int = 256):
        self.Y = np.zeros((n_samples, cols), dtype=np.float64)
        self.row_fn = row_fn
        self._z = np.empty((batch_size, n_samples), dtype=np.float32)
        self._a = np.empty((batch_size, cols), dtype=np.float32)
        self._fill = 0

    def update(self, feature_id: str, z: np.ndarray) -> None:
        self._z[self._fill] = z
        self._a[self._fill] = self.row_fn(feature_id)
        self._fill += 1
        if self._fill == self._z.shape[0]:
            self._flush()

    def _flush(self) -> None:
        if self._fill:
            self.Y += self._z[: self._fill].T.astype(np.float64) @ self._a[: self._fill].astype(np.float64)
            self._fill = 0

    def result(self) -> np.ndarray:
        self._flush()
        return self.Y


def _peak_rss_mib() -> float | None:
    """Memoria pico del proceso (no se usa tracemalloc: hace el parseo del stream ~5 veces más lento)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**20
    except ImportError:
        return None


def _standardize(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray | None:
    """z-score con μ, σ de la base (misma regla que el acumulador); None si el rasgo se descarta."""
    ref = x[fit_mask]
    ref = ref[np.isfinite(ref)]
    if ref.size < 2:
        return None
    mu, sd = ref.mean(), ref.std()
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    z = (x - mu) / sd
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def _split(n_total: int, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Consultas fijas y orden aleatorio de la base (las bases B_n son prefijos: anidadas)."""
    perm = np.random.default_rng(seed + 1).permutation(n_total)
    return np.sort(perm[:n_queries]), perm[n_queries:]


def _purity_vote(idx: np.ndarray, base_y: np.ndarray, q_y: np.ndarray, k: int = 5) -> tuple[float, float]:
    pur, vote = [], []
    for i in range(idx.shape[0]):
        nb = idx[i, :k]
        nb = nb[nb >= 0]
        if nb.size == 0:
            pur.append(0.0)
            vote.append(0.0)
            continue
        labs = base_y[nb]
        pur.append(float(np.mean(labs == q_y[i])))
        cnt = Counter(labs.tolist())
        best = max(cnt.values())
        winners = {c for c, v in cnt.items() if v == best}
        pred = next(lab for lab in labs.tolist() if lab in winners)  # desempate: el vecino más cercano
        vote.append(float(pred == q_y[i]))
    return float(np.mean(pur)), float(np.mean(vote))


def _timed_queries(fn: Callable[[slice], tuple[np.ndarray, np.ndarray, dict]], nq: int, warmup: int = 5):
    """Consulta a consulta: devuelve (idx, n_candidates, latencias en ms)."""
    for i in range(min(warmup, nq)):
        fn(slice(i, i + 1))
    idx_rows, cand, lat = [], [], []
    for i in range(nq):
        t0 = time.perf_counter()
        idx, _, st = fn(slice(i, i + 1))
        lat.append((time.perf_counter() - t0) * 1e3)
        idx_rows.append(idx[0])
        nc = st.get("n_candidates")
        cand.append(int(np.asarray(nc).ravel()[0]) if nc is not None else -1)
    return np.stack(idx_rows), np.asarray(cand), np.asarray(lat)


def _row(method: str, medida: str, n: int, k: int, idx, truth, cand, lat, build_s, nbytes, base_y, q_y, **extra):
    pur, vote = _purity_vote(idx, base_y, q_y)
    r = {
        "metodo": method,
        "medida": medida,
        "n": n,
        "recall_at_k": recall_at_k(idx, truth, k),
        "frac_candidatos": float(np.mean(cand) / n) if np.all(cand >= 0) else 1.0,
        "t_consulta_ms_mediana": float(np.median(lat)),
        "t_consulta_ms_p95": float(np.percentile(lat, 95)),
        "t_construccion_s": float(build_s),
        "bytes_indice": int(nbytes),
        "bytes_por_paciente": float(nbytes) / n,
        "pureza_at_5": pur,
        "voto_at_5": vote,
    }
    r.update(extra)
    return r


# --------------------------------------------------------------------------
# Expresión
# --------------------------------------------------------------------------


def stream_expression(
    root: Path | str,
    *,
    g_max: int,
    n_max: int,
    n_queries: int,
    seed: int,
    cols: int = 1024,
    e2_r: int = 4,
    e2_L: int = 16,
    labels: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Una pasada del stream de expresión: sketches + Z (solo evaluación)."""
    labels = load_labels(root) if labels is None else labels
    e2 = {p: E2LSH(p=p, w=1.0, r=e2_r, L=e2_L, seed=seed, rerank="proj") for p in (1, 2)}
    t0 = time.perf_counter()
    acc = rowacc = None
    zrows: list[np.ndarray] = []
    fids: list[str] = []
    n_rows = n_skipped = 0
    for rec in StreamService(root, data_types=["expression"]):
        if rec.kind == "schema":
            barcodes = rec.fields["columns"]
            cols_idx, y, info = select_samples(barcodes, labels, n_max + n_queries, seed)
            if cols_idx.size < n_queries + 2:
                raise ValueError("muy pocos pacientes elegibles")
            q_pos, base_order = _split(cols_idx.size, n_queries, seed)
            fit_mask = np.zeros(cols_idx.size, dtype=bool)
            fit_mask[base_order] = True
            N = cols_idx.size
            acc = StreamingProjectionAccumulator(N, cols, seed, standardize=False)
            rowacc = {p: RowAccumulator(N, e2[p].n_funcs, e2[p].a_row) for p in (1, 2)}
            continue
        if rec.kind != "feature_row":
            continue
        if n_rows >= g_max:
            break
        n_rows += 1
        z = _standardize(log2p1(rec.values[cols_idx].astype(np.float64)), fit_mask)
        if z is None:
            n_skipped += 1
            continue
        acc.update(rec.key, z)
        for p in (1, 2):
            rowacc[p].update(rec.key, z)
        zrows.append(z)
        fids.append(rec.key)
        if n_rows % 1000 == 0:
            log(f"  expresión: {n_rows:,} genes leídos, {time.perf_counter() - t0:,.0f} s")
    pass_s = time.perf_counter() - t0
    Z = np.stack(zrows, axis=1)  # N × g (solo evaluación)
    sketch_bytes = acc.Y.nbytes + acc.sqnorm.nbytes + sum(r.Y.nbytes for r in rowacc.values())
    acc.raw()  # vacía el último lote: sqnorm solo se actualiza al vaciar
    norms = np.sqrt(acc.sqnorm)[:, None]
    safe = np.where(norms > 1e-12, norms, 1.0)
    return {
        "Z": l2_normalize(Z),
        "P": acc.projections(256),
        "bits": acc.sign_bits(cols),
        "Y": {p: rowacc[p].result() / safe for p in (1, 2)},  # proyecciones de los vectores L2-normalizados
        "e2": e2,
        "y": np.asarray(y),
        "q_pos": q_pos,
        "base_order": base_order,
        "pasada": {
            "genes_leidos": n_rows,
            "genes_descartados_var0": n_skipped,
            "genes_usados": len(fids),
            "pacientes": int(N),
            "consultas": int(n_queries),
            "segundos": pass_s,
            "genes_por_segundo": n_rows / pass_s if pass_s else 0.0,
            "memoria_pico_proceso_MiB": _peak_rss_mib(),
            "bytes_sketches": int(sketch_bytes),
            "bytes_Z_evaluacion": int(Z.nbytes),
            "seleccion": info,
        },
    }


def evaluate_expression(data: dict[str, Any], sizes: Sequence[int], k: int = 10, log: Callable[[str], None] = print):
    Z, P, bits, Y, y = data["Z"], data["P"], data["bits"], data["Y"], data["y"]
    q, order = data["q_pos"], data["base_order"]
    q_y = y[q]
    Pq = P[q]
    SHq = pack_bits(bits[q, :256])
    rows: list[dict] = []
    # w de E2LSH: calibrado una vez con la distancia al k-ésimo vecino exacto de las consultas
    calib_base = order[: min(1000, order.size)]
    w = {}
    for p, metric in ((2, "l2"), (1, "l1")):
        ex = ExactKNN(metric).fit(Z[calib_base])
        _, d = ex.query(Z[q[: min(50, q.size)]], k)[:2]
        w[p] = suggest_w(d[:, -1])
    for n in sizes:
        if n > order.size:
            log(f"  n = {n} > base disponible ({order.size}); se omite")
            continue
        b = order[:n]
        by = y[b]
        # Verdad de terreno (coseno y L1) con la matriz guardada solo para evaluar
        ex = ExactKNN("cosine").fit(Z[b])
        truth_cos = ex.query(Z[q], k)[0]
        truth_l1 = ExactKNN("l1").fit(Z[b]).query(Z[q], k)[0]

        idx, cand, lat = _timed_queries(lambda s: ex.query(Z[q][s], k), q.size)
        rows.append(_row("EX (coseno exacto)", "coseno", n, k, idx, truth_cos, cand, lat,
                         ex.build_time_s, ex.index_bytes, by, q_y))

        t0 = time.perf_counter()
        rp = ExactKNN("l2").fit(P[b])
        bt = time.perf_counter() - t0
        idx, cand, lat = _timed_queries(lambda s: rp.query(Pq[s], k), q.size)
        rows.append(_row("RP-256", "coseno", n, k, idx, truth_cos, cand, lat, bt, P[b].nbytes, by, q_y))

        t0 = time.perf_counter()
        sh = SimHashKNN.from_signatures(pack_bits(bits[b, :256]), 256)
        bt = time.perf_counter() - t0
        idx, cand, lat = _timed_queries(lambda s: sh.query_packed(SHq[s], k), q.size)
        rows.append(_row("SimHash-256", "coseno", n, k, idx, truth_cos, cand, lat, bt, sh.packed.nbytes, by, q_y))

        t0 = time.perf_counter()
        shr = SimHashKNN.from_signatures(pack_bits(bits[b, :256]), 256, base=P[b], rerank_factor=10)
        bt = time.perf_counter() - t0
        idx, cand, lat = _timed_queries(lambda s: shr.query_packed(SHq[s], k, Q_base=Pq[s]), q.size)
        rows.append(_row("SimHash-256 + re-ranking RP", "coseno", n, k, idx, truth_cos, cand, lat, bt,
                         shr.packed.nbytes + P[b].nbytes, by, q_y))

        for L, r in ((16, 12), (32, 8)):
            lsh = BandedLSH(L=L, r=r, rerank="m", m=256).fit_bits(bits[b, : L * r], rerank_data=P[b])
            qb = bits[q, : L * r]
            idx, cand, lat = _timed_queries(lambda s: lsh.query_bits(qb[s], k, rerank_data=Pq[s]), q.size)
            rows.append(_row(f"LSH bandas {L}×{r}", "coseno", n, k, idx, truth_cos, cand, lat,
                             lsh.build_time_s, lsh.index_bytes()["total_sin_omega"], by, q_y))

        for p, truth, medida in ((2, truth_cos, "coseno (L2 en unitarios)"), (1, truth_l1, "L1")):
            e = data["e2"][p]
            idx_e = E2LSH(p=p, w=w[p], r=e.r, L=e.L, seed=e.seed, rerank="proj").fit_projections(Y[p][b])
            Yq = Y[p][q]
            idx, cand, lat = _timed_queries(lambda s: idx_e.query_projections(Yq[s], k), q.size)
            rows.append(_row(f"E2LSH-L{p}", medida, n, k, idx, truth, cand, lat,
                             idx_e.build_time_s, idx_e.index_bytes(), by, q_y, w=w[p]))
        log(f"  expresión n = {n:,}: listo")
    return rows


# --------------------------------------------------------------------------
# Mutaciones
# --------------------------------------------------------------------------


def stream_mutations(
    root: Path | str,
    *,
    n_total: int,
    seed: int,
    num_perm: int = 128,
    labels: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Lee eventos MAF hasta juntar ``n_total`` pacientes elegibles (claves contiguas en el archivo)."""
    labels = load_labels(root) if labels is None else labels
    smh = StreamingMinHash(num_perm, seed)
    sets: dict[str, set[str]] = {}
    seen_patients: set[str] = set()
    current = None
    events = 0
    t0 = time.perf_counter()
    for rec in StreamService(root, data_types=["mutations"]):
        if rec.kind != "event":
            continue
        events += 1
        key = rec.key
        if key != current:
            if len(sets) >= n_total:
                break  # el archivo agrupa los eventos por muestra: la anterior quedó completa
            current = key
            p = patient_barcode(key)
            ok = labels.get(p) is not None and sample_type_code(key) in TUMOR_CODES and p not in seen_patients
            if ok:
                seen_patients.add(p)
                sets[key] = set()
                smh.add_key(key)
        if key not in sets:
            continue
        f = rec.fields
        if f.get("Variant_Classification") in NON_SILENT_EXCLUDED or f.get("FILTER") != "PASS":
            continue
        gene = (f.get("Hugo_Symbol") or "").strip()
        if gene and gene != "Unknown":
            sets[key].add(gene)
            smh.update(key, gene)
    pass_s = time.perf_counter() - t0
    keys, sig = smh.signatures()
    pos = {k: i for i, k in enumerate(keys)}
    order_keys = list(sets)
    S = sig[[pos[k] for k in order_keys]]
    y = np.asarray([labels[patient_barcode(k)] for k in order_keys])
    q_pos, base_order = _split(len(order_keys), min(200, len(order_keys) // 5), seed)
    log(f"  mutaciones: {events:,} eventos, {len(order_keys):,} pacientes, {pass_s:,.0f} s")
    return {
        "sets": [frozenset(sets[k]) for k in order_keys],
        "sig": S,
        "y": y,
        "q_pos": q_pos,
        "base_order": base_order,
        "pasada": {"eventos_leidos": events, "pacientes": len(order_keys), "segundos": pass_s,
                   "bytes_firmas": int(S.nbytes), "num_perm": num_perm},
    }


def _jaccard_matrix(qsets: Sequence[frozenset], bsets: Sequence[frozenset]) -> np.ndarray:
    genes = {g: i for i, g in enumerate(sorted(set().union(*qsets, *bsets)))}

    def onehot(sets):
        M = np.zeros((len(sets), len(genes)), dtype=np.float32)
        for i, s in enumerate(sets):
            M[i, [genes[g] for g in s]] = 1.0
        return M

    Bq, Bb = onehot(qsets), onehot(bsets)
    inter = Bq @ Bb.T
    union = Bq.sum(1)[:, None] + Bb.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def evaluate_mutations(data: dict[str, Any], sizes: Sequence[int], k: int = 10, bands: int = 32, rows_per_band: int = 4,
                       log: Callable[[str], None] = print):
    sets, sig, y = data["sets"], data["sig"], data["y"]
    q, order = data["q_pos"], data["base_order"]
    qsets = [sets[i] for i in q]
    q_y = y[q]
    out = []
    for n in sizes:
        if n > order.size:
            log(f"  mutaciones: n = {n} > base disponible ({order.size}); se omite")
            continue
        b = order[:n]
        bsets = [sets[i] for i in b]
        t0 = time.perf_counter()
        J = _jaccard_matrix(qsets, bsets)
        exact_s = time.perf_counter() - t0
        truth = topk(J, k)[0]
        idx, cand, lat = _timed_queries(lambda s: (*topk(J[s], k), {"n_candidates": np.array([n])}), q.size)
        out.append(_row("Jaccard exacto", "jaccard", n, k, idx, truth, cand, lat, exact_s,
                        sum(len(s) for s in bsets) * 8, y[b], q_y))

        Sb, Sq = sig[b], sig[q]

        def linear(s):
            est = (Sq[s][:, None, :] == Sb[None, :, :]).mean(axis=2)
            return (*topk(est, k), {"n_candidates": np.array([n])})

        idx, cand, lat = _timed_queries(linear, q.size)
        out.append(_row("MinHash-128 lineal", "jaccard", n, k, idx, truth, cand, lat, 0.0, Sb.nbytes, y[b], q_y))

        mlsh = MinHashLSH(bands=bands, rows=rows_per_band, seed=0, rerank="estimate").fit(Sb)
        idx, cand, lat = _timed_queries(lambda s: mlsh.query(Sq[s], k), q.size)
        out.append(_row(f"MinHashLSH {bands}×{rows_per_band}", "jaccard", n, k, idx, truth, cand, lat,
                        mlsh.build_time_s, mlsh.index_bytes(), y[b], q_y))
        log(f"  mutaciones n = {n:,}: listo")
    return out


def run(root: Path | str, out: Path | str, *, g_max: int = 5000, n_max: int = 4000, n_queries: int = 200,
        sizes: Sequence[int] = SIZES, seed: int = 2026, k: int = 10, do_mutations: bool = True,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Corre el experimento completo y escribe CSV y JSON en ``out``."""
    import csv

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(root)
    log(f"Etiquetas: {len(labels):,} pacientes")
    resumen: dict[str, Any] = {"parametros": {"g_max": g_max, "n_max": n_max, "n_queries": n_queries,
                                               "sizes": list(sizes), "seed": seed, "k": k}}

    def write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            wr.writeheader()
            wr.writerows(rows)

    log("Expresión: pasada del stream")
    data = stream_expression(root, g_max=g_max, n_max=n_max, n_queries=n_queries, seed=seed, labels=labels, log=log)
    resumen["expresion_pasada"] = data["pasada"]
    rows = evaluate_expression(data, sizes, k, log)
    write("expresion.csv", rows)
    resumen["expresion"] = rows
    if do_mutations:
        log("Mutaciones: pasada del stream")
        mdata = stream_mutations(root, n_total=n_max + n_queries, seed=seed, labels=labels, log=log)
        resumen["mutaciones_pasada"] = mdata["pasada"]
        mrows = evaluate_mutations(mdata, sizes, k, log=log)
        write("mutaciones.csv", mrows)
        resumen["mutaciones"] = mrows
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return resumen
