"""Benchmark opcional y pareado: una consulta densa vs ANN + proyección.

La preparación/entrenamiento del proyecto CONTINÚA siendo streaming. Este
benchmark crea una matriz de genes temporal en disco, con una sola pasada A1,
únicamente para comparar consultas online en las mismas condiciones.

Etapas excluidas de TODOS los tiempos: leer A1, estandarizar genes, cargar la
base e inicializar índices, generar los pesos aleatorios. Etapas incluidas:
 - exacto: normalizar y calcular distancias contra toda la base, top-k;
 - ANN: proyectar q desde el vector completo de genes, hashear, consultar
   índice y reordenar. Las bases/índices se mantienen preparados en RAM.
No representa latencia end-to-end desde archivo crudo o red.
"""
from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .algoritmos.base import omega_matrix
from .algoritmos.exacto.exact import ExactKNN
from .algoritmos.lp.e2lsh import E2LSH
from .cohorte_validada import build_cohort
from .experimento_streaming import _build, _phase, _truth, QUALITY_POLICY
from .preparacion_streaming import _single_source, _transform, peak_rss_mib


def _dense_from_single_pass(data_dir: Path, raw_dir: Path, phase: str,
                            indices: np.ndarray, path: Path) -> np.memmap:
    """Reproduce Z exactamente usando stats de train; escritura genes como columnas."""
    with np.load(data_dir / 'splits.npz', allow_pickle=False) as sp:
        sample_ids = sp['sample_ids'].astype(str).tolist()
        patient_ids = sp['patient_ids'].astype(str).tolist()
        genes = sp['feature_ids'].astype(str).tolist()
        means = sp['gene_mean_train']
        sds = sp['gene_std_train']
    meta = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    if not meta['zscore']:
        raise ValueError('Benchmark pareado requiere zscore=True para reproducir las muestras')
    a1 = _single_source(raw_dir, 'expression')
    it = a1.records()
    schema = next(it)
    it.close()
    cohort = build_cohort(schema.fields['columns'], raw_dir,
                          check_annotations=meta['check_annotations'])
    if cohort.sample_ids != sample_ids or cohort.patient_ids != patient_ids:
        raise ValueError('A1/A2/A3 no coinciden con la cohorte preparada')
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                    shape=(len(indices), len(genes)))
    mapping = {gene: j for j, gene in enumerate(genes)}
    if len(mapping) != len(genes):
        raise ValueError('Genes duplicados en splits.npz')
    seen = set()
    for rec in a1.records():
        if rec.kind != 'feature_row':
            continue
        gid = str(rec.key)
        if gid not in mapping:
            continue
        if gid in seen:
            raise ValueError(f'Gen repetido al crear Z: {gid}')
        seen.add(gid)
        j = mapping[gid]
        values = _transform(rec.values[cohort.columns[indices]], meta['transform'])
        z = (values - means[j]) / sds[j]
        out[:, j] = np.where(np.isfinite(z), z, 0).astype(np.float32)
    if len(seen) != len(genes):
        raise ValueError(f'Faltan {len(genes) - len(seen)} genes para benchmark pareado')
    out.flush()
    del out
    return np.load(path, mmap_mode='r')


def _weights(method: str, data: dict, feature_ids: list[str]) -> np.ndarray:
    meta = data['meta']
    if method == 'lsh':
        return omega_matrix(feature_ids, int(meta['cols']), int(meta['seed'])).astype(np.float64)
    p = 1 if method == 'e2lsh_p1' else 2
    return E2LSH(p=p, r=1, L=int(meta['lp_cols']), seed=int(meta['seed']),
                 rerank='proj').a_matrix(feature_ids).astype(np.float64)


def _online_query(z: np.ndarray, weights: np.ndarray, config: dict, idx, k: int,
                  replicas: int):
    """Proyecta SOLO la consulta y luego ejecuta la consulta indexada."""
    method = config['method']
    start = time.perf_counter_ns()
    projected = np.asarray(z, dtype=np.float64) @ weights
    replica = config.get('replica', 0)
    offset = replica * (weights.shape[1] // replicas)
    p = projected[offset:]
    if method == 'lsh':
        m = config['m']
        norms = float(np.linalg.norm(z.astype(np.float64)))
        rerank = (p[:m] / (np.sqrt(m) * norms)).astype(np.float32)[None, :]
        bits = (p[:config['L'] * config['r']] >= 0)[None, :]
    else:
        count = config['L'] * config['r']
        rerank = p[:config['rerank_cols']]
        hashed = p[:count]
        if method == 'e2lsh_p2':
            norms = float(np.linalg.norm(z.astype(np.float64)))
            hashed = hashed / norms
            rerank = rerank / norms
    mid = time.perf_counter_ns()
    if method == 'lsh':
        found, _, _ = idx.query_bits(bits, k, rerank_data=rerank)
    else:
        found, _, _ = idx.query_projections(hashed[None, :], k,
                                            rerank_query_projections=rerank[None, :])
    end = time.perf_counter_ns()
    return found[0], (mid - start) / 1e6, (end - mid) / 1e6


def _med(v: list[float]) -> float:
    return float(np.median(np.asarray(v)))


def _p95(v: list[float]) -> float:
    return float(np.percentile(np.asarray(v), 95))


def benchmark_pareado(data_dir: Path | str, raw_dir: Path | str,
                       selection_path: Path | str, out_csv: Path | str, *,
                       phase: str = 'test', queries: int = 0, seed: int = 777,
                       max_dense_mib: float = 1024.0, log=print) -> list[dict]:
    """Mismo subconjunto de consultas, misma base precargada y medición 1×1.

    `queries=0` usa todas las consultas. Para sistemas pequeños usar, p.ej.,
    32 y repetir con varias semillas. Los tiempos no incluyen E/S ni fit.
    """
    if phase not in {'val', 'test'}:
        raise ValueError('phase debe ser val o test')
    if queries < 0 or max_dense_mib <= 0:
        raise ValueError('queries debe ser >=0 y max_dense_mib >0')
    data_dir, raw_dir = Path(data_dir), Path(raw_dir)
    frozen = json.loads(Path(selection_path).read_text(encoding='utf-8'))
    data = _phase(data_dir, phase, verify_hash=False)
    if frozen.get('dataset_id') != data['meta']['dataset_id']:
        raise ValueError('Selección congelada de otro dataset_id')
    if frozen.get('quality_thresholds') != QUALITY_POLICY:
        raise ValueError('Benchmark requiere política preregistrada vigente')
    sizes = [int(x['n']) for x in frozen['selected_by_size']]
    k = int(frozen['k'])
    replicas = int(frozen.get('replicas', 1))
    qall = np.asarray(data['query'], dtype=np.int64)
    rng = np.random.default_rng(seed)
    select = (np.sort(rng.choice(len(qall), queries, replace=False))
              if queries and queries < len(qall) else np.arange(len(qall)))
    q_ids = qall[select]
    max_n = max(sizes)
    with np.load(data_dir / 'splits.npz', allow_pickle=False) as sp:
        feature_ids = sp['feature_ids'].astype(str).tolist()
    d = len(feature_ids)
    # RAM estimada: Z de benchmark temporal + dos copias potenciales de base
    # (carga + normalización exacta) + pesos aleatorios (worst-case).
    dense_mib = (3 * max_n + len(q_ids)) * d * 4 / 2**20
    projected_cols = max(int(data['meta']['cols']), int(data['meta']['lp_cols']))
    projection_mib = d * projected_cols * 8 / 2**20
    estimated_mib = dense_mib + projection_mib
    if estimated_mib > max_dense_mib:
        raise MemoryError(f'Benchmark pareado requiere ~{estimated_mib:.1f} MiB en RAM; '
                          f'límite {max_dense_mib:.1f} MiB. Bajar --sizes en la '
                          'validación o elevar conscientemente el límite.')
    indices = np.concatenate((data['order'][:max_n], q_ids))
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix='tcgaknn-paired-') as temp:
        preprocessing_start = time.perf_counter()
        z = _dense_from_single_pass(data_dir, raw_dir, phase, indices,
                                     Path(temp) / 'Z_benchmark.npy')
        reconstruction_seconds = time.perf_counter() - preprocessing_start
        q = np.ascontiguousarray(z[max_n:])
        # El exacto usa el MISMO conjunto de consultas para ambas métricas.
        for size in frozen['selected_by_size']:
            n = int(size['n'])
            base = np.ascontiguousarray(z[:n])
            cached_exact = {}
            for metric in ('cosine', 'l1'):
                exact = ExactKNN(metric=metric).fit(base)
                # Precalentamiento de BLAS/numba; fuera del tiempo medido.
                exact.query(q[:1], k)
                latencies = []
                ids = []
                for row in q:
                    start = time.perf_counter_ns()
                    neigh, _, _ = exact.query(row[None, :], k)
                    latencies.append((time.perf_counter_ns() - start) / 1e6)
                    ids.append(neigh[0])
                cached_truth = _truth(data, n, k, metric)[0][select]
                # El benchmark denso debe ser coherente con la verdad streaming.
                # Reportamos acuerdo exacto y NO alteramos las predicciones.
                oracle_agreement = float(np.mean([
                    np.array_equal(a, b) for a, b in zip(ids, cached_truth)
                ]))
                cached_exact[metric] = (_med(latencies), ids, oracle_agreement)
                results.append({
                    'phase': phase, 'n': n, 'method': f'exact_dense_{metric}',
                    'metric': metric, 'replica': 0, 'k': k,
                    'n_queries': len(q), 'exact_single_query_median_ms': _med(latencies),
                    'exact_single_query_p95_ms': _p95(latencies),
                    'ann_projection_median_ms': None, 'ann_search_median_ms': None,
                    'ann_projection_plus_search_median_ms': None, 'speedup_vs_dense_exact': None,
                    'same_topk_as_dense_exact_rate': 1.0,
                    'dense_exact_agreement_with_streaming_oracle': oracle_agreement,
                    'dense_base_bytes': int(base.nbytes), 'projector_bytes': 0,
                    'exact_fit_seconds_excluded': float(exact.build_time_s),
                    'ann_build_seconds_excluded': None,
                    'dense_reconstruction_seconds_excluded': reconstruction_seconds,
                    'resident_index_bytes_estimate': int(exact.index_bytes),
                    'benchmark_stage': 'single_query_preprocessed_dense_in_ram',
                    'process_peak_rss_mib_cumulative': peak_rss_mib(),
                })
                del exact
            for selected in size['selected']:
                config = selected['config']
                method = config['method']
                if method.startswith('exact'):
                    continue
                metric = 'l1' if method == 'e2lsh_p1' else 'cosine'
                _, resident, ann_build, idx = _build(config, data, n, replicas, with_index=True)
                weights = _weights(method, data, feature_ids)
                # Calienta el índice y el producto con pesos precargados.
                _online_query(q[0], weights, config, idx, k, replicas)
                proj_lat, search_lat, total_lat, agree = [], [], [], []
                for qi, row in enumerate(q):
                    found, p_ms, s_ms = _online_query(row, weights, config, idx, k, replicas)
                    proj_lat.append(p_ms)
                    search_lat.append(s_ms)
                    total_lat.append(p_ms + s_ms)
                    agree.append(np.array_equal(found, cached_exact[metric][1][qi]))
                reference = cached_exact[metric][0]
                median_online = _med(total_lat)
                results.append({
                    'phase': phase, 'n': n, 'method': method, 'metric': metric,
                    'replica': config.get('replica', 0), 'k': k,
                    'n_queries': len(q), 'exact_single_query_median_ms': reference,
                    'exact_single_query_p95_ms': None,
                    'ann_projection_median_ms': _med(proj_lat),
                    'ann_search_median_ms': _med(search_lat),
                    'ann_projection_plus_search_median_ms': median_online,
                    'speedup_vs_dense_exact': reference / median_online if median_online else None,
                    'same_topk_as_dense_exact_rate': float(np.mean(agree)),
                    'dense_exact_agreement_with_streaming_oracle': cached_exact[metric][2],
                    'dense_base_bytes': int(base.nbytes),
                    'projector_bytes': int(weights.nbytes),
                    'exact_fit_seconds_excluded': None,
                    'ann_build_seconds_excluded': float(ann_build),
                    'dense_reconstruction_seconds_excluded': reconstruction_seconds,
                    'resident_index_bytes_estimate': int(resident),
                    'benchmark_stage': 'single_query_preprocessed_ann_projection_plus_search',
                    'process_peak_rss_mib_cumulative': peak_rss_mib(),
                })
                del idx, weights
            log(f'[benchmark pareado] {phase}: n={n}, consultas={len(q)}')
            del base
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['phase', 'n', 'method', 'metric', 'replica', 'k', 'n_queries',
               'exact_single_query_median_ms', 'exact_single_query_p95_ms',
               'ann_projection_median_ms', 'ann_search_median_ms',
               'ann_projection_plus_search_median_ms', 'speedup_vs_dense_exact',
               'same_topk_as_dense_exact_rate', 'dense_exact_agreement_with_streaming_oracle',
               'dense_base_bytes',
               'projector_bytes', 'exact_fit_seconds_excluded',
               'ann_build_seconds_excluded', 'dense_reconstruction_seconds_excluded',
               'resident_index_bytes_estimate',
               'benchmark_stage', 'process_peak_rss_mib_cumulative']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    return results
