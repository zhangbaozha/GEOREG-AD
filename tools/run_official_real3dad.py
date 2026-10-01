"""Fixed four-arm Real3D rerun with official TXT geometry and centering.

Fresh template/test registration, labels loaded after all predictions, no tuning.
The old experiment is read only for its frozen parameters and comparison table.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import json
import shutil
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georeg3dad.runtime import configure_threads, read_json, write_json, sha256
configure_threads(2)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from georeg3dad.config import method_from_dict
from georeg3dad.datasets import inspect_dataset, official_labels
from georeg3dad.geometry import GeoReg3DAD
from georeg3dad.runner import input_hashes, check_hashes
from georeg3dad.metrics import point_metrics
from georeg3dad.scoring import interpolate, object_score
from tfce_fixed import voxel_mapping, voxel_edges, enhance

VARIANTS = ['knn128_p0', 'knn3_p1', 'voxel_raw', 'voxel_tfce']
METRICS = ['p_auroc', 'p_ap', 'i_auroc', 'i_ap']


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('A new empty output directory is required')
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise ValueError('At least 20 GiB free required')
    source = args.previous.resolve()
    previous = read_json(source / 'PLAN.json')
    method = previous['method']
    cfg = method_from_dict(method)
    if previous['variants'] != VARIANTS or (cfg.interpolation.k, cfg.interpolation.power, cfg.object_top_fraction) != (128, 0, .01):
        raise ValueError('Expected historical four-arm k128 / Top 1% experiment')
    if read_json(source / 'status.json')['state'] != 'COMPLETE':
        raise ValueError('Previous experiment is not complete')
    if read_json(source / 'verification.json')['status'] != 'PASS':
        raise ValueError('Previous experiment did not pass verification')
    selected = ['chicken', 'seahorse', 'starfish'] if args.smoke else []
    manifest = inspect_dataset('real3dad', args.data_root, selected,
                               input_protocol='real3dad-official')
    old_rows = {r['category']: r for r in previous['rows']}
    for row in manifest['categories']:
        old_cases = {c['sample']: c for c in old_rows[row['category']]['cases']}
        if set(old_cases) != {c['sample'] for c in row['cases']}:
            raise ValueError('Source cohort differs')
        for case in row['cases']:
            case['previous_point_gt_valid'] = old_cases[case['sample']]['point_gt_valid']
        if args.smoke:
            # All five historically excluded scans and one normal per affected class.
            row['cases'] = [next(c for c in row['cases'] if not c['is_anomaly'])] + [
                c for c in row['cases'] if not c['previous_point_gt_valid']]
        row['input_sha256'] = input_hashes(row)
        # Verify raw input parity against the historical experiment, not just paths.
        prior_hashes = old_rows[row['category']]['input_sha256']
        for path, digest in row['input_sha256'].items():
            if prior_hashes.get(path) != digest:
                raise ValueError(f'Historical input differs: {path}')
    manifest['test_scans'] = sum(len(r['cases']) for r in manifest['categories'])
    manifest['point_valid_scans'] = manifest['test_scans']
    manifest['smoke'] = args.smoke
    expected = 8 if args.smoke else 1206
    if manifest['test_scans'] != expected:
        raise ValueError('Unexpected scan coverage')
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'georeg3dad').rglob('*.py')) + list((root / 'georeg3dad/protocols').glob('*.json'))
    files += [Path(__file__).resolve(), Path(__file__).with_name('tfce_fixed.py').resolve()]
    source_files = [source / name for name in ('PLAN.json', 'summary.json', 'status.json', 'verification.json')]
    spec = dict(created_at=now(), manifest=manifest, method=method, variants=VARIANTS,
        workers=2 if args.smoke else args.workers, threads=2,
        code_sha256={str(p): sha256(p) for p in files},
        previous_sha256={str(p): sha256(p) for p in source_files}, previous=str(source),
        input_protocol='Official normal PCD / anomaly TXT columns 0:3; independently mean-center all templates and scans.',
        labels='TXT column 3, loaded only after all four score arrays exist; no point exclusions.',
        registration='Fresh seeded FGR+ICP for templates and scans, shared across four arms; no historical pose replay.',
        tfce=dict(E=.5, H=2., connectivity=26, extent='occupied voxel count', normalization='none'),
        selection='None. All historical numerical parameters and four variants retained, object Top 1%.',
        limitation='Previously observed benchmark and tuned geometry. Historical deltas combine input protocol and fresh registration effects.',
        source_reference='https://github.com/M-3LAB/Real3D-AD/blob/main/dataset_pc.py')
    write_json(output / 'PLAN.json', spec)
    write_json(output / 'status.json', dict(state='PLANNED', scans=expected, created_at=now()))
    print(f'OFFICIAL_PLAN_FROZEN scans={expected} points={sum(c["points"] for r in manifest["categories"] for c in r["cases"])} sha256={sha256(output / "PLAN.json")}', flush=True)


def worker(row, method, destination):
    cfg = method_from_dict(method)
    output = Path(destination) / 'categories' / row['category']
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_hashes(row['input_sha256'])
    model = GeoReg3DAD(cfg, threads=2, center=True)
    library = model.prepare(row['templates'])
    write_json(output / 'templates.json', dict(h=library.h, registrations=library.records))
    total = sum(c['points'] for c in row['cases'])
    labels_map = np.lib.format.open_memmap(output / 'point_labels.npy', mode='w+', dtype=np.int8, shape=(total,))
    scores_map = np.lib.format.open_memmap(output / 'point_scores.npy', mode='w+', dtype=np.float64, shape=(4, total))
    records, cursor = [], 0
    for case in row['cases']:
        tick = time.perf_counter()
        scores, xyz, record, parts = model.predict(case['input'], case['sample'] + '.pcd', return_intermediates=True)
        if len(xyz) != case['points']:
            raise ValueError('Input count differs from frozen plan')
        grid, mapping = voxel_mapping(xyz, parts['anchor_xyz'], cfg.features.voxel)
        a, b = voxel_edges(grid)
        enhanced = enhance(parts['raw_scores'], a, b)
        predictions = [scores,
            interpolate(parts['registered_xyz'], parts['registered_anchors'], parts['raw_scores'],
                        replace(cfg.interpolation, k=3, power=1.), threads=2),
            parts['raw_scores'][mapping], enhanced[mapping]]
        # All geometry, registration and four predictions are complete before label access.
        labels = official_labels(case)
        object_scores = {}
        for i, (name, values) in enumerate(zip(VARIANTS, predictions)):
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError('Invalid predictions')
            scores_map[i, cursor:cursor + len(labels)] = values
            object_scores[name] = object_score(values, cfg.object_top_fraction)
        labels_map[cursor:cursor + len(labels)] = labels
        records.append(dict(sample=case['sample'], input=case['input'], gt=case['gt'],
            is_anomaly=case['is_anomaly'], point_gt_valid=True,
            previous_point_gt_valid=case['previous_point_gt_valid'],
            offset=cursor, points=len(labels), positive_points=int(labels.sum()), scores=object_scores,
            prediction=record, seconds=time.perf_counter() - tick))
        cursor += len(labels)
        write_json(output / 'objects.json', records)
        write_json(output / 'progress.json', dict(phase='predict', done=len(records), total=len(row['cases'])))
        print(f'SCAN {row["category"]} {len(records)}/{len(row["cases"])} {case["sample"]} points={len(labels)}', flush=True)
    if cursor != total:
        raise ValueError('Pooled coverage mismatch')
    labels_map.flush(); scores_map.flush()
    metrics = []
    ys = [int(r['is_anomaly']) for r in records]
    for i, name in enumerate(VARIANTS):
        write_json(output / 'progress.json', dict(phase='metrics', variant=name, done=len(records), total=len(records)))
        objects = [r['scores'][name] for r in records]
        metrics.append(dict(variant=name, **point_metrics(labels_map, scores_map[i]),
            i_auroc=float(roc_auc_score(ys, objects)), i_ap=float(average_precision_score(ys, objects))))
    del labels_map, scores_map
    check_hashes(row['input_sha256'])
    result = dict(category=row['category'], scans=len(records), point_valid_scans=len(records), points=total,
        positive_points=sum(r['positive_points'] for r in records), metrics=metrics, seconds=time.perf_counter() - started,
        files_sha256={name: sha256(output / name) for name in ('point_labels.npy', 'point_scores.npy', 'objects.json', 'templates.json')})
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', dict(phase='COMPLETE', done=len(records), total=len(records)))
    return result


def report(output, rows, elapsed, spec):
    aggregates = []
    for name in VARIANTS:
        metrics = [next(m for m in r['metrics'] if m['variant'] == name) for r in rows]
        aggregates.append(dict(variant=name, **{k: statistics.mean(m[k] for m in metrics) for k in METRICS}))
    summary = dict(categories=len(rows), scans=sum(r['scans'] for r in rows),
        point_valid_scans=sum(r['point_valid_scans'] for r in rows), points=sum(r['points'] for r in rows),
        aggregates=aggregates, categories_detail=rows, elapsed_seconds=elapsed)
    write_json(output / 'summary.json', summary)
    lines = ['# Real3D-AD official-input rerun', '', spec['input_protocol'], '', spec['registration'], '',
        f"Coverage: {summary['categories']} categories / {summary['scans']} scans / {summary['point_valid_scans']} point-valid scans / {summary['points']:,} points.",
        '', 'Fixed historical parameters; object Top 1%; no retuning. Point AUROC/AP pooled within class, then equal class means.', '',
        '|Method|P-AUROC|P-AP|I-AUROC|I-AP|', '|---|---:|---:|---:|---:|']
    for r in aggregates:
        lines.append('|' + r['variant'] + '|' + '|'.join(f'{r[k]:.9f}' for k in METRICS) + '|')
    if not spec['manifest']['smoke']:
        old = read_json(Path(spec['previous']) / 'summary.json')
        lines += ['', 'Changes from historical PCD-input experiment, in percentage points. These combine input protocol, centering and fresh-registration effects.', '',
            '|Method|P-AUROC delta|P-AP delta|I-AUROC delta|I-AP delta|', '|---|---:|---:|---:|---:|']
        for r in aggregates:
            before = next(m for m in old['aggregates'] if m['variant'] == r['variant'])
            lines.append('|' + r['variant'] + '|' + '|'.join(f'{100 * (r[k] - before[k]):+.6f}' for k in METRICS) + '|')
    lines += ['', spec['limitation'], '', 'Independent sklearn/GT verification is recorded separately in INDEPENDENT_VERIFICATION.json.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def run(args):
    output = args.output.resolve(); spec = read_json(output / 'PLAN.json')
    if read_json(output / 'status.json')['state'] != 'PLANNED':
        raise ValueError('Require fresh PLANNED state')
    started = time.perf_counter(); digest = sha256(output / 'PLAN.json')
    status = dict(state='RUNNING', started_at=now(), plan_sha256=digest, completed_categories=[])
    write_json(output / 'status.json', status)
    try:
        check_hashes(spec['code_sha256']); check_hashes(spec['previous_sha256'])
        rows = []
        with ProcessPoolExecutor(max_workers=spec['workers'], mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(worker, row, spec['method'], str(output)) for row in spec['manifest']['categories']]
            for future in as_completed(tasks):
                rows.append(future.result())
                status['completed_categories'] = sorted(r['category'] for r in rows)
                write_json(output / 'status.json', status)
        check_hashes(spec['code_sha256']); check_hashes(spec['previous_sha256'])
        if sha256(output / 'PLAN.json') != digest:
            raise ValueError('Plan changed')
        summary = report(output, sorted(rows, key=lambda r: r['category']), time.perf_counter() - started, spec)
        if summary['scans'] != spec['manifest']['test_scans'] or summary['point_valid_scans'] != summary['scans']:
            raise ValueError('Final cohort mismatch')
        write_json(output / 'verification.json', dict(status='PASS', plan_sha256=digest,
            scans=summary['scans'], point_valid_scans=summary['point_valid_scans'], points=summary['points'],
            checks=['input and code hashes', 'full scan/point coverage', 'fresh shared registration', 'label isolation', 'all four fixed variants']))
        status.update(state='COMPLETE', finished_at=now(), elapsed_seconds=summary['elapsed_seconds'])
        write_json(output / 'status.json', status)
        print('OFFICIAL_EXPERIMENT_COMPLETE ' + json.dumps(summary['aggregates']), flush=True)
    except BaseException:
        status.update(state='FAILED', error=traceback.format_exc()); write_json(output / 'status.json', status)
        raise


def verify_category(output, row):
    folder = Path(output) / 'categories' / row['category']
    summary = read_json(folder / 'summary.json')
    check_hashes({str(folder / k): v for k, v in summary['files_sha256'].items()})
    y = np.load(folder / 'point_labels.npy', mmap_mode='r')
    scores = np.load(folder / 'point_scores.npy', mmap_mode='r')
    objects = read_json(folder / 'objects.json')
    if [r['sample'] for r in objects] != [c['sample'] for c in row['cases']]:
        raise ValueError('Saved case coverage differs')
    if scores.shape != (4, len(y)) or len(y) != sum(c['points'] for c in row['cases']):
        raise ValueError('Saved shape mismatch')
    cursor = 0
    for case, obj in zip(row['cases'], objects):
        # Independent parsing of raw fourth column, not official_labels().
        truth = np.loadtxt(case['gt'], usecols=3, ndmin=1) if case['gt'] else np.zeros(case['points'])
        if obj['offset'] != cursor or not np.array_equal(y[cursor:cursor + len(truth)], truth):
            raise ValueError('Archived labels differ from source GT')
        for i, name in enumerate(VARIANTS):
            values = scores[i, cursor:cursor + len(truth)]
            n = max(1, int(np.ceil(.01 * len(values))))
            reference = float(np.sort(values)[-n:].mean())
            if not np.isclose(reference, obj['scores'][name], rtol=1e-12, atol=1e-12):
                raise ValueError('Saved object score differs')
        cursor += len(truth)
    error = 0.
    for i, name in enumerate(VARIANTS):
        ys = [int(r['is_anomaly']) for r in objects]; ss = [r['scores'][name] for r in objects]
        actual = dict(p_auroc=float(roc_auc_score(y, scores[i])), p_ap=float(average_precision_score(y, scores[i])),
            i_auroc=float(roc_auc_score(ys, ss)), i_ap=float(average_precision_score(ys, ss)))
        previous = next(m for m in summary['metrics'] if m['variant'] == name)
        error = max(error, max(abs(actual[k] - previous[k]) for k in METRICS))
    if error > 1e-10:
        raise ValueError(f'Independent metric mismatch: {error}')
    print('INDEPENDENT_CATEGORY_PASS ' + row['category'], flush=True)
    return dict(category=row['category'], scans=len(objects), max_metric_error=error)


def verify(args):
    output = args.output.resolve(); spec = read_json(output / 'PLAN.json')
    if read_json(output / 'status.json')['state'] != 'COMPLETE':
        raise ValueError('Experiment not complete')
    check_hashes(spec['code_sha256']); check_hashes(spec['previous_sha256'])
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(verify_category, str(output), row) for row in spec['manifest']['categories']]
        checks = [f.result() for f in as_completed(futures)]
    summary = read_json(output / 'summary.json')
    for name in VARIANTS:
        rows = [next(m for m in r['metrics'] if m['variant'] == name) for r in summary['categories_detail']]
        aggregate = next(m for m in summary['aggregates'] if m['variant'] == name)
        for k in METRICS:
            if abs(statistics.mean(r[k] for r in rows) - aggregate[k]) > 1e-12:
                raise ValueError('Macro mismatch')
    write_json(output / 'INDEPENDENT_VERIFICATION.json', dict(status='PASS', categories=checks,
        max_metric_error=max(r['max_metric_error'] for r in checks),
        checks=['sklearn full-array AUROC/AP', 'raw GT fourth-column equality', 'sorted Top1% object scores', 'archive hashes', 'macro averages']))
    print('INDEPENDENT_VERIFICATION_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    for name in ('data-root', 'previous', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=4)
    for name in ('run', 'verify'):
        p = sub.add_parser(name); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'plan': plan, 'run': run, 'verify': verify}[args.command](args)
