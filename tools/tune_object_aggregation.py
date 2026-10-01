"""Tune object aggregation from verified immutable point predictions on either OS.

The existing grouped validation partition alone selects one fraction per dataset.
No point labels, registration, interpolation, or point scores are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from georeg3dad.scoring import object_score
from georeg3dad.config import method_from_dict

# 1e-12 is representable by the existing config and gives exactly one point
# for every supported scan; enforce the one-point condition below.
FRACTIONS = (1e-12, .0001, .00025, .0005, .001, .0025, .005, .01,
             .02, .05, .1, .2, .3, .5, 1.)
BASELINE = 'top_0.01'
SPLITS = ('development', 'validation', 'remainder', 'full')


def name(fraction):
    return 'maximum' if fraction == 1e-12 else f'top_{fraction:g}'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores_for(values, fractions=FRACTIONS):
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite nonempty 1-D point scores')
    if 1e-12 in fractions and np.ceil(len(values) * 1e-12) != 1:
        raise ValueError('Maximum shortcut is not one point')
    return {name(f): object_score(values, f) for f in fractions}


def check_splits(dataset, category, cases, assignment):
    if set(assignment) != {c['sample'] for c in cases}:
        raise ValueError(f'Split coverage changed: {category}')
    groups = {}
    for c in cases:
        sample = c['sample']
        group = sample.split('_', 1)[0] if dataset == 'real3dad' else re.search(r'(\d+)$', sample).group(1)
        groups.setdefault(group, set()).add(assignment[sample])
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError(f'Variant group crosses partitions: {category}')
    for split in SPLITS[:-1]:
        if {bool(c['is_anomaly']) for c in cases if assignment[c['sample']] == split} != {False, True}:
            raise ValueError(f'Both object labels required: {category}/{split}')


def prepare(args):
    root = args.source_root.resolve()
    dataset = args.dataset
    verification = read(root/'verification.json')
    if verification['status'] != 'PASS' or read(root/'status.json')['state'] != 'COMPLETE':
        raise ValueError('Source must be COMPLETE/PASS')
    expected = (12, 1206, 1201) if dataset == 'real3dad' else (52, 1723, 1718)
    if tuple(verification[k] for k in ('categories', 'test_scans', 'point_valid_scans')) != expected:
        raise ValueError('Source dataset coverage mismatch')
    split_raw = read(args.splits)
    splits = {k: v['samples'] if dataset == 'real3dad' else v for k, v in split_raw.items()}
    config_path = root/('config.json' if dataset == 'real3dad' else 'config_shapenet_selected.json')
    config = read(config_path)
    method_from_dict(config['method'])
    if config['method']['object_top_fraction'] != .01:
        raise ValueError('This experiment expects top-1% as the incumbent')
    paths = [root/'verification.json', root/'status.json', root/'protocol.json', config_path, args.splits.resolve()]
    rows = []
    if dataset == 'real3dad':
        manifest = read(root/'dataset.json')
        paths.append(root/'dataset.json')
        categories = [(r['category'], r['source_split']) for r in manifest['categories']]
        folders = {category: root/'results'/category for category, _ in categories}
        config_id = 'current_point_selected'
    else:
        config_id = args.config_id
        configs = read(root/'final/full/configs.json')
        selected = next(c for c in configs if c['id'] == config_id)
        if selected['method'] != config['method']:
            raise ValueError('Requested prediction config is not the selected method')
        paths.append(root/'final/full/configs.json')
        folders = {p.name: p for p in (root/'final/full').iterdir() if p.is_dir() and (p/'summary.json').exists()}
        categories = [(category, read(p/'summary.json')['source_split']) for category, p in folders.items()]
    if set(splits) != {c for c, _ in categories} or len(categories) != expected[0]:
        raise ValueError('Category coverage mismatch')
    for category, source_split in sorted(categories):
        folder = folders[category]
        summary = read(folder/'summary.json')
        paths.append(folder/'summary.json')
        cases_path = folder/('cases.json' if dataset == 'real3dad' else 'objects.json')
        if sha(cases_path) != summary['files_sha256'][cases_path.name]:
            raise ValueError('Case metadata hash mismatch')
        paths.append(cases_path)
        cases = read(cases_path)
        if len(cases) != summary['test_scans']:
            raise ValueError('Incomplete source cases')
        check_splits(dataset, category, cases, splits[category])
        if dataset == 'real3dad':
            if summary['config_sha256'] != sha(config_path):
                raise ValueError('Source config hash mismatch')
            metrics = summary['metrics']
        else:
            if summary['config_sha256'] != sha(root/'final/full/configs.json'):
                raise ValueError('Source config list hash mismatch')
            metrics = next(m for m in summary['metrics'] if m['config'] == config_id)
        rows.append(dict(category=category, source_split=source_split, folder=str(folder),
                         summary=summary, cases=cases, assignment=splits[category],
                         original_metrics=metrics, config_id=config_id, dataset=dataset))
    if sum(len(r['cases']) for r in rows) != expected[1]:
        raise ValueError('Total scan count mismatch')
    metadata = {str(p): sha(p) for p in paths}
    if args.smoke:
        rows = [r for r in rows if dataset == 'real3dad' or r['source_split'] == 'pcd'][:2]
        for row in rows:
            chosen = []
            for split in SPLITS[:-1]:
                for label in (False, True):
                    chosen.append(next(c for c in row['cases'] if row['assignment'][c['sample']] == split and bool(c['is_anomaly']) == label))
            row['cases'] = chosen
    return rows, config, metadata, splits


def category_worker(row, output, smoke):
    folder = Path(row['folder'])
    fractions = row.get('fractions', FRACTIONS)
    previous = {r['sample']: r for r in read(row['previous_objects'])} if row.get('previous_objects') else {}
    records, hashes = [], {}
    max_error = 0.
    previous_error = 0.
    for case in row['cases']:
        sample = case['sample']
        relative = case['score_file'] if row['dataset'] == 'real3dad' else f'predictions/{sample}.npz'
        path = folder/relative
        actual_hash = sha(path)
        if actual_hash != row['summary']['files_sha256'][relative]:
            raise ValueError(f'Prediction hash mismatch: {path}')
        hashes[str(path)] = actual_hash
        with np.load(path, allow_pickle=False) as archive:
            values = archive['scores' if row['dataset'] == 'real3dad' else row['config_id']]
        object_scores = scores_for(values, fractions)
        baseline = case['object_score'] if row['dataset'] == 'real3dad' else case['scores'][row['config_id']]
        error = abs(object_scores[BASELINE] - baseline)
        max_error = max(max_error, error)
        if error > 1e-12 * max(1., abs(baseline)):
            raise ValueError(f'Object score replay mismatch: {path}')
        if previous:
            old = previous[sample]
            if old['is_anomaly'] != bool(case['is_anomaly']) or old['split'] != row['assignment'][sample] or old['points'] != len(values):
                raise ValueError('Previous object metadata changed')
            for key in old['scores'].keys() & object_scores.keys():
                delta = abs(old['scores'][key] - object_scores[key])
                previous_error = max(previous_error, delta)
                if delta > 1e-12 * max(1., abs(old['scores'][key])):
                    raise ValueError(f'Previous object score replay failed: {sample}/{key}')
        records.append(dict(sample=sample, is_anomaly=bool(case['is_anomaly']),
                            split=row['assignment'][sample], points=len(values), scores=object_scores))
        write(Path(output)/'categories'/row['category']/'progress.json',
              dict(done=len(records), total=len(row['cases'])))
    results = []
    for split in SPLITS:
        cases = [r for r in records if split == 'full' or r['split'] == split]
        labels = [int(c['is_anomaly']) for c in cases]
        if set(labels) != {0, 1}:
            raise ValueError('Object AUROC needs both labels')
        for fraction in fractions:
            scores = [c['scores'][name(fraction)] for c in cases]
            results.append(dict(category=row['category'], source_split=row['source_split'],
                split=split, aggregation=name(fraction), top_fraction=fraction,
                scans=len(cases), normal=labels.count(0), anomaly=labels.count(1),
                i_auroc=float(roc_auc_score(labels, scores)), i_ap=float(average_precision_score(labels, scores))))
    control = next(r for r in results if r['split'] == 'full' and r['aggregation'] == BASELINE)
    metric_error = max(abs(control[k] - row['original_metrics'][k]) for k in ('i_auroc', 'i_ap')) if not smoke else None
    if metric_error is not None and metric_error > 1e-12:
        raise ValueError('Full baseline metric replay failed')
    base = Path(output)/'categories'/row['category']
    write(base/'objects.json', records)
    write(base/'prediction_sha256.json', hashes)
    result = dict(category=row['category'], source_split=row['source_split'], scans=len(records),
                  metrics=results, max_object_replay_error=max_error,
                  max_previous_object_replay_error=previous_error if previous else None,
                  full_metric_replay_error=metric_error, point_metrics_preserved={k: row['original_metrics'][k] for k in ('p_auroc','p_ap')})
    write(base/'summary.json', result)
    return result


def aggregate(rows, dataset, fractions=FRACTIONS):
    flat = [m for r in rows for m in r['metrics']]
    scopes = ['all'] if dataset == 'real3dad' else ['all', 'official_pcd', 'new_pcd']
    result = []
    for scope in scopes:
        for split in SPLITS:
            for fraction in fractions:
                selected = [r for r in flat if r['split'] == split and r['aggregation'] == name(fraction)
                            and (scope == 'all' or r['source_split'] == ('pcd' if scope == 'official_pcd' else 'new_pcd'))]
                if not selected:
                    continue
                result.append(dict(scope=scope, split=split, aggregation=name(fraction), top_fraction=fraction,
                    categories=len(selected), scans=sum(r['scans'] for r in selected),
                    normal=sum(r['normal'] for r in selected), anomaly=sum(r['anomaly'] for r in selected),
                    i_auroc=statistics.mean(r['i_auroc'] for r in selected), i_ap=statistics.mean(r['i_ap'] for r in selected)))
    return result


def select(aggregates, dataset, incumbent=.01):
    scope = 'all' if dataset == 'real3dad' else 'official_pcd'
    candidates = [r for r in aggregates if r['scope'] == scope and r['split'] == 'validation']
    baseline = next(r for r in candidates if r['aggregation'] == name(incumbent))
    best_auc = max(r['i_auroc'] for r in candidates)
    tied = [r for r in candidates if abs(r['i_auroc'] - best_auc) <= 1e-12]
    best = max(tied, key=lambda r: (r['i_ap'], r['aggregation'] == name(incumbent), -abs(r['top_fraction']-incumbent), -r['top_fraction']))
    return best if best['i_auroc'] > baseline['i_auroc'] + 1e-4 else baseline


def self_test():
    rng = np.random.default_rng(20260926)
    for values in (np.array([2.]), np.array([0., 1., 1., 2., 3., 3.]), rng.normal(size=2003)):
        original = values.copy()
        fractions = tuple(sorted(set(FRACTIONS + (.00075, .00125, .0015, .00175, .002, .00225, .00275, .003, .0035, .004, .0045))))
        actual = scores_for(values, fractions)
        for fraction in fractions:
            count = max(1, int(np.ceil(fraction * len(values))))
            expected = float(np.sort(values)[-count:].mean())
            np.testing.assert_allclose(actual[name(fraction)], expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(values, original)
    # Exact ties must have AUC 0.5; no numerical score perturbations are used.
    assert roc_auc_score([0, 1, 0, 1], [1., 1., 1., 1.]) == .5
    candidates = [dict(scope='all', split='validation', aggregation=name(f), top_fraction=f, i_auroc=a, i_ap=p)
                  for f, a, p in ((.001, .8, .7), (.00125, .8, .9), (.0025, .81, .8))]
    assert select(candidates[:2], 'real3dad', .001)['top_fraction'] == .001
    assert select(candidates, 'real3dad', .001)['top_fraction'] == .0025
    print('SELF_TEST_PASS fractions=26 ties=PASS sorted_reference=PASS incumbent=PASS')


def run(args):
    start = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        raise ValueError('Invalid worker count')
    status = dict(state='PREPARING', started_at=now(), dataset=args.dataset, workers=args.workers, smoke=args.smoke)
    write(output/'status.json', status)
    try:
        rows, config, metadata, splits = prepare(args)
        fractions = tuple(sorted(set(args.fractions or FRACTIONS)))
        incumbent = args.incumbent
        if any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions) or .01 not in fractions or incumbent not in fractions:
            raise ValueError('Fractions must be finite in (0, 1], including top-1% control and incumbent')
        previous_summary = None
        if args.previous_run:
            previous_root = args.previous_run.resolve()
            if read(previous_root/'status.json')['state'] != 'COMPLETE' or read(previous_root/'verification.json')['status'] != 'PASS':
                raise ValueError('Previous aggregation run must be COMPLETE/PASS')
            previous_summary = read(previous_root/'summary.json')
            if previous_summary['dataset'] != args.dataset or previous_summary['selected']['top_fraction'] != incumbent or read(previous_root/'splits.json') != splits:
                raise ValueError('Previous selected fraction or split mismatch')
            if not set(read(previous_root/'protocol.json')['fractions']).issubset(fractions):
                raise ValueError('Continuation must replay every previous fraction')
            for filename in ('status.json', 'verification.json', 'summary.json', 'splits.json', 'protocol.json'):
                path = previous_root/filename
                metadata[str(path)] = sha(path)
            for row in rows:
                path = previous_root/'categories'/row['category']/'objects.json'
                row['previous_objects'] = str(path)
                metadata[str(path)] = sha(path)
        for row in rows:
            row['fractions'] = fractions
        write(output/'splits.json', splits)
        write(output/'source_config.json', config)
        protocol = dict(dataset=args.dataset, source_root=str(args.source_root.resolve()),
            fractions=list(fractions), baseline=BASELINE, incumbent=incumbent,
            previous_run=str(args.previous_run.resolve()) if args.previous_run else None, source_metadata_sha256=metadata,
            driver_sha256=sha(__file__), scoring_sha256=sha(Path(__file__).resolve().parents[1]/'georeg3dad/scoring.py'),
            selection='One global fraction per dataset. Maximize macro validation I-AUROC; ties within 1e-12 use I-AP, then incumbent, then nearest incumbent, then smaller fraction. Retain incumbent unless I-AUROC improves >1e-4.',
            selection_categories='12 Real3D categories or official 40 ShapeNet categories',
            scope='Post-hoc tuning on previously observed benchmark; remainder/full are descriptive internal checks, not an independent unseen test.',
            point_predictions='Reused unchanged, every source archive SHA256 verified. No point labels used.',
            execution=dict(workers=args.workers, threads_per_worker=1, device='CPU'), smoke=args.smoke)
        write(output/'protocol.json', protocol)
        done = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(category_worker, row, str(output), args.smoke): row['category'] for row in rows}
            status.update(state='RUNNING', total_categories=len(rows), completed_categories=[])
            write(output/'status.json', status)
            for future in as_completed(futures):
                result = future.result()
                done.append(result)
                status['completed_categories'] = sorted(r['category'] for r in done)
                write(output/'status.json', status)
                print(f"DONE {result['category']} {len(done)}/{len(rows)}", flush=True)
        done.sort(key=lambda r: r['category'])
        for path, expected in metadata.items():
            if sha(path) != expected:
                raise ValueError(f'Source metadata changed during run: {path}')
        if sha(__file__) != protocol['driver_sha256'] or sha(Path(__file__).resolve().parents[1]/'georeg3dad/scoring.py') != protocol['scoring_sha256']:
            raise ValueError('Source code changed during run')
        aggregates = aggregate(done, args.dataset, fractions)
        previous_metric_error = None
        if previous_summary and not args.smoke:
            index = {(r['scope'], r['split'], r['aggregation']): r for r in aggregates}
            previous_metric_error = max(abs(index[(r['scope'], r['split'], r['aggregation'])][k] - r[k])
                                        for r in previous_summary['aggregates'] for k in ('i_auroc', 'i_ap'))
            if previous_metric_error > 1e-12:
                raise ValueError('Previous full/partition metric replay failed')
        selected = select(aggregates, args.dataset, incumbent)
        result = dict(dataset=args.dataset, selected=None if args.smoke else selected,
                      aggregates=aggregates, categories=[r['category'] for r in done],
                      elapsed_seconds=time.perf_counter()-start, smoke=args.smoke)
        write(output/'summary.json', result)
        csv_write(output/'comparison.csv', aggregates)
        csv_write(output/'per_category.csv', [m for r in done for m in r['metrics']])
        if not args.smoke:
            exported = deepcopy(config)
            exported['method']['object_top_fraction'] = selected['top_fraction']
            exported['description'] = 'Object I-AUROC validation-selected aggregation; point method preserved. See object tuning protocol.'
            method_from_dict(exported['method'])
            write(output/f'config_{args.dataset}_object.json', exported)
            write(output/'selected.json', selected)
        verification = dict(status='PASS', categories=len(done), scans=sum(r['scans'] for r in done),
            fractions=len(fractions), max_object_replay_error=max(r['max_object_replay_error'] for r in done),
            max_previous_object_replay_error=max(r['max_previous_object_replay_error'] for r in done) if args.previous_run else None,
            max_previous_metric_replay_error=previous_metric_error,
            max_full_metric_replay_error=None if args.smoke else max(r['full_metric_replay_error'] for r in done),
            checks=['source archive hashes', 'metadata/source stability', 'grouped split coverage and no variant leakage',
                    'top-1% per-object replay', 'original full object metric replay (production)', 'both labels in every class/partition',
                    'all object-only scans included', 'validation-only global selection'])
        write(output/'verification.json', verification)
        lines = [f'# {args.dataset}: object aggregation search', '',
                 'SMOKE ONLY' if args.smoke else f"Validation-selected: {selected['aggregation']}", '',
                 protocol['scope'], '', '|Scope|Split|Aggregation|I-AUROC (%)|I-AP (%)|', '|---|---|---|---:|---:|']
        for r in aggregates:
            lines.append(f"|{r['scope']}|{r['split']}|{r['aggregation']}|{r['i_auroc']*100:.5f}|{r['i_ap']*100:.5f}|")
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        status.update(state='COMPLETE', finished_at=now(), elapsed_seconds=result['elapsed_seconds'])
        write(output/'status.json', status)
        print(json.dumps(dict(state='COMPLETE', verification=verification, selected=result['selected'], seconds=result['elapsed_seconds'])), flush=True)
    except BaseException:
        status.update(state='FAILED', failed_at=now(), error=traceback.format_exc())
        write(output/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    p = sub.add_parser('run')
    p.add_argument('--dataset', choices=('real3dad', 'shapenet'), required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config-id', default='c_ff06afd13023')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fractions', type=float, nargs='+', help='Fractions in (0, 1], including top-1% control and incumbent')
    p.add_argument('--incumbent', type=float, default=.01)
    p.add_argument('--previous-run', type=Path, help='Replay an earlier aggregation run before selecting a continuation')
    args = parser.parse_args()
    self_test() if args.command == 'self-test' else run(args)


if __name__ == '__main__':
    main()
