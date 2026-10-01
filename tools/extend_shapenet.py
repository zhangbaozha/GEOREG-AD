"""Continue a verified ShapeNet search without modifying its scoring engine."""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_shapenet as engine

read = engine.read_json
write = engine.write_json
STAGES = ('plane_candidates', 'penalty_recheck', 'interpolation_recheck')
RANGES = {
    'plane_candidates': {'plane_weight': [2., 3., 4.], 'candidate_k': [32, 64]},
    'penalty_recheck': {'unmatched_penalty': [4., 6., 8., 12., 16.]},
    'interpolation_recheck': {'k': [12, 16, 24], 'power': [.25, .5, .75]},
}
PREVIOUS_FILES = ('protocol.json', 'status.json', 'verification.json', 'splits.json',
                  'selected_config.json', 'history.json', 'full_summary.json', 'final/full/configs.json')
SEARCH_FIELDS = {'matching.' + key for key in ('plane_weight', 'normal_weight',
    'distance_weight', 'unmatched_penalty', 'radius_h', 'candidate_k')} | {'interpolation.k', 'interpolation.power'}


def load_plan(path):
    if path is None:
        return None
    plan = read(path)
    if plan.get('version') != 1 or not isinstance(plan.get('stages'), list) or not plan['stages']:
        raise ValueError('Expected a version-1 search plan with nonempty stages')
    if plan.get('final_evaluation', 'selected') not in ('selected', 'last_validation'):
        raise ValueError('Final evaluation must be selected or last_validation')
    names = set()
    for stage in plan['stages']:
        name, axes = stage.get('name'), stage.get('grid')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise ValueError('Unsafe stage name')
        if name in names or name in ('final', 'smoke', 'source_snapshot'):
            raise ValueError('Duplicate or reserved stage name')
        if stage.get('validation', 'shortlist') not in ('shortlist', 'all'):
            raise ValueError('Validation policy must be shortlist or all')
        names.add(name)
        if not isinstance(axes, dict) or not axes or set(axes) - SEARCH_FIELDS:
            raise ValueError('Only scoring and k/p fields can change with frozen geometry')
        for values in axes.values():
            if not isinstance(values, list) or not values or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError('Grid values must be finite numbers')
    return plan


def ranges_for(plan):
    return RANGES if plan is None else {s['name']: s['grid'] for s in plan['stages']}


def controls_from_previous(previous_root, baseline, previous, plan):
    controls = [baseline, previous]
    if plan is not None:
        controls = engine.unique([baseline, *read(previous_root / 'final/full/configs.json'), previous])
    ids = {m['config'] for m in read(previous_root / 'full_summary.json')['metrics']}
    for config in controls:
        if engine.setting(config['method']) != config or config['id'] not in ids:
            raise ValueError('Previous control configuration differs')
        for section in ('features', 'registration', 'templates', 'object_top_fraction'):
            if config['method'][section] != baseline['method'][section]:
                raise ValueError('Control geometry differs')
    return controls


def grid(stage, incumbent, baseline, previous, controls=None, grid_spec=None):
    values = [*(controls if controls is not None else [baseline, previous]), incumbent]
    if grid_spec is not None:
        fields = list(grid_spec)
        for combination in product(*(grid_spec[field] for field in fields)):
            method = deepcopy(incumbent['method'])
            for field, value in zip(fields, combination):
                section, key = field.split('.')
                old = method[section][key]
                if isinstance(old, int) and value != int(value):
                    raise ValueError('Integer parameter received a fractional value: ' + field)
                method[section][key] = type(old)(value)
            values.append(engine.setting(method))
    elif stage == 'plane_candidates':
        for plane in RANGES[stage]['plane_weight']:
            for candidates in RANGES[stage]['candidate_k']:
                method = deepcopy(incumbent['method'])
                method['matching'].update(plane_weight=plane, candidate_k=candidates)
                values.append(engine.setting(method))
    elif stage == 'penalty_recheck':
        values += engine.variations(incumbent, 'matching', 'unmatched_penalty',
                                    RANGES[stage]['unmatched_penalty'])
    elif stage == 'interpolation_recheck':
        for k in RANGES[stage]['k']:
            for power in RANGES[stage]['power']:
                method = deepcopy(incumbent['method'])
                method['interpolation'].update(k=k, power=power)
                values.append(engine.setting(method))
    else:
        raise ValueError('Unknown extension stage: ' + stage)
    return engine.unique(values)


def preflight(previous_root, plan=None):
    status, verification = (read(previous_root / name) for name in ('status.json', 'verification.json'))
    if status['state'] != 'COMPLETE' or verification['status'] != 'PASS':
        raise ValueError('Previous search must be complete and verified')
    if [verification[k] for k in ('categories', 'test_scans', 'point_valid_scans')] != [52, 1723, 1718]:
        raise ValueError('Unexpected previous search coverage')
    _, baseline_root = engine.checked_protocol(previous_root)
    baseline = engine.setting(read(baseline_root / 'config.json')['method'])
    previous = read(previous_root / 'selected_config.json')
    if engine.setting(previous['method']) != previous or status['selected'] != previous['id']:
        raise ValueError('Previous configuration identity differs')
    manifest = read(baseline_root / 'dataset.json')
    if manifest['smoke'] or manifest['dataset'] != 'shapenet' or len(manifest['categories']) != 52:
        raise ValueError('Expected a full ShapeNet baseline')
    if read(previous_root / 'splits.json') != engine.make_splits(manifest):
        raise ValueError('Previous grouped split differs from declared policy')
    for section in ('features', 'registration', 'templates', 'object_top_fraction'):
        if baseline['method'][section] != previous['method'][section]:
            raise ValueError('Frozen geometry or object aggregation changed')
    for config in (baseline, previous):
        if not any(m['config'] == config['id'] for m in read(previous_root / 'full_summary.json')['metrics']):
            raise ValueError('Previous full comparison is missing a control')
    controls = controls_from_previous(previous_root, baseline, previous, plan)
    counts = {s: len(grid(s, previous, baseline, previous, controls, axes if plan is not None else None))
              for s, axes in ranges_for(plan).items()}
    return baseline_root, baseline, previous, counts


def checked(root, previous_root):
    protocol, baseline_root = engine.checked_protocol(root)
    if protocol['driver_sha256'] != engine.sha256(__file__):
        raise ValueError('Extension driver changed during run')
    plan = protocol.get('search_plan')
    if protocol['ranges'] != ranges_for(plan) or protocol['previous_root'] != str(previous_root):
        raise ValueError('Extension protocol changed')
    if plan is not None:
        engine.check_hashes({protocol['plan_input']: protocol['plan_input_sha256'],
                            str(root / 'search_plan.json'): protocol['plan_snapshot_sha256']})
        if read(root / 'search_plan.json') != plan:
            raise ValueError('Search plan snapshot differs')
    engine.check_hashes({str(previous_root / k): v for k, v in protocol['previous_sha256'].items()})
    return protocol, baseline_root


def report(root, history, baseline, previous, selected, full=None, controls=None):
    lines = ['# ShapeNet continuation search', '',
             'Same official-40 grouped development/validation split and frozen registration as the previous search.',
             'Observed benchmark development; remainder groups are not independent unseen test evidence.', '',
             '|Stage|Development configs|Validation configs|Selected|Validation P-AUROC|Validation P-AP|',
             '|---|---:|---:|---|---:|---:|']
    for row in history:
        m = row['selected_validation']
        lines.append(f'|{row["stage"]}|{row["development_count"]}|{row["validation_count"]}|{row["selected"]}|{100*m["p_auroc"]:.5f}%|{100*m["p_ap"]:.5f}%|')
    import json
    lines += ['', 'Selected method:', '```json', json.dumps(selected['method'], indent=2), '```']
    if full is not None:
        for scope, summary in [('all_52', full), ('official_40', full['official_pcd'])]:
            lines += ['', scope, '', '|Configuration|P-AUROC|P-AP|I-AUROC|I-AP|',
                      '|---|---:|---:|---:|---:|']
            configs = read(root / 'final/full/configs.json')
            for config in configs:
                label = ('Original baseline' if config['id'] == baseline['id'] else
                         'Previous/new selected' if config['id'] == previous['id'] == selected['id'] else
                         'Previous selected' if config['id'] == previous['id'] else
                         'New selected' if config['id'] == selected['id'] else
                         'Earlier selected ' + config['id'] if config['id'] in {c['id'] for c in (controls or [])} else
                         'Additional candidate ' + config['id'])
                m = next(m for m in summary['metrics'] if m['config'] == config['id'])
                lines.append('|' + label + '|' + '|'.join(f'{100*m[k]:.5f}%' for k in engine.METRICS) + '|')
            old = next(m for m in summary['metrics'] if m['config'] == previous['id'])
            new = next(m for m in summary['metrics'] if m['config'] == selected['id'])
            lines += ['', f'Change vs previous selection: P-AUROC {100*(new["p_auroc"]-old["p_auroc"]):+.5f} pp; P-AP {100*(new["p_ap"]-old["p_ap"]):+.5f} pp.']
    temporary = root / 'REPORT.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    engine.replace(temporary, root / 'REPORT.md')


def compare_controls(previous_full, current_full, controls):
    maximum = 0.
    for old, new in [(previous_full, current_full), (previous_full['official_pcd'], current_full['official_pcd'])]:
        for config in controls:
            a = next(m for m in old['metrics'] if m['config'] == config['id'])
            b = next(m for m in new['metrics'] if m['config'] == config['id'])
            errors = [abs(a[k] - b[k]) for k in engine.METRICS]
            errors += [abs(a['remainder'][k] - b['remainder'][k]) for k in ('p_auroc', 'p_ap')]
            maximum = max(maximum, *errors)
    if maximum > 1e-12:
        raise ValueError(f'Previous full control metrics differ: {maximum}')
    return maximum


def execute(args):
    previous_root, root = args.previous_root.resolve(), args.run_root.resolve()
    plan = load_plan(args.plan)
    ranges = ranges_for(plan)
    baseline_root, baseline, previous, _ = preflight(previous_root, plan)
    controls = controls_from_previous(previous_root, baseline, previous, plan)
    smoke = args.command == 'smoke'
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > (os.cpu_count() or 1):
        raise ValueError('Invalid CPU allocation')
    if smoke:
        baseline_root = args.smoke_baseline.resolve()
        if not read(baseline_root / 'dataset.json')['smoke']:
            raise ValueError('Smoke requires a smoke baseline')
        if read(baseline_root / 'config.json')['method'] != baseline['method']:
            raise ValueError('Smoke baseline method differs')
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('Use a new run root or explicit --resume')
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.run.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(os.getpid()))
    try:
        if (root / 'protocol.json').exists():
            protocol, saved_baseline = checked(root, previous_root)
            if saved_baseline != baseline_root or protocol['smoke'] != smoke or protocol.get('search_plan') != plan:
                raise ValueError('Resume scope differs')
        else:
            if shutil.disk_usage(root).free < (2 if smoke else 15) * engine.GIB:
                raise ValueError('Insufficient free disk space')
            if smoke:
                manifest = read(baseline_root / 'dataset.json')
                splits = {r['category']: {c['sample']: 'development' for c in r['cases']} for r in manifest['categories']}
            else:
                splits = read(previous_root / 'splits.json')
            write(root / 'splits.json', splits)
            protocol = {'started_at': engine.now(), 'source_sha256': engine.all_sources(),
                'driver_sha256': engine.sha256(__file__), 'baseline_root': str(baseline_root),
                'baseline_sha256': {n: engine.sha256(baseline_root / n) for n in ('config.json', 'dataset.json', 'protocol.json', 'verification.json', 'summary.json')},
                'previous_root': str(previous_root),
                'previous_sha256': {n: engine.sha256(previous_root / n) for n in PREVIOUS_FILES},
                'splits_sha256': engine.sha256(root / 'splits.json'), 'stages': list(ranges),
                'ranges': ranges, 'smoke': smoke, 'search_plan': plan, 'controls': controls,
                'selection': 'Declared stage validation policy (default: development shortlist), same validation max-min rule; retain all controls and incumbent.',
                'scope': 'Observed benchmark development; frozen registration; final full evaluation after selection.'}
            if plan is not None:
                write(root / 'search_plan.json', plan)
                protocol.update(plan_input=str(args.plan.resolve()), plan_input_sha256=engine.sha256(args.plan),
                                plan_snapshot_sha256=engine.sha256(root / 'search_plan.json'))
            write(root / 'protocol.json', protocol)
            snapshot = root / 'source_snapshot'
            shutil.copytree(engine.CODE / 'georeg3dad', snapshot / 'georeg3dad', ignore=shutil.ignore_patterns('__pycache__'))
            for source in (Path(engine.__file__), Path(__file__)):
                shutil.copy2(source, snapshot / source.name)
        checked(root, previous_root)
        if smoke:
            first_stage = next(iter(ranges))
            configs = grid(first_stage, previous, baseline, previous, controls,
                           ranges[first_stage] if plan is not None else None)
            result = engine.execute_stage(root, 'smoke', 'development', configs, args.workers, args.threads)
            if result['categories'] != 2 or result['test_scans'] != 4:
                raise ValueError('Expected two-category, four-scan smoke')
            checked(root, previous_root)
            write(root / 'verification.json', {'status': 'PASS', **result})
            write(root / 'status.json', {'state': 'COMPLETE', 'finished_at': engine.now(), 'smoke': True})
            print(f'EXTENSION_SMOKE_PASS {len(configs)} configs 2 categories 4 scans', flush=True)
            return
        selected, history = previous, []
        for stage, axes in ranges.items():
            checked(root, previous_root)
            configs = grid(stage, selected, baseline, previous, controls, axes if plan is not None else None)
            development = engine.execute_stage(root, stage, 'development', configs, args.workers, args.threads)
            validation_policy = ('shortlist' if plan is None else
                next(s for s in plan['stages'] if s['name'] == stage).get('validation', 'shortlist'))
            ids = ({c['id'] for c in configs} if validation_policy == 'all' else
                   engine.shortlist(development, selected, baseline) | {c['id'] for c in controls})
            candidates = [c for c in configs if c['id'] in ids]
            validation = engine.execute_stage(root, stage, 'validation', candidates, args.workers, args.threads)
            config_id = engine.choose(validation, selected)
            selected = next(c for c in candidates if c['id'] == config_id)
            history.append({'stage': stage, 'development_count': len(configs), 'validation_count': len(candidates),
                'validation_policy': validation_policy,
                'selected': config_id, 'selected_validation': next(m for m in validation['metrics'] if m['config'] == config_id)})
            write(root / 'history.json', history)
            write(root / 'selected_config.json', selected)
            report(root, history, baseline, previous, selected, controls=controls)
        final_configs = engine.unique([*controls, selected])
        if plan is not None and plan.get('final_evaluation') == 'last_validation':
            final_configs = engine.unique([*controls, *candidates, selected])
        full = engine.execute_stage(root, 'final', 'full', final_configs, args.workers, args.threads)
        if [full[k] for k in ('categories', 'test_scans', 'point_valid_scans')] != [52, 1723, 1718]:
            raise ValueError('Final coverage differs')
        control_error = compare_controls(read(previous_root / 'full_summary.json'), full, controls)
        checked(root, previous_root)
        write(root / 'full_summary.json', full)
        write(root / 'config_shapenet_selected.json', {'dataset': 'shapenet',
            'description': 'Continuation selection on observed benchmark, frozen registration. See protocol.json.', 'method': selected['method']})
        report(root, history, baseline, previous, selected, full, controls)
        write(root / 'verification.json', {'status': 'PASS', 'categories': 52, 'test_scans': 1723,
            'point_valid_scans': 1718, 'max_control_error': full['max_control_error'],
            'previous_full_metric_error': control_error,
            'checks': ['source/driver hashes', 'previous run hashes', 'unchanged grouped splits', 'per-case baseline parity',
                       'prior and original full metric replay', 'compressed array equality', 'coverage']})
        write(root / 'status.json', {'state': 'COMPLETE', 'started_at': protocol['started_at'],
            'finished_at': engine.now(), 'selected': selected['id']})
        print('SHAPENET_EXTENSION_COMPLETE', root, flush=True)
    except BaseException:
        write(root / 'status.json', {'state': 'FAILED', 'updated_at': engine.now(), 'error': traceback.format_exc()})
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'smoke', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--previous-root', type=Path, required=True)
        p.add_argument('--plan', type=Path, help='Optional version-1 JSON scoring-search plan')
        if name != 'preflight':
            p.add_argument('--run-root', type=Path, required=True)
            p.add_argument('--workers', type=int, default=16 if name == 'run' else 2)
            p.add_argument('--threads', type=int, default=2 if name == 'run' else 1)
            p.add_argument('--resume', action='store_true')
        if name == 'smoke': p.add_argument('--smoke-baseline', type=Path, required=True)
    args = parser.parse_args()
    engine.configure_threads(getattr(args, 'threads', 1))
    if args.command == 'preflight':
        _, _, _, counts = preflight(args.previous_root.resolve(), load_plan(args.plan))
        print('EXTENSION_PREFLIGHT_PASS', counts, flush=True)
    else:
        execute(args)


if __name__ == '__main__': main()
