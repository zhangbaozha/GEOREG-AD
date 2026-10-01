"""Portable category subprocesses, bounded memory, resumable output and metrics."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .runtime import available_memory, read_json, replace, sha256, source_hashes, stop_process_tree, write_json

GIB = 1024**3
METRICS = ("p_auroc", "p_ap", "i_auroc", "i_ap")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def input_hashes(row):
    paths = set(row['templates'])
    paths.update(c['test'] for c in row['cases'])
    paths.update(c['gt'] for c in row['cases'] if c['gt'])
    return {p: sha256(p) for p in sorted(paths)}


def check_hashes(values):
    for path, expected in values.items():
        if sha256(path) != expected:
            raise ValueError(f"File changed: {path}")


def evaluate_category(run_root, category, threads):
    # Numerical imports happen only after the CLI sets thread environment variables.
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    from .config import method_from_dict
    from .datasets import align_labels, official_labels
    from .geometry import GeoReg3DAD
    from .metrics import point_metrics

    run_root = Path(run_root)
    manifest = read_json(run_root/'dataset.json')
    row = next(r for r in manifest['categories'] if r['category'] == category)
    check_hashes(row['input_sha256'])
    protocol = read_json(run_root/'protocol.json')
    if protocol['source_sha256'] != source_hashes():
        raise ValueError("Source code changed after launch")
    settings = read_json(run_root/'config.json')
    model = GeoReg3DAD(method_from_dict(settings['method']), threads,
                      center=manifest.get('center_inputs', False))
    output = run_root/'results'/category
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    library = model.prepare(row['templates'])
    write_json(output/'templates.json', {'h': library.h, 'points': len(library.xyz), 'registrations': library.records})
    total_points = sum(c['points'] for c in row['cases'] if c['point_gt_valid'])
    work = output/'work'; work.mkdir(exist_ok=True)
    labels_map = np.lib.format.open_memmap(work/'labels.npy', mode='w+', dtype=np.int8, shape=(total_points,))
    scores_map = np.lib.format.open_memmap(work/'scores.npy', mode='w+', dtype=np.float64, shape=(total_points,))
    records, cursor, files = [], 0, {}
    try:
        for index, case in enumerate(row['cases'], 1):
            scores, xyz, record = model.predict(case.get('input', case['test']), case['sample']+'.pcd')
            if len(scores) != case['points']:
                raise ValueError("Input point count differs from dataset manifest")
            if manifest.get('input_protocol') == 'real3dad-official':
                labels = official_labels(case)
            elif not case['point_gt_valid']:
                labels = np.empty(0, dtype=np.int8)
            elif case['gt']:
                labels = align_labels(xyz, case['gt'], manifest['dataset'], threads)
            else:
                labels = np.zeros(len(scores), dtype=np.int8)
            if case['point_gt_valid']:
                labels_map[cursor:cursor+len(labels)] = labels
                scores_map[cursor:cursor+len(labels)] = scores
                cursor += len(labels)
            path = output/'scores'/f"{case['sample']}.npz"
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
            np.savez_compressed(temporary, scores=scores, labels=labels)
            replace(temporary, path)
            # Validate the serialized arrays, including excluded-GT object-only scans.
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(archive['scores'], scores) or not np.array_equal(archive['labels'], labels):
                    raise ValueError(f"Prediction archive verification failed: {path}")
            files[path.relative_to(output).as_posix()] = sha256(path)
            records.append({**case, **record, 'score_file': path.relative_to(output).as_posix()})
            write_json(output/'progress.json', {'phase':'predict','done':index,'total':len(row['cases']),'sample':case['sample']})
        if cursor != total_points or len(np.unique(labels_map)) != 2:
            raise ValueError("Point metrics require both labels and complete point coverage")
        write_json(output/'progress.json', {'phase':'metrics','done':len(records),'total':len(records)})
        ys = [int(c['is_anomaly']) for c in records]
        ss = [c['object_score'] for c in records]
        metrics = {**point_metrics(labels_map, scores_map),
                   'i_auroc': float(roc_auc_score(ys, ss)), 'i_ap': float(average_precision_score(ys, ss))}
        write_json(output/'cases.json', records)
    finally:
        labels_map.flush(); scores_map.flush()
        labels_map._mmap.close(); scores_map._mmap.close()
    # Delete only this category's transient arrays after successful metrics.
    (work/'labels.npy').unlink(); (work/'scores.npy').unlink()
    for name in ('templates.json', 'cases.json'):
        files[name] = sha256(output/name)
    write_json(output/'summary.json', {'category':category,'source_split':row['source_split'],
        'test_scans':len(records),'point_valid_scans':sum(c['point_gt_valid'] for c in records),
        'points':total_points,'metrics':metrics,'seconds':time.perf_counter()-started,
        'config_sha256':sha256(run_root/'config.json'),'files_sha256':files})


def verify_category(run_root, row):
    output = Path(run_root)/'results'/row['category']
    summary = read_json(output/'summary.json')
    if summary['config_sha256'] != sha256(Path(run_root)/'config.json'):
        raise ValueError("Configuration changed")
    if summary['test_scans'] != len(row['cases']) or summary['point_valid_scans'] != sum(c['point_gt_valid'] for c in row['cases']):
        raise ValueError("Incomplete category coverage")
    expected_files = {f"scores/{c['sample']}.npz" for c in row['cases']}
    if expected_files != {p.relative_to(output).as_posix() for p in (output/'scores').glob('*.npz')}:
        raise ValueError("Unexpected or missing predictions")
    check_hashes({str(output/path): expected for path, expected in summary['files_sha256'].items()})
    return summary


def memory_estimate(row):
    # One global int64 sort index, mapped inputs, and bounded threshold chunks.
    points = sum(c['points'] for c in row['cases'] if c['point_gt_valid'])
    maximum = max(c['points'] for c in row['cases'])
    return 768*1024**2 + points*24 + maximum*160


def summarize(run_root, manifest, rows):
    import statistics
    groups = {'all_selected':rows}
    if manifest['dataset'] == 'shapenet':
        groups['official_pcd'] = [r for r in rows if r['source_split']=='pcd']
    result = {'scope': 'smoke' if manifest['smoke'] else 'selected_categories', 'groups':{}}
    for name, selected in groups.items():
        if selected:
            result['groups'][name] = {'categories':len(selected),
                'test_scans':sum(r['test_scans'] for r in selected),
                'point_valid_scans':sum(r['point_valid_scans'] for r in selected),
                'metrics':{key:statistics.mean(r['metrics'][key] for r in selected) for key in METRICS}}
    write_json(Path(run_root)/'summary.json', result)
    lines = ['# GeoReg3DAD results', '', 'Scope: '+result['scope'], '',
             '|Group|Classes|Scans|Point-valid|P-AUROC|P-AP|I-AUROC|I-AP|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in result['groups'].items():
        lines.append(f"|{name}|{value['categories']}|{value['test_scans']}|{value['point_valid_scans']}|"+
                     '|'.join(f"{100*value['metrics'][k]:.5f}%" for k in METRICS)+'|')
    lines += ['', 'Point metrics pool valid points within each category, then average categories equally.',
              'Known invalid GT scans remain in object metrics. Smoke results are not benchmark results.',
              'Fresh registration can vary across machines and Open3D builds; fixed-transform parity is tested separately.']
    (Path(run_root)/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(manifest, settings, output, workers=4, threads=2, resume=False):
    output = Path(output).expanduser().resolve()
    if workers < 1 or threads < 1 or workers*threads > (os.cpu_count() or 1):
        raise ValueError("workers * threads must fit the available logical CPUs")
    if output == Path(manifest['data_root']) or Path(manifest['data_root']) in output.parents:
        raise ValueError("Keep output outside the source dataset")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 5*GIB:
        raise ValueError("At least 5 GiB free disk space required")
    lock = output/'.run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    processes = {}
    status = {'state':'PREPARING','started_at':now(),'workers':workers,'threads':threads}
    accepted = False
    try:
        for row in manifest['categories']:
            row['input_sha256'] = input_hashes(row)
        if resume:
            if read_json(output/'config.json') != settings or read_json(output/'dataset.json') != manifest:
                raise ValueError("Resume requires unchanged configuration and inputs")
            if read_json(output/'protocol.json')['source_sha256'] != source_hashes():
                raise ValueError("Resume requires unchanged source code")
            status['started_at'] = read_json(output/'status.json')['started_at']
        else:
            import importlib.metadata
            import platform
            write_json(output/'config.json', settings)
            write_json(output/'dataset.json', manifest)
            write_json(output/'protocol.json', {'source_sha256':source_hashes(), 'python':sys.version,
                'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','open3d','scikit-learn')},
                'seed_policy':'CRC32 of canonical POSIX test/filename; template seed from config',
                'registration':'fresh; not a replay of historical transforms',
                'input_protocol':manifest.get('input_protocol', 'legacy-pcd'),
                'center_inputs':manifest.get('center_inputs', False)})
        accepted = True
        budget = 7*GIB
        if available_memory() < 2*GIB:
            raise ValueError("Insufficient free memory")
        completed, pending = [], []
        for row in sorted(manifest['categories'], key=memory_estimate, reverse=True):
            if resume and (output/'results'/row['category']/'summary.json').exists():
                completed.append(verify_category(output, row))
            else:
                pending.append(row)
        logs = output/'logs'; logs.mkdir(exist_ok=True)
        failure = None
        while pending or processes:
            for name, item in list(processes.items()):
                process, row, stream = item
                if process.poll() is not None:
                    stream.close(); del processes[name]
                    if process.returncode:
                        failure = f"Category {name} failed ({process.returncode}); see logs/{name}.log"
                    else:
                        completed.append(verify_category(output, row))
            if failure:
                if not processes:
                    raise RuntimeError(failure)
            else:
                for row in list(pending):
                    used = sum(memory_estimate(item[1]) for item in processes.values())
                    need = memory_estimate(row)
                    if len(processes) >= workers:
                        break
                    if need > min(budget-used, available_memory()-768*1024**2):
                        continue
                    name = row['category']
                    stream = (logs/f'{name}.log').open('a' if resume else 'w', encoding='utf-8')
                    command = [sys.executable, '-u', '-m', 'georeg3dad', '_category',
                               '--run-root', str(output), '--category', name, '--threads', str(threads)]
                    try:
                        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                                   stdout=stream, stderr=subprocess.STDOUT,
                                                   start_new_session=os.name != 'nt')
                    except BaseException:
                        stream.close(); raise
                    processes[name] = (process, row, stream); pending.remove(row)
                if not processes and pending:
                    raise MemoryError("No pending category fits the memory budget; reduce workload or use a larger machine")
            status.update(state='RUNNING', active=list(processes), pending=[r['category'] for r in pending],
                          completed_categories=sorted(r['category'] for r in completed), updated_at=now())
            write_json(output/'status.json', status)
            if processes:
                time.sleep(1)
        if source_hashes() != read_json(output/'protocol.json')['source_sha256']:
            raise ValueError("Source changed during run")
        if sum(r['test_scans'] for r in completed) != manifest['test_scans']:
            raise ValueError("Final coverage mismatch")
        summarize(output, manifest, completed)
        write_json(output/'verification.json', {'status':'PASS','categories':len(completed),
            'test_scans':manifest['test_scans'],'point_valid_scans':manifest['point_valid_scans'],
            'checks':['input hashes','source/config hashes','saved arrays and labels','category coverage','output hashes'],
            'limit':'Exact chunked point ranking metrics are tested against sklearn; object metrics use sklearn.'})
        status.update(state='COMPLETE',finished_at=now(),active=[],pending=[])
        write_json(output/'status.json', status)
        print('COMPLETE', str(output), flush=True)
    except BaseException:
        if accepted:
            status.update(state='FAILED',error=traceback.format_exc(),updated_at=now())
            write_json(output/'status.json', status)
        raise
    finally:
        for process, _, stream in processes.values():
            stop_process_tree(process)
            stream.close()
        lock.unlink()
