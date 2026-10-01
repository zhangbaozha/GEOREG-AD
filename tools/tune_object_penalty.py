"""Frozen-pose penalty search, selected only by grouped object validation AUROC.

Neighbor queries are shared; every penalty uses the core arithmetic directly.
Point metrics are pooled by category, with temporary arrays deleted on success.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
from types import SimpleNamespace

import tune_object_aggregation as base
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from georeg3dad.config import method_from_dict, load_config
from georeg3dad.geometry import transform_xyz
from georeg3dad.scoring import residuals, anchor_scores, interpolate, object_score
from georeg3dad.metrics import point_metrics

PENALTIES = {'real3dad': [.125,.25,.5,.75,1.,1.5,2.,4.,8.,16.],
             'shapenet': [.125,.25,.5,1.,2.,4.,8.,16.,32.,64.]}
POINT_TOLERANCES = {'real3dad': {'p_auroc': 1e-10, 'p_ap': 1e-7},
                    'shapenet': {'p_auroc': 1e-12, 'p_ap': 1e-12}}


def key(p):
    return f'penalty_{p:g}'


def code_hashes():
    code = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(base.__file__)] + sorted((code/'georeg3dad').glob('*.py'))
    return {str(p): base.sha(p) for p in files}


def check_hashes(entries):
    for path, digest in entries.items():
        if base.sha(path) != digest:
            raise ValueError(f'Input changed: {path}')


def prepare(args):
    previous = args.previous_root.resolve()
    if base.read(previous/'status.json')['state'] != 'COMPLETE' or base.read(previous/'verification.json')['status'] != 'PASS':
        raise ValueError('Previous object run must be COMPLETE/PASS')
    protocol = base.read(previous/'protocol.json')
    if protocol['dataset'] != args.dataset:
        raise ValueError('Dataset mismatch')
    check_hashes(protocol['source_metadata_sha256'])
    source = Path(protocol['source_root'])
    original_split = source.parent/'real3dad_splits.json' if args.dataset=='real3dad' else source/'splits.json'
    rows, oldconfig, metadata, splits = base.prepare(SimpleNamespace(dataset=args.dataset, source_root=source,
        splits=original_split, config_id='c_ff06afd13023', smoke=args.smoke))
    config = base.read(previous/f'config_{args.dataset}_object.json')
    method = method_from_dict(config['method'])
    comparable = deepcopy(config['method']); comparable['object_top_fraction'] = oldconfig['method']['object_top_fraction']
    if comparable != oldconfig['method'] or base.read(previous/'splits.json') != splits:
        raise ValueError('Previous method or grouped partitions changed')
    if method.object_top_fraction != base.read(previous/'selected.json')['top_fraction']:
        raise ValueError('Previous selected aggregation mismatch')
    if method.matching.unmatched_penalty not in PENALTIES[args.dataset]:
        raise ValueError('Incumbent absent from grid')
    geometry_root = args.geometry_root.resolve()
    manifest = base.read(geometry_root/'dataset.json')
    geometry_config = base.read(geometry_root/'config.json')['method']
    if any(geometry_config[s] != config['method'][s] for s in ('features','registration','templates')):
        raise ValueError('Frozen geometry configuration differs')
    if manifest['dataset'] != args.dataset or manifest['smoke']:
        raise ValueError('Full matching geometry manifest required')
    geometry = {r['category']:r for r in manifest['categories']}
    for filename in ('protocol.json','summary.json','selected.json','splits.json','status.json','verification.json',f'config_{args.dataset}_object.json'):
        metadata[str(previous/filename)] = base.sha(previous/filename)
    metadata[str(geometry_root/'dataset.json')] = base.sha(geometry_root/'dataset.json')
    metadata[str(geometry_root/'config.json')] = base.sha(geometry_root/'config.json')
    receipt_path = previous.parent/'receipt.json'
    receipt = base.read(receipt_path)
    receipt = receipt.get('files_sha256', receipt)
    metadata[str(receipt_path)] = base.sha(receipt_path)
    for path in list(metadata):
        local = Path(path)
        if local.is_relative_to(previous):
            if base.sha(local) != receipt[local.relative_to(previous.parent).as_posix()]:
                raise ValueError('Previous aggregation artifact receipt mismatch')
    for row in rows:
        g = geometry[row['category']]
        folder = args.data_root/g['source_category'] if args.dataset=='real3dad' else args.data_root/g['source_split']/g['source_category']
        def mapped(original):
            parts = original.replace('\\','/').split('/')
            return str((folder/parts[-2]/parts[-1]).resolve())
        row['geometry'] = {c['sample']:c for c in g['cases']}
        row['raw_paths'] = {s:mapped(s) for s in g['input_sha256']}
        selected_tests = {row['geometry'][c['sample']]['test'] for c in row['cases']}
        row['raw_hashes'] = {mapped(s):h for s,h in g['input_sha256'].items() if s in g['templates'] or s in selected_tests}
        directory = geometry_root/'results'/row['category']
        geometry_summary = base.read(directory/'summary.json')
        for filename in ('templates.json','cases.json'):
            if base.sha(directory/filename) != geometry_summary['files_sha256'][filename]:
                raise ValueError('Frozen pose metadata hash mismatch')
            metadata[str(directory/filename)] = base.sha(directory/filename)
        metadata[str(directory/'summary.json')] = base.sha(directory/'summary.json')
        row['templates'] = base.read(directory/'templates.json')
        row['records'] = {c['sample']:c for c in base.read(directory/'cases.json')}
        row['method'] = config['method']
        row['penalties'] = PENALTIES[args.dataset]
        row['previous_objects'] = {c['sample']:c for c in base.read(previous/'categories'/row['category']/'objects.json')}
        row['previous_summary'] = base.read(previous/'categories'/row['category']/'summary.json')
        row['previous_aggregation'] = base.read(previous/'selected.json')['aggregation']
        for filename in ('objects.json','summary.json'):
            path = previous/'categories'/row['category']/filename
            if base.sha(path) != receipt[path.relative_to(previous.parent).as_posix()]:
                raise ValueError('Previous category artifact receipt mismatch')
            metadata[str(path)] = base.sha(path)
    return rows, config, metadata, splits


def frozen_library(row):
    import open3d as o3d
    cfg = method_from_dict(row['method'])
    points, normals = [], []
    for r in row['templates']['registrations']:
        path = row['raw_paths'][r['template']]
        cloud = o3d.io.read_point_cloud(path).voxel_down_sample(cfg.features.voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.features.normal_radius*cfg.features.voxel, max_nn=cfg.features.normal_neighbors))
        xyz = np.asarray(cloud.points); matrix = np.asarray(r['transformation'])
        if r.get('reference'):
            h = float(np.median(cKDTree(xyz).query(xyz,k=2,workers=1)[0][:,1]))
            if abs(h-row['templates']['h']) > 1e-12:
                raise ValueError('Frozen template spacing mismatch')
        points.append(transform_xyz(xyz,matrix)); normals.append(np.asarray(cloud.normals)@matrix[:3,:3].T)
    xyz, normals = np.concatenate(points), np.concatenate(normals)
    if len(xyz) != row['templates']['points']:
        raise ValueError('Frozen template count mismatch')
    return xyz, normals, row['templates']['h']


def batch_interpolate(full, query, anchors, cfg, chunk=4096):
    """Same operations/order as core interpolate, with shared spatial queries."""
    tree = cKDTree(query); k = min(cfg.k,len(query))
    results = {p:np.empty(len(full),dtype=np.float64) for p in anchors}
    for start in range(0,len(full),chunk):
        stop = min(start+chunk,len(full))
        distance, ids = tree.query(full[start:stop],k=list(range(1,k+1)),workers=1)
        if cfg.power != 0:
            weights = 1/np.maximum(distance,cfg.epsilon) if cfg.power==1 else np.maximum(distance,cfg.epsilon)**(-cfg.power)
            weights /= weights.sum(axis=1,keepdims=True)
        for p, values in anchors.items():
            if cfg.power == 0:
                results[p][start:stop] = np.cumsum(values[ids],axis=1)[:,-1]/k
            else:
                results[p][start:stop] = (values[ids]*weights).sum(axis=1)
    return results


def predict(row, case, library):
    import open3d as o3d
    method = method_from_dict(row['method'])
    raw = row['geometry'][case['sample']]
    record = row['records'][case['sample']]
    cloud = o3d.io.read_point_cloud(row['raw_paths'][raw['test']])
    down = cloud.voxel_down_sample(method.features.voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=method.features.normal_radius*method.features.voxel,max_nn=method.features.normal_neighbors))
    transform = np.asarray(record['transformation'])
    full = transform_xyz(np.asarray(cloud.points),transform)
    query = transform_xyz(np.asarray(down.points),transform)
    normals = np.asarray(down.normals)@transform[:3,:3].T
    if len(full)!=raw['points'] or len(query)!=record['anchor_points']:
        raise ValueError('Frozen point/anchor count mismatch')
    components = residuals(query,normals,*library,method.matching,1)
    anchors = {}
    for penalty in row['penalties']:
        anchors[penalty], missing = anchor_scores(components,replace(method.matching,unmatched_penalty=penalty))
    return batch_interpolate(full,query,anchors,method.interpolation,method.interpolation.chunk_size), float(missing.mean())


def category_worker(row, output, smoke):
    started = time.perf_counter()
    output = Path(output)/'categories'/row['category']; output.mkdir(parents=True,exist_ok=True)
    check_hashes(row['raw_hashes'])
    library = frozen_library(row)
    cfg = method_from_dict(row['method']); incumbent = cfg.matching.unmatched_penalty
    penalties = row['penalties']
    total = sum(row['geometry'][c['sample']]['points'] for c in row['cases'] if row['geometry'][c['sample']]['point_gt_valid'])
    work = output/'work'; work.mkdir()
    y = np.lib.format.open_memmap(work/'labels.npy',mode='w+',dtype=np.int8,shape=(total,))
    maps = np.lib.format.open_memmap(work/'scores.npy',mode='w+',dtype=np.float64,shape=(len(penalties),total))
    cursor = 0; records = []; hashes = dict(row['raw_hashes'])
    max_point_error = max_object_error = max_old_object_error = 0.
    try:
        for case in row['cases']:
            sample = case['sample']; geo = row['geometry'][sample]
            relative = case['score_file'] if row['dataset']=='real3dad' else f'predictions/{sample}.npz'
            path = Path(row['folder'])/relative
            digest = base.sha(path)
            if digest != row['summary']['files_sha256'][relative]:
                raise ValueError('Archived prediction SHA256 mismatch')
            hashes[str(path)] = digest
            with np.load(path,allow_pickle=False) as old:
                reference = old['scores' if row['dataset']=='real3dad' else row['config_id']]
                labels = old['labels']
            predictions, unmatched = predict(row,case,library)
            if len(labels) != (len(reference) if geo['point_gt_valid'] else 0):
                raise ValueError('Point label coverage mismatch')
            if len(reference) != geo['points'] or any(len(s)!=len(reference) or not np.isfinite(s).all() for s in predictions.values()):
                raise ValueError('Nonfinite or incomplete prediction')
            error = float(np.max(np.abs(predictions[incumbent]-reference)))
            max_point_error = max(max_point_error,error)
            if error > 1e-10:
                raise ValueError(f'Frozen point replay mismatch {sample}: {error}')
            scores = {key(p):object_score(s,cfg.object_top_fraction) for p,s in predictions.items()}
            previous = row['previous_objects'][sample]
            if previous['split'] != row['assignment'][sample] or previous['is_anomaly'] != bool(case['is_anomaly']):
                raise ValueError('Object labels/partitions changed')
            oldscore = previous['scores'][row['previous_aggregation']]
            objerror = abs(scores[key(incumbent)]-oldscore)
            max_object_error = max(max_object_error,objerror)
            original_score = case['object_score'] if row['dataset']=='real3dad' else case['scores'][row['config_id']]
            olderror = abs(object_score(predictions[incumbent],.01)-original_score)
            max_old_object_error = max(max_old_object_error,olderror)
            if objerror > 1e-10*max(1.,abs(oldscore)) or olderror > 1e-10*max(1.,abs(original_score)):
                raise ValueError('Object score replay mismatch')
            if len(labels):
                y[cursor:cursor+len(labels)] = labels
                for i,p in enumerate(penalties):
                    maps[i,cursor:cursor+len(labels)] = predictions[p]
                cursor += len(labels)
            records.append(dict(sample=sample,is_anomaly=bool(case['is_anomaly']),split=row['assignment'][sample],
                points=len(reference),point_gt_valid=geo['point_gt_valid'],unmatched_fraction=unmatched,scores=scores))
            base.write(output/'progress.json',dict(phase='predict',done=len(records),total=len(row['cases']),sample=sample,seconds=time.perf_counter()-started))
        if cursor != total:
            raise ValueError('Pooled point coverage mismatch')
        y.flush(); maps.flush()
        metrics = []
        for split in base.SPLITS:
            cases = [r for r in records if split=='full' or r['split']==split]
            labels = [int(r['is_anomaly']) for r in cases]
            for p in penalties:
                values = [r['scores'][key(p)] for r in cases]
                metrics.append(dict(category=row['category'],source_split=row['source_split'],split=split,penalty=p,
                    scans=len(cases),normal=labels.count(0),anomaly=labels.count(1),
                    i_auroc=float(roc_auc_score(labels,values)),i_ap=float(average_precision_score(labels,values))))
        point = []
        for i,p in enumerate(penalties):
            base.write(output/'progress.json',dict(phase='point_metrics',done=len(records),total=len(row['cases']),penalty=p,seconds=time.perf_counter()-started))
            point.append(dict(penalty=p,points=total,**point_metrics(y,maps[i])))
        metric_error = point_metric_error = point_metric_errors = None
        if not smoke:
            metric_error = max(abs(next(m for m in metrics if m['split']==split and m['penalty']==incumbent)[k]-
                next(m for m in row['previous_summary']['metrics'] if m['split']==split and m['aggregation']==row['previous_aggregation'])[k])
                for split in base.SPLITS for k in ('i_auroc','i_ap'))
            point_control = next(m for m in point if m['penalty']==incumbent)
            point_metric_errors = {k: abs(point_control[k]-row['original_metrics'][k]) for k in ('p_auroc','p_ap')}
            point_metric_error = max(point_metric_errors.values())
            if metric_error > 1e-12 or any(point_metric_errors[k] > tolerance for k,tolerance in POINT_TOLERANCES[row['dataset']].items()):
                raise ValueError(f'Baseline metric replay mismatch: object={metric_error} point={point_metric_error}')
        result = dict(category=row['category'],source_split=row['source_split'],scans=len(records),
            point_valid_scans=sum(r['point_gt_valid'] for r in records),metrics=metrics,point_metrics=point,
            max_point_replay_error=max_point_error,max_object_replay_error=max_object_error,max_old_object_replay_error=max_old_object_error,
            metric_replay_error=metric_error,point_metric_replay_error=point_metric_error,
            point_metric_replay_errors=point_metric_errors,seconds=time.perf_counter()-started)
        base.write(output/'objects.json',records); base.write(output/'input_sha256.json',hashes)
        base.write(output/'summary.json',result)
    finally:
        y.flush(); maps.flush(); y._mmap.close(); maps._mmap.close()
    # Exact known files belonging only to this new category; keep scratch on failure.
    (work/'labels.npy').unlink(); (work/'scores.npy').unlink(); work.rmdir()
    return result


def aggregate(rows,dataset):
    result = []
    scopes = ['all'] if dataset=='real3dad' else ['all','official_pcd','new_pcd']
    for scope in scopes:
        cats = [c for c in rows if scope=='all' or c['source_split']==('pcd' if scope=='official_pcd' else 'new_pcd')]
        if not cats:
            continue
        for split in base.SPLITS:
            for p in PENALTIES[dataset]:
                selected = [next(m for m in c['metrics'] if m['split']==split and m['penalty']==p) for c in cats]
                row = dict(scope=scope,split=split,penalty=p,categories=len(cats),scans=sum(m['scans'] for m in selected),
                    **{k:statistics.mean(m[k] for m in selected) for k in ('i_auroc','i_ap')})
                if split=='full':
                    points = [next(m for m in c['point_metrics'] if m['penalty']==p) for c in cats]
                    row.update({k:statistics.mean(m[k] for m in points) for k in ('p_auroc','p_ap')})
                result.append(row)
    return result


def choose(aggregates,dataset,incumbent):
    scope = 'all' if dataset=='real3dad' else 'official_pcd'
    candidates = [m for m in aggregates if m['scope']==scope and m['split']=='validation']
    old = next(m for m in candidates if m['penalty']==incumbent)
    best_auc = max(m['i_auroc'] for m in candidates)
    best = max((m for m in candidates if abs(m['i_auroc']-best_auc)<=1e-12),
        key=lambda m:(m['i_ap'],m['penalty']==incumbent,-abs(m['penalty']-incumbent),-m['penalty']))
    return best if best['i_auroc']>old['i_auroc']+1e-4 else old


def self_test():
    rng = np.random.default_rng(20260926)
    query=rng.normal(size=(173,3)); full=rng.normal(size=(311,3))
    from georeg3dad.config import Interpolation, Matching
    components = dict(distance=np.array([[1.,2.],[np.inf,np.inf],[3.,4.]]),
        plane=np.array([[2.,1.],[1.,1.],[0.,1.]]),normal=np.zeros((3,2)),search_distance=np.array([[1.,2.],[np.inf,np.inf],[7.,8.]]))
    for p in PENALTIES['real3dad']:
        s,m=anchor_scores(components,Matching(candidate_k=2,radius_h=6.,plane_weight=1.,unmatched_penalty=p))
        np.testing.assert_array_equal(s,[3.,p,p]);np.testing.assert_array_equal(m,[False,True,True])
    anchors={p:rng.random(len(query)) for p in PENALTIES['shapenet']}
    for k,power in ((128,0.),(16,.5),(3,1.),(256,2.)):
        cfg=Interpolation(k=k,power=power)
        batched=batch_interpolate(full,query,anchors,cfg,chunk=53)
        for p,values in anchors.items():
            np.testing.assert_array_equal(batched[p],interpolate(full,query,values,cfg,1))
    print('OBJECT_PENALTY_SELF_TEST_PASS absolute_penalty=PASS shared_neighbors_exact=PASS',flush=True)


def run(args):
    start=time.perf_counter(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use new empty output directory')
    status=dict(state='PREPARING',dataset=args.dataset,started_at=base.now(),workers=args.workers,smoke=args.smoke)
    base.write(output/'status.json',status)
    try:
        rows,config,metadata,splits=prepare(args)
        cfg=method_from_dict(config['method']); codes=code_hashes()
        scratch=sum(sum(r['geometry'][c['sample']]['points'] for c in r['cases'] if r['geometry'][c['sample']]['point_gt_valid']) for r in rows)*(8*len(PENALTIES[args.dataset])+1)
        if shutil.disk_usage(output).free < scratch + 10*1024**3:
            raise ValueError('Insufficient disk for bounded transient score arrays')
        protocol=dict(dataset=args.dataset,previous_root=str(args.previous_root.resolve()),geometry_root=str(args.geometry_root.resolve()),
            data_root=str(args.data_root.resolve()),penalties=PENALTIES[args.dataset],incumbent=cfg.matching.unmatched_penalty,
            object_top_fraction=cfg.object_top_fraction,source_metadata_sha256=metadata,source_code_sha256=codes,
            selection='One global penalty; maximize grouped validation macro I-AUROC. Ties within 1e-12 use I-AP, incumbent, nearest incumbent, smaller penalty. Keep incumbent unless gain >1e-4. Real3D all12; ShapeNet official40. Full/remainder do not select.',
            scope='Previously observed benchmarks; not independent unseen test. Fixed original templates, poses and other method parameters.',
            numerical='Shared KNN queries; direct core anchor_scores and exact original interpolation arithmetic per penalty; no affine score approximation.',
            point_metrics='Report full pooled category P-AUROC/P-AP for every penalty; temporary memmaps removed after successful category.',
            point_metric_replay_tolerances=POINT_TOLERANCES[args.dataset],
            execution=dict(workers=args.workers,math_threads=1,device='CPU',maximum_total_scratch_bytes=scratch),smoke=args.smoke)
        base.write(output/'protocol.json',protocol);base.write(output/'splits.json',splits);base.write(output/'source_config.json',config)
        done=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(category_worker,row,str(output),args.smoke):row['category'] for row in rows}
            status.update(state='RUNNING',total_categories=len(rows),completed_categories=[]);base.write(output/'status.json',status)
            for future in as_completed(futures):
                r=future.result();done.append(r)
                status['completed_categories']=sorted(c['category'] for c in done);base.write(output/'status.json',status)
                print(f"DONE {r['category']} {len(done)}/{len(rows)}",flush=True)
        done.sort(key=lambda r:r['category']);check_hashes(metadata);check_hashes(codes)
        aggregates=aggregate(done,args.dataset);selected=None if args.smoke else choose(aggregates,args.dataset,cfg.matching.unmatched_penalty)
        summary=dict(dataset=args.dataset,selected=selected,aggregates=aggregates,categories=[r['category'] for r in done],
            elapsed_seconds=time.perf_counter()-start,smoke=args.smoke)
        base.write(output/'summary.json',summary)
        fields=('scope','split','penalty','categories','scans','i_auroc','i_ap','p_auroc','p_ap')
        base.csv_write(output/'comparison.csv',[{k:r.get(k) for k in fields} for r in aggregates])
        base.csv_write(output/'per_category.csv',[m for r in done for m in r['metrics']])
        if selected:
            exported=dict(dataset=args.dataset,description='Object AUROC validation-selected penalty and fixed Top fraction; frozen-pose search.',method=deepcopy(config['method']))
            exported['method']['matching']['unmatched_penalty']=selected['penalty']
            path=output/f'config_{args.dataset}_object.json';base.write(path,exported);load_config(path)
            base.write(output/'selected.json',selected)
        verification=dict(status='PASS',categories=len(done),scans=sum(r['scans'] for r in done),
            point_valid_scans=sum(r['point_valid_scans'] for r in done),penalties=len(PENALTIES[args.dataset]),
            max_point_replay_error=max(r['max_point_replay_error'] for r in done),
            max_object_replay_error=max(r['max_object_replay_error'] for r in done),
            max_old_object_replay_error=max(r['max_old_object_replay_error'] for r in done),
            max_metric_replay_error=None if args.smoke else max(r['metric_replay_error'] for r in done),
            max_point_metric_replay_error=None if args.smoke else max(r['point_metric_replay_error'] for r in done),
            max_point_metric_replay_errors=None if args.smoke else {k:max(r['point_metric_replay_errors'][k] for r in done) for k in ('p_auroc','p_ap')},
            point_metric_replay_tolerances=POINT_TOLERANCES[args.dataset],
            temporary_arrays_remaining=len(list(output.glob('categories/*/work/*.npy'))))
        if verification['temporary_arrays_remaining']:
            raise ValueError('Temporary arrays not cleaned up')
        base.write(output/'verification.json',verification)
        status.update(state='COMPLETE',finished_at=base.now(),elapsed_seconds=summary['elapsed_seconds']);base.write(output/'status.json',status)
        print(json.dumps(dict(verification=verification,selected=selected,seconds=summary['elapsed_seconds'])),flush=True)
    except BaseException:
        status.update(state='FAILED',failed_at=base.now(),error=traceback.format_exc());base.write(output/'status.json',status);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    p=sub.add_parser('run')
    p.add_argument('--dataset',choices=tuple(PENALTIES),required=True)
    for option in ('previous-root','geometry-root','data-root','output'):
        p.add_argument('--'+option,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    self_test() if args.command=='self-test' else run(args)


if __name__=='__main__':
    main()
