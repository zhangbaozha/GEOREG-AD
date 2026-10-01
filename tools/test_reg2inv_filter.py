"""Transfer Reg2Inv's geometric filter and spatial mean-max to saved GeoReg scores.

This is a full-resolution postprocessing experiment, not Reg2Inv reproduction.
Original score arrays and point metrics remain unchanged. KNN includes self.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

import tune_object_aggregation as base  # Also fixes numerical thread limits.
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from georeg3dad.scoring import object_score

KS = (16, 32, 64, 128)
CHUNK = 2048
CENTROID_K = 64
UPSTREAM = '4dfb3470353cd239966c3522d6fc5dd57323f327'


def variants(fraction):
    result = [dict(id='old_top1', mode='top', fraction=.01, eligible=False),
              dict(id='current_top', mode='top', fraction=fraction, eligible=True),
              dict(id='raw_max', mode='max', eligible=False),
              dict(id='center64_max', mode='center_max', center_k=64, eligible=True),
              dict(id='center64_current_top', mode='center_top', center_k=64, fraction=fraction, eligible=True)]
    for k in KS:
        result += [dict(id=f'pool{k}', mode='mean_max', mean_k=k, eligible=True),
                   dict(id=f'center64_pool{k}', mode='center_mean_max', center_k=64, mean_k=k, eligible=True)]
    return result


def neighbor_pass(xyz, scores, center_filter=False, chunk=CHUNK):
    """Bounded-memory KNN pooling; optionally test equation (11) for every point."""
    count = len(xyz)
    if count < 1 or xyz.shape != (count, 3) or len(scores) != count:
        raise ValueError('One score per nonempty 3D point is required')
    if not np.isfinite(xyz).all() or not np.isfinite(scores).all():
        raise ValueError('Coordinates and scores must be finite')
    tree = cKDTree(xyz)
    max_k = min(max(KS), count)
    maxima = {k: -np.inf for k in KS}
    keep = np.zeros(count, dtype=bool) if center_filter else None
    for start in range(0, count, chunk):
        stop = min(start+chunk, count)
        _, ids = tree.query(xyz[start:stop], k=list(range(1, max_k+1)), workers=1)
        neighborhood_scores = scores[ids]
        for k in KS:
            maxima[k] = max(maxima[k], float(neighborhood_scores[:, :min(k, count)].mean(axis=1).max()))
        if center_filter:
            points = xyz[ids[:, :min(CENTROID_K, count)]]
            center = points.mean(axis=1)
            distances = ((points-center[:, None])**2).sum(axis=2)
            own_distance = ((xyz[start:stop]-center)**2).sum(axis=1)
            # Equation (11): retain all exact closest-to-centroid ties. The
            # released torch topk implementation can choose only one such tie.
            keep[start:stop] = own_distance <= distances.min(axis=1)
    return maxima, keep


def filtered_scores(xyz, scores, fraction):
    pooled, keep = neighbor_pass(xyz, scores, center_filter=True)
    retained = int(keep.sum())
    if retained == 0:
        raise ValueError('Centroid filter removed every point; do not silently drop the object')
    center_pooled, _ = neighbor_pass(xyz[keep], scores[keep])
    result = dict(old_top1=object_score(scores, .01), current_top=object_score(scores, fraction),
                  raw_max=float(scores.max()), center64_max=float(scores[keep].max()),
                  center64_current_top=object_score(scores[keep], fraction))
    for k in KS:
        result[f'pool{k}'] = pooled[k]
        result[f'center64_pool{k}'] = center_pooled[k]
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite object score')
    return result, dict(points=len(scores), retained=retained, retained_fraction=retained/len(scores),
                        center_k_effective=min(CENTROID_K, len(scores)),
                        pooling_clamped=retained < max(KS))


def sources(args):
    previous = args.previous_root.resolve()
    if base.read(previous/'status.json')['state'] != 'COMPLETE' or base.read(previous/'verification.json')['status'] != 'PASS':
        raise ValueError('Previous aggregation run must be COMPLETE/PASS')
    previous_protocol = base.read(previous/'protocol.json')
    if previous_protocol['dataset'] != args.dataset:
        raise ValueError('Previous dataset differs')
    split_paths = [Path(p) for p in previous_protocol['source_metadata_sha256']
                   if Path(p).name in ('splits.json', 'real3dad_splits.json')]
    if len(split_paths) != 1:
        raise ValueError('Ambiguous original grouped split')
    for path, digest in previous_protocol['source_metadata_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError(f'Previous source changed: {path}')
    fraction = base.read(previous/'selected.json')['top_fraction']
    inherited = SimpleNamespace(dataset=args.dataset, source_root=Path(previous_protocol['source_root']),
                                splits=split_paths[0], config_id='c_ff06afd13023', smoke=args.smoke)
    rows, config, metadata, splits = base.prepare(inherited)
    manifest = base.read(args.geometry_manifest)
    if manifest['dataset'] != args.dataset or manifest['smoke']:
        raise ValueError('Expected complete geometry manifest')
    geometry = {r['category']: r for r in manifest['categories']}
    previous_receipt = base.read(previous.parent/'receipt.json')['files_sha256']
    metadata.update({str(previous/name):base.sha(previous/name) for name in
                     ('summary.json','protocol.json','selected.json','status.json','verification.json')})
    metadata[str(args.geometry_manifest.resolve())] = base.sha(args.geometry_manifest)
    metadata[str(previous.parent/'receipt.json')] = base.sha(previous.parent/'receipt.json')
    for row in rows:
        g = geometry[row['category']]
        all_cases = {c['sample']: c for c in g['cases']}
        folder = args.data_root/g['source_category'] if args.dataset == 'real3dad' else args.data_root/g['source_split']/g['source_category']
        row['geometry'] = {}
        for c in row['cases']:
            original = all_cases[c['sample']]
            if bool(original['is_anomaly']) != bool(c['is_anomaly']):
                raise ValueError('Geometry and score object labels differ')
            filename = original['test'].replace('\\','/').rsplit('/',1)[-1]
            path = (folder/'test'/filename).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row['geometry'][c['sample']] = dict(path=str(path), sha256=g['input_sha256'][original['test']], points=original['points'])
        previous_case_path = previous/'categories'/row['category']/'objects.json'
        previous_summary_path = previous/'categories'/row['category']/'summary.json'
        for path in (previous_case_path, previous_summary_path):
            key = path.relative_to(previous.parent).as_posix()
            if base.sha(path) != previous_receipt[key]:
                raise ValueError(f'Previous artifact hash mismatch: {path}')
            metadata[str(path)] = base.sha(path)
        row['previous_cases'] = {c['sample']: c for c in base.read(previous_case_path)}
        row['previous_summary'] = base.read(previous_summary_path)
        row['previous_aggregation'] = base.read(previous/'selected.json')['aggregation']
        row['fraction'] = fraction
    return rows, config, metadata, splits, variants(fraction)


def worker(row, output, smoke):
    import open3d as o3d
    begin = time.perf_counter()
    output = Path(output)/'categories'/row['category']
    records, hashes = [], {}
    replay_error = 0.
    for case in row['cases']:
        sample = case['sample']
        relative = case['score_file'] if row['dataset'] == 'real3dad' else f'predictions/{sample}.npz'
        path = Path(row['folder'])/relative
        digest = base.sha(path)
        if digest != row['summary']['files_sha256'][relative]:
            raise ValueError(f'Point score hash mismatch: {path}')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as archive:
            point_scores = archive['scores' if row['dataset'] == 'real3dad' else row['config_id']]
        geo = row['geometry'][sample]
        digest = base.sha(geo['path'])
        if digest != geo['sha256']:
            raise ValueError(f'Geometry hash mismatch: {geo["path"]}')
        hashes[geo['path']] = digest
        cloud = o3d.io.read_point_cloud(geo['path'])
        xyz = np.asarray(cloud.points)
        if len(xyz) != len(point_scores) or len(xyz) != geo['points']:
            raise ValueError('Geometry and score lengths disagree')
        scores, diagnostic = filtered_scores(xyz, point_scores, row['fraction'])
        old = row['previous_cases'][sample]
        for key, oldkey in [('old_top1','top_0.01'), ('current_top',row['previous_aggregation']), ('raw_max','maximum')]:
            error = abs(scores[key]-old['scores'][oldkey])
            replay_error = max(replay_error, error)
            if error > 1e-12*max(1., abs(old['scores'][oldkey])):
                raise ValueError('Previous object score replay failed')
        if bool(old['is_anomaly']) != bool(case['is_anomaly']) or old['split'] != row['assignment'][sample]:
            raise ValueError('Previous labels or grouped partition changed')
        records.append(dict(sample=sample, is_anomaly=bool(case['is_anomaly']), split=row['assignment'][sample],
                            scores=scores, **diagnostic))
        base.write(output/'progress.json', dict(done=len(records), total=len(row['cases']), sample=sample,
                                              seconds=time.perf_counter()-begin))
    metrics = []
    for split in base.SPLITS:
        selected = [r for r in records if split == 'full' or r['split'] == split]
        labels = [int(c['is_anomaly']) for c in selected]
        if set(labels) != {0,1}:
            raise ValueError('Both object labels required')
        for variant in variants(row['fraction']):
            scores = [r['scores'][variant['id']] for r in selected]
            metrics.append(dict(category=row['category'], source_split=row['source_split'], split=split,
                variant=variant['id'], scans=len(selected), normal=labels.count(0), anomaly=labels.count(1),
                i_auroc=float(roc_auc_score(labels,scores)), i_ap=float(average_precision_score(labels,scores))))
    metric_error = None
    if not smoke:
        errors = []
        for split in base.SPLITS:
            for key, oldkey in [('old_top1','top_0.01'), ('current_top',row['previous_aggregation']), ('raw_max','maximum')]:
                old = next(m for m in row['previous_summary']['metrics'] if m['split']==split and m['aggregation']==oldkey)
                new = next(m for m in metrics if m['split']==split and m['variant']==key)
                errors.extend(abs(new[k]-old[k]) for k in ('i_auroc','i_ap'))
        metric_error = max(errors)
        if metric_error > 1e-12:
            raise ValueError('Previous metric replay failed')
    result = dict(category=row['category'], source_split=row['source_split'], scans=len(records), metrics=metrics,
        seconds=time.perf_counter()-begin, max_object_replay_error=replay_error, metric_replay_error=metric_error,
        retained_fraction=statistics.mean(r['retained_fraction'] for r in records),
        minimum_retained=min(r['retained'] for r in records),
        clamped_objects=sum(r['pooling_clamped'] for r in records),
        point_metrics_preserved={k:row['original_metrics'][k] for k in ('p_auroc','p_ap')})
    base.write(output/'objects.json', records)
    base.write(output/'input_sha256.json', hashes)
    base.write(output/'summary.json', result)
    return result


def aggregate(rows, dataset, settings):
    flat = [m for r in rows for m in r['metrics']]
    scopes = ('all',) if dataset=='real3dad' else ('all','official_pcd','new_pcd')
    result = []
    for scope in scopes:
        for split in base.SPLITS:
            for variant in settings:
                chosen = [r for r in flat if r['split']==split and r['variant']==variant['id']
                          and (scope=='all' or r['source_split']==('pcd' if scope=='official_pcd' else 'new_pcd'))]
                if chosen:
                    result.append(dict(scope=scope, split=split, variant=variant['id'], categories=len(chosen),
                        scans=sum(r['scans'] for r in chosen),
                        i_auroc=statistics.mean(r['i_auroc'] for r in chosen), i_ap=statistics.mean(r['i_ap'] for r in chosen)))
    return result


def choose(rows, dataset, settings):
    scope = 'all' if dataset=='real3dad' else 'official_pcd'
    selected = [r for r in rows if r['scope']==scope and r['split']=='validation']
    baseline = next(r for r in selected if r['variant']=='current_top')
    eligible = {s['id'] for s in settings if s['eligible']}
    best_auc = max(r['i_auroc'] for r in selected if r['variant'] in eligible)
    tied = [r for r in selected if r['variant'] in eligible and abs(r['i_auroc']-best_auc)<=1e-12]
    order = {s['id']: i for i,s in enumerate(settings)}
    winner = max(tied, key=lambda r:(r['i_ap'], r['variant']=='current_top', -order[r['variant']]))
    return winner if winner['i_auroc'] > baseline['i_auroc']+1e-4 else baseline


def self_test():
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(20260926)
    for n in (1, 13, 257):
        xyz = rng.normal(size=(n,3)); scores = rng.uniform(0.,10.,size=n)
        saved_xyz, saved_scores = xyz.copy(), scores.copy()
        result, diagnostic = filtered_scores(xyz,scores,.001)
        model = NearestNeighbors(n_neighbors=min(64,n)).fit(xyz)
        ids = model.kneighbors(xyz,return_distance=False)
        neighbors = xyz[ids]; center = neighbors.mean(axis=1)
        distance = ((neighbors-center[:,None])**2).sum(axis=2)
        mask = ((xyz-center)**2).sum(axis=1) <= distance.min(axis=1)
        assert diagnostic['retained']==int(mask.sum())
        for suffix, points, values in [('',xyz,scores), ('center64_',xyz[mask],scores[mask])]:
            for k in KS:
                ids = NearestNeighbors(n_neighbors=min(k,len(points))).fit(points).kneighbors(points,return_distance=False)
                reference = float(values[ids].mean(axis=1).max())
                np.testing.assert_allclose(result[f'{suffix}pool{k}'],reference,rtol=1e-13,atol=1e-13)
        permutation = rng.permutation(n)
        other, _ = filtered_scores(xyz[permutation],scores[permutation],.001)
        np.testing.assert_allclose(list(result.values()),list(other.values()),rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal(xyz,saved_xyz);np.testing.assert_array_equal(scores,saved_scores)
        # A rigid transform leaves spatial neighborhoods and aggregate scores invariant.
        matrix,_ = np.linalg.qr(rng.normal(size=(3,3)))
        rotated,_ = filtered_scores(xyz@matrix+np.array([2.,-4.,1.]),scores,.001)
        np.testing.assert_allclose(list(result.values()),list(rotated.values()),rtol=1e-13,atol=1e-13)
    duplicate = np.zeros((4,3))
    _,mask=neighbor_pass(duplicate,np.ones(4),center_filter=True)
    assert mask.all(), 'Equation (11) must retain exact centroid ties'
    print('SELF_TEST_PASS sklearn_reference=PASS permutation=PASS rigid_invariance=PASS ties=PASS')


def run(args):
    begin=time.perf_counter();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    status=dict(state='PREPARING',started_at=base.now(),dataset=args.dataset,workers=args.workers,smoke=args.smoke)
    base.write(output/'status.json',status)
    try:
        rows,config,metadata,splits,settings=sources(args)
        code_paths=[Path(__file__),Path(base.__file__),Path(base.__file__).resolve().parents[1]/'georeg3dad/scoring.py',
                    Path(base.__file__).resolve().parents[1]/'georeg3dad/config.py']
        code_hashes={str(p):base.sha(p) for p in code_paths}
        base.write(output/'splits.json',splits);base.write(output/'variants.json',settings)
        base.write(output/'source_config.json',config)
        protocol=dict(dataset=args.dataset,previous_root=str(args.previous_root.resolve()),input_metadata_sha256=metadata,
            source_sha256=code_hashes,variants=settings,upstream_commit=UPSTREAM,
            source_reference='Paper section 3.2 equations (10)-(11), appendix B.1; released predict_score uses center K=64, pooling K=64.',
            adaptation='Spatial postprocessing of GeoReg full-resolution interpolated scores. Paper uses downsampled learned-feature points. No Reg2Inv backbone, normalization, memory bank, retraining, or point metric change.',
            tie_handling='Retain all exact centroid-distance minima per paper equation; released torch topk selects one tied index. KNN includes self; k is min(requested, available points). No scan omitted.',
            selection='Macro validation I-AUROC, official40 for ShapeNet/all12 for Real3D. I-AP breaks ties within 1e-12, then incumbent, then declared variant order. Keep incumbent unless AUROC gain >1e-4. Full/remainder never select.',
            limitation='Previously observed benchmark splits; full and remainder are descriptive internal checks, not independent unseen tests.',
            execution=dict(device='CPU',workers=args.workers,threads_per_worker=1,chunk_points=CHUNK),smoke=args.smoke)
        base.write(output/'protocol.json',protocol)
        done=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,row,str(output),args.smoke):row['category'] for row in rows}
            status.update(state='RUNNING',total_categories=len(rows),completed_categories=[])
            base.write(output/'status.json',status)
            for future in as_completed(futures):
                result=future.result();done.append(result)
                status['completed_categories']=sorted(r['category'] for r in done)
                base.write(output/'status.json',status)
                print(f"DONE {result['category']} {len(done)}/{len(rows)}",flush=True)
        for path,digest in {**metadata,**code_hashes}.items():
            if base.sha(path)!=digest:raise ValueError(f'Source changed during computation: {path}')
        done.sort(key=lambda r:r['category'])
        aggregates=aggregate(done,args.dataset,settings)
        selected=None if args.smoke else choose(aggregates,args.dataset,settings)
        summary=dict(dataset=args.dataset,selected=selected,aggregates=aggregates,categories=[r['category'] for r in done],
            elapsed_seconds=time.perf_counter()-begin,smoke=args.smoke)
        base.write(output/'summary.json',summary)
        base.csv_write(output/'comparison.csv',aggregates)
        base.csv_write(output/'per_category.csv',[m for r in done for m in r['metrics']])
        if selected:
            base.write(output/'selected.json',selected)
            base.write(output/f'object_filter_{args.dataset}.json',dict(version=1,dataset=args.dataset,
                description='Postprocessing configuration for tools/test_reg2inv_filter.py; not a core Method config.',
                point_method=config['method'],object_filter=next(v for v in settings if v['id']==selected['variant'])))
        verification=dict(status='PASS',categories=len(done),scans=sum(r['scans'] for r in done),variants=len(settings),
            max_object_replay_error=max(r['max_object_replay_error'] for r in done),
            max_metric_replay_error=None if args.smoke else max(r['metric_replay_error'] for r in done),
            minimum_retained_points=min(r['minimum_retained'] for r in done),
            clamped_objects=sum(r['clamped_objects'] for r in done),
            checks=['every geometry and prediction SHA256','previous current/top1/max scores and all partition metric replay',
                    'complete grouped split and category coverage','all object-only scans included','frozen source/metadata',
                    'validation-only selection','finite scores and nonempty centroid masks'])
        base.write(output/'verification.json',verification)
        status.update(state='COMPLETE',finished_at=base.now(),elapsed_seconds=summary['elapsed_seconds'])
        base.write(output/'status.json',status)
        print(json.dumps(dict(state='COMPLETE',verification=verification,selected=selected,seconds=summary['elapsed_seconds'])),flush=True)
    except BaseException:
        status.update(state='FAILED',error=traceback.format_exc(),failed_at=base.now())
        base.write(output/'status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('self-test')
    p=sub.add_parser('run')
    p.add_argument('--dataset',choices=('real3dad','shapenet'),required=True)
    for name in ('previous-root','geometry-manifest','data-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    self_test() if a.command=='self-test' else run(a)


if __name__=='__main__':
    main()
