"""Fixed-parameter Real3D TFCE pilot on frozen poses; no result-based selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from georeg3dad.runtime import configure_threads, read_json, write_json, sha256
configure_threads(2)
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score, average_precision_score
from georeg3dad.config import method_from_dict
from georeg3dad.geometry import transform_xyz
from georeg3dad.scoring import residuals, anchor_scores, interpolate, object_score
from georeg3dad.metrics import point_metrics
from tfce_fixed import voxel_mapping, voxel_edges, enhance, self_test

VARIANTS = ['knn128_p0','knn3_p1','voxel_raw','voxel_tfce']


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def hashes():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__),Path(__file__).with_name('tfce_fixed.py')] + list((root/'georeg3dad').glob('*.py'))
    return {str(p):sha256(p) for p in files}


def verify(entries):
    for path, digest in entries.items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def select(cases, category, number):
    """Stratified hash selection, one scan per original numeric object id."""
    selected = []
    used = set()
    for anomaly in (False,True):
        candidates = [c for c in cases if bool(c['is_anomaly']) == anomaly]
        def key(c):
            return hashlib.sha256(f'TFCE-fixed-20260928|{category}|{c["sample"]}'.encode()).hexdigest()
        count = 0
        for case in sorted(candidates,key=key):
            group = case['sample'].split('_')[0]
            if group in used:
                continue
            used.add(group); selected.append(case); count += 1
            if count == number:
                break
        if count != number:
            raise ValueError(f'Insufficient independent groups: {category}')
    return selected


def plan(args):
    output, source = args.output.resolve(), args.source.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Plan requires new empty output directory')
    config = read_json(source/'config.json')
    manifest = read_json(source/'dataset.json')
    if config['dataset'] != 'real3dad' or read_json(source/'status.json')['state'] != 'COMPLETE':
        raise ValueError('Expected completed Real3D source run')
    if read_json(source/'verification.json')['status'] != 'PASS':
        raise ValueError('Source must have passed its checks')
    method = method_from_dict(config['method'])
    if method.interpolation.k != 128 or method.interpolation.power != 0:
        raise ValueError('Expected archived k128,p0 control')
    metadata = {str(source/name):sha256(source/name) for name in
                ('config.json','dataset.json','status.json','verification.json')}
    rows = []
    categories = manifest['categories'][:1] if args.smoke else manifest['categories']
    for original in categories:
        folder = source/'results'/original['category']
        summary = read_json(folder/'summary.json')
        for name in ('templates.json','cases.json'):
            if sha256(folder/name) != summary['files_sha256'][name]:
                raise ValueError('Original metadata hash differs')
            metadata[str(folder/name)] = sha256(folder/name)
        metadata[str(folder/'summary.json')] = sha256(folder/'summary.json')
        records = {c['sample']:c for c in read_json(folder/'cases.json')}
        cases = select(original['cases'],original['category'],1 if args.smoke else 5)
        selected_hashes = {p:original['input_sha256'][p] for p in original['templates']}
        for c in cases:
            selected_hashes[c['test']] = original['input_sha256'][c['test']]
            if c['gt']:
                selected_hashes[c['gt']] = original['input_sha256'][c['gt']]
            relative = records[c['sample']]['score_file']
            selected_hashes[str(folder/relative)] = summary['files_sha256'][relative]
        rows.append(dict(category=original['category'],cases=cases,folder=str(folder),
                         records={c['sample']:records[c['sample']] for c in cases},
                         templates=read_json(folder/'templates.json'),input_sha256=selected_hashes))
    spec = dict(created_at=now(),source=str(source),source_metadata_sha256=metadata,
        code_sha256=hashes(),dataset='real3dad',smoke=args.smoke,rows=rows,method=config['method'],
        variants=VARIANTS,tfce=dict(E=.5,H=2.,connectivity=26,extent='occupied voxel count',
            integration='exact continuous integral via merge tree',presmoothing=False,normalization='none'),
        mapping='Original Open3D per-scan voxel grid min(full_xyz)-voxel/2, before registration; each raw point inherits its voxel score.',
        selection='No winner selection or hyperparameter search. All four arms reported. Hash-stratified 5 normal + 5 anomalous scans per class, unique numeric object groups; smoke uses 1+1 in first class.',
        primary='Class-macro pooled point AUROC and AP; same complete-resolution scans in all arms.',
        secondary='Object AUROC/AP with source archived top_fraction=0.01, identical across arms.',
        limitation='Exploratory pilot on previously observed public test benchmark, conditional on previously tuned raw geometry. Not clean unseen-test evidence; k3 arm is NOT the paper baseline.',
        references=['https://pubmed.ncbi.nlm.nih.gov/18501637/','https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/randomise.html'],
        workers=2,threads_per_worker=2,replay_tolerance=1e-10)
    verify(metadata)
    for row in rows:
        verify(row['input_sha256'])
    write_json(output/'PLAN.json',spec)
    write_json(output/'status.json',dict(state='PLANNED',created_at=now(),categories=len(rows),scans=sum(len(r['cases']) for r in rows)))
    print(f'PLAN_FROZEN categories={len(rows)} scans={sum(len(r["cases"]) for r in rows)} sha256={sha256(output/"PLAN.json")}',flush=True)


def frozen_library(row, cfg):
    import open3d as o3d
    points, normals = [], []
    for record in row['templates']['registrations']:
        cloud = o3d.io.read_point_cloud(record['template']).voxel_down_sample(cfg.features.voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.features.normal_radius*cfg.features.voxel,max_nn=cfg.features.normal_neighbors))
        xyz = np.asarray(cloud.points); pose = np.asarray(record['transformation'])
        if record.get('reference'):
            h = float(np.median(cKDTree(xyz).query(xyz,k=2,workers=2)[0][:,1]))
            if abs(h-row['templates']['h']) > 1e-12:
                raise ValueError('Template spacing replay mismatch')
        points.append(transform_xyz(xyz,pose)); normals.append(np.asarray(cloud.normals)@pose[:3,:3].T)
    points, normals = np.concatenate(points),np.concatenate(normals)
    if len(points) != row['templates']['points']:
        raise ValueError('Template count replay mismatch')
    return points,normals,row['templates']['h']


def category_worker(row, method, destination):
    import open3d as o3d
    started = time.perf_counter()
    cfg = method_from_dict(method)
    output = Path(destination)/'categories'/row['category']; output.mkdir(parents=True,exist_ok=True)
    verify(row['input_sha256'])
    library = frozen_library(row,cfg)
    total = sum(c['points'] for c in row['cases'] if c['point_gt_valid'])
    y = np.lib.format.open_memmap(output/'point_labels.npy',mode='w+',dtype=np.int8,shape=(total,))
    scoremap = np.lib.format.open_memmap(output/'point_scores.npy',mode='w+',dtype=np.float64,shape=(len(VARIANTS),total))
    cursor, max_replay = 0, 0.
    records = []
    for case in row['cases']:
        tick = time.perf_counter(); sample = case['sample']; previous = row['records'][sample]
        cloud = o3d.io.read_point_cloud(case['test'])
        full_xyz = np.asarray(cloud.points)
        down = cloud.voxel_down_sample(cfg.features.voxel)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.features.normal_radius*cfg.features.voxel,max_nn=cfg.features.normal_neighbors))
        anchor_xyz = np.asarray(down.points)
        pose = np.asarray(previous['transformation'])
        query = transform_xyz(anchor_xyz,pose)
        full_registered = transform_xyz(full_xyz,pose)
        normals = np.asarray(down.normals)@pose[:3,:3].T
        if len(full_xyz)!=case['points'] or len(anchor_xyz)!=previous['anchor_points']:
            raise ValueError('Point count replay mismatch')
        parts = residuals(query,normals,*library,cfg.matching,threads=2)
        raw,unmatched = anchor_scores(parts,cfg.matching)
        del parts
        grid,mapping = voxel_mapping(full_xyz,anchor_xyz,cfg.features.voxel)
        edges_a,edges_b = voxel_edges(grid)
        enhanced = enhance(raw,edges_a,edges_b)
        predictions = dict(
            knn128_p0=interpolate(full_registered,query,raw,cfg.interpolation,threads=2),
            knn3_p1=interpolate(full_registered,query,raw,replace(cfg.interpolation,k=3,power=1.),threads=2),
            voxel_raw=raw[mapping],voxel_tfce=enhanced[mapping])
        # GT enters only after all predictions have been generated.
        with np.load(Path(row['folder'])/previous['score_file'],allow_pickle=False) as archive:
            reference = archive['scores']; labels = archive['labels']
        replay = float(np.max(np.abs(reference-predictions['knn128_p0'])))
        max_replay = max(max_replay,replay)
        if replay > 1e-10:
            raise ValueError(f'Prediction replay failed: {sample}: {replay}')
        if len(labels) != (len(full_xyz) if case['point_gt_valid'] else 0):
            raise ValueError('Invalid label coverage')
        metrics, object_scores = {}, {}
        for index,name in enumerate(VARIANTS):
            values = predictions[name]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite scores')
            object_scores[name] = object_score(values,cfg.object_top_fraction)
            if len(labels):
                scoremap[index,cursor:cursor+len(labels)] = values
                if np.any(labels) and np.any(labels==0):
                    metrics[name] = point_metrics(labels,values)
        if len(labels):
            y[cursor:cursor+len(labels)] = labels
            cursor += len(labels)
        n = len(raw)
        graph = csr_matrix((np.ones(2*len(edges_a)),(np.r_[edges_a,edges_b],np.r_[edges_b,edges_a])),shape=(n,n))
        nc, cc = connected_components(graph,directed=False)
        diagnostic = dict(components=int(nc),largest_component=int(np.bincount(cc).max()),
            isolated_voxels=int(np.count_nonzero(np.diff(graph.indptr)==0)),edges=len(edges_a),
            raw_median=float(np.median(raw)),raw_max=float(raw.max()),tfce_median=float(np.median(enhanced)),tfce_max=float(enhanced.max()))
        np.savez_compressed(output/f'{sample}_anchors.npz',grid=grid,raw=raw,tfce=enhanced)
        records.append(dict(sample=sample,is_anomaly=case['is_anomaly'],point_gt_valid=case['point_gt_valid'],
            points=len(full_xyz),anchors=n,positive_points=int(labels.sum()),scores=object_scores,
            scan_point_metrics=metrics,diagnostic=diagnostic,replay_error=replay,
            unmatched_fraction=float(unmatched.mean()),seconds=time.perf_counter()-tick))
        write_json(output/'objects.json',records)
        write_json(output/'progress.json',dict(phase='predict',done=len(records),total=len(row['cases']),seconds=time.perf_counter()-started))
        print(f'SCAN {row["category"]} {len(records)}/{len(row["cases"])} {sample} seconds={records[-1]["seconds"]:.2f}',flush=True)
    if cursor != total:
        raise ValueError('Incomplete pooled point arrays')
    y.flush(); scoremap.flush()
    labels_object = [int(r['is_anomaly']) for r in records]
    metrics = []
    for index,name in enumerate(VARIANTS):
        write_json(output/'progress.json',dict(phase='metrics',variant=name,done=len(records),total=len(records)))
        obj = [r['scores'][name] for r in records]
        metrics.append(dict(variant=name,**point_metrics(y,scoremap[index]),
            i_auroc=float(roc_auc_score(labels_object,obj)),i_ap=float(average_precision_score(labels_object,obj))))
    del y,scoremap
    verify(row['input_sha256'])
    result = dict(category=row['category'],scans=len(records),point_valid_scans=sum(r['point_gt_valid'] for r in records),
        points=total,positive_points=sum(r['positive_points'] for r in records),metrics=metrics,
        max_replay_error=max_replay,seconds=time.perf_counter()-started)
    write_json(output/'summary.json',result)
    write_json(output/'progress.json',dict(phase='COMPLETE',done=len(records),total=len(records)))
    return result


def report(output, summaries, elapsed):
    aggregates = []
    for name in VARIANTS:
        rows = [next(m for m in s['metrics'] if m['variant']==name) for s in summaries]
        aggregates.append(dict(variant=name,**{k:statistics.mean(r[k] for r in rows) for k in ('p_auroc','p_ap','i_auroc','i_ap')}))
    summary = dict(categories=len(summaries),scans=sum(r['scans'] for r in summaries),
        points=sum(r['points'] for r in summaries),aggregates=aggregates,categories_detail=summaries,elapsed_seconds=elapsed)
    write_json(output/'summary.json',summary)
    lines = ['# 固定默认 TFCE 替代 kNN 平滑：Real3D 先导实验','',
        f'完成 {len(summaries)} 类、{summary["scans"]} 个完整分辨率扫描、{summary["points"]:,} 个有效评估点。',
        'TFCE 固定 E=0.5、H=2、26 邻接；无参数搜索、无结果筛选。全部使用相同原始几何分数和历史配准。',
        '','|方法|P-AUROC|P-AP|I-AUROC|I-AP|','|---|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append('|'+row['variant']+'|'+'|'.join(f'{row[k]:.6f}' for k in ('p_auroc','p_ap','i_auroc','i_ap'))+'|')
    lines += ['','指标先在每类内汇总，再对类别宏平均。物体分数固定为归档配置的 Top 1%。',
        'knn3_p1 仅改变插值，不能称为论文原始基线。voxel_raw 用于隔离取消插值的影响；voxel_tfce 为唯一 TFCE 设置。',
        '', '|类别|k128 P-AUC|TFCE P-AUC|差值|k128 P-AP|TFCE P-AP|差值|','|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        a = next(m for m in row['metrics'] if m['variant']=='knn128_p0')
        b = next(m for m in row['metrics'] if m['variant']=='voxel_tfce')
        lines.append(f'|{row["category"]}|{a["p_auroc"]:.6f}|{b["p_auroc"]:.6f}|{b["p_auroc"]-a["p_auroc"]:+.6f}|{a["p_ap"]:.6f}|{b["p_ap"]:.6f}|{b["p_ap"]-a["p_ap"]:+.6f}|')
    lines += ['','限制：已观察的公开测试集上的固定方案先导对照，底层几何参数继承此前调参；不是独立未见测试，不能追溯声称整个方法零调参。',
        '体素连通性沿用原始扫描的体素网格，分辨率受 voxel 约束；不做预平滑或逐扫描归一化；输出不是显著性 p 值。',
        f'运行时间：{elapsed:.1f} 秒。原 k128 逐点重放最大误差：{max(r["max_replay_error"] for r in summaries):.3g}。',
        '计划、代码哈希、固定样本、源文件哈希见 PLAN.json；每类完整分数保存在 point_scores.npy，行序为 PLAN.json 的 variants，扫描顺序为 rows/cases。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(args):
    output = args.output.resolve(); spec = read_json(output/'PLAN.json'); started = time.perf_counter()
    if read_json(output/'status.json')['state'] != 'PLANNED':
        raise ValueError('Run requires a fresh frozen plan')
    plan_hash = sha256(output/'PLAN.json')
    status = dict(state='RUNNING',started_at=now(),plan_sha256=plan_hash,completed_categories=[])
    write_json(output/'status.json',status)
    try:
        verify(spec['code_sha256']); verify(spec['source_metadata_sha256'])
        done = []
        with ProcessPoolExecutor(max_workers=spec['workers']) as pool:
            futures = [pool.submit(category_worker,row,spec['method'],str(output)) for row in spec['rows']]
            for future in as_completed(futures):
                result = future.result(); done.append(result)
                status['completed_categories'] = sorted(r['category'] for r in done)
                write_json(output/'status.json',status)
                print(f'CATEGORY_COMPLETE {result["category"]} {len(done)}/{len(spec["rows"])}',flush=True)
        verify(spec['code_sha256']); verify(spec['source_metadata_sha256'])
        if sha256(output/'PLAN.json') != plan_hash:
            raise ValueError('Plan changed during execution')
        done.sort(key=lambda r:r['category'])
        summary = report(output,done,time.perf_counter()-started)
        write_json(output/'verification.json',dict(status='PASS',categories=len(done),scans=summary['scans'],
            max_replay_error=max(r['max_replay_error'] for r in done),plan_sha256=plan_hash,
            checks=['frozen code and input hashes','original k128 full-point replay','complete predeclared scan/point coverage','labels used after predictions','all fixed arms retained']))
        status.update(state='COMPLETE',finished_at=now(),elapsed_seconds=summary['elapsed_seconds'])
        write_json(output/'status.json',status)
        print('TFCE_EXPERIMENT_COMPLETE '+json.dumps(summary['aggregates']),flush=True)
    except BaseException:
        status.update(state='FAILED',error=traceback.format_exc()); write_json(output/'status.json',status); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    p = sub.add_parser('plan'); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke',action='store_true')
    p = sub.add_parser('run'); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'self-test': self_test()
    elif args.command == 'plan': plan(args)
    else: run(args)


if __name__ == '__main__':
    main()
