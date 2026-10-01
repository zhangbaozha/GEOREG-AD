"""Sequential ShapeNet scoring search using a verified full run's frozen poses.

Uses the shared method functions. No registration reruns, old-script imports,
or persistent neighbor caches. Selection uses official-40 development and
validation groups; final evaluation includes all 52 categories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from georeg3dad.config import method_from_dict
from georeg3dad.runtime import (available_memory, configure_threads, read_json,
    replace, sha256, source_hashes, stop_process_tree, write_json)
from georeg3dad.runner import check_hashes, now, verify_category

GIB = 1024**3
METRICS = ('p_auroc', 'p_ap', 'i_auroc', 'i_ap')
STAGES = ('interpolation', 'interpolation_refine', 'penalty_coarse', 'radius_coarse',
          'candidates', 'penalty_fine', 'radius_fine', 'plane_weight', 'normal_weight')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def setting(method):
    method_from_dict(method)
    return {'id': 'c_'+signature(method)[:12], 'method': method}


def unique(values):
    return list({c['id']: c for c in values}.values())


def variations(base, section, key, values):
    result = []
    for value in values:
        method = deepcopy(base['method']); method[section][key] = value
        result.append(setting(method))
    return result


def grid(stage, incumbent, baseline):
    values = [baseline, incumbent]
    if stage == 'interpolation':
        for k in (1, 3, 5, 8, 16, 32, 64, 128):
            for p in ((1.0,) if k == 1 else (0.0, .5, 1.0, 2.0)):
                m = deepcopy(incumbent['method']); m['interpolation'].update(k=k, power=p)
                values.append(setting(m))
    elif stage == 'interpolation_refine':
        ks = [1, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
        old = incumbent['method']['interpolation']; index = ks.index(old['k'])
        chosen = ks[max(0,index-1):min(len(ks),index+3)]
        if old['k'] == 128: chosen += [256]
        powers = sorted({0., 1., max(0., old['power']-.25), old['power'], old['power']+.25})
        for k in chosen:
            for p in powers:
                m=deepcopy(incumbent['method']); m['interpolation'].update(k=k,power=p)
                values.append(setting(m))
    else:
        fields = {
            'penalty_coarse': ('unmatched_penalty', [4.,8.,12.,16.]),
            'radius_coarse': ('radius_h', [4.,6.,8.,12.]),
            'candidates': ('candidate_k', [4,8,16,32]),
            'penalty_fine': ('unmatched_penalty', [.5,.75,1.,1.5,2.,3.,4.,6.,8.,12.,16.,24.,32.]),
            'radius_fine': ('radius_h', sorted({max(1.,incumbent['method']['matching']['radius_h']+d) for d in (-2,-1,0,1,2)})),
            'plane_weight': ('plane_weight', [0.,.25,.5,1.,2.]),
            'normal_weight': ('normal_weight', [0.,.1,.25,.5,1.]),
        }
        key, candidates = fields[stage]
        values += variations(incumbent, 'matching', key, candidates)
    return unique(values)


def make_splits(manifest):
    result = {}
    for row in manifest['categories']:
        groups = defaultdict(list)
        for case in row['cases']:
            match = re.search(r'(\d+)$', case['sample'])
            if match is None: raise ValueError('Cannot identify variant group: '+case['sample'])
            groups[match.group(1)].append(case)
        strata = defaultdict(list)
        for group,cases in groups.items():
            strata[(any(c['is_anomaly'] for c in cases), any(not c['is_anomaly'] for c in cases))].append(group)
        assignment = {}
        for members in strata.values():
            members.sort(key=lambda g: signature([20260924,row['category'],g]))
            count = max(1, len(members)//5) if len(members)>=3 else 0
            for index, group in enumerate(members):
                split = 'development' if index<count else 'validation' if index<2*count else 'remainder'
                for case in groups[group]: assignment[case['sample']] = split
        for split in ('development','validation','remainder'):
            cases = [c for c in row['cases'] if assignment[c['sample']]==split]
            if not any(not c['is_anomaly'] for c in cases) or not any(c['is_anomaly'] and c['point_gt_valid'] for c in cases):
                raise ValueError(f'Insufficient grouped labels: {row["category"]}/{split}')
        result[row['category']] = assignment
    return result


def all_sources():
    return {**{'core/'+k:v for k,v in source_hashes().items()}, 'tools/tune_shapenet.py':sha256(__file__)}


def checked_protocol(root):
    protocol = read_json(root/'protocol.json')
    if protocol['source_sha256'] != all_sources(): raise ValueError('Tuning source changed')
    if sha256(root/'splits.json') != protocol['splits_sha256']: raise ValueError('Split assignments changed')
    baseline = Path(protocol['baseline_root'])
    check_hashes({str(baseline/name):value for name,value in protocol['baseline_sha256'].items()})
    return protocol, baseline


def frozen_library(baseline, row, method):
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    from georeg3dad.geometry import transform_xyz
    saved = read_json(baseline/'results'/row['category']/'templates.json')
    points, normals = [], []
    for record in saved['registrations']:
        cloud=o3d.io.read_point_cloud(record['template']).voxel_down_sample(method.features.voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=method.features.normal_radius*method.features.voxel,max_nn=method.features.normal_neighbors))
        xyz=np.asarray(cloud.points); matrix=np.asarray(record['transformation'])
        if record.get('reference'):
            h=float(np.median(cKDTree(xyz).query(xyz,k=2)[0][:,1]))
            if abs(h-saved['h'])>1e-12: raise ValueError('Template spacing replay differs')
        points.append(transform_xyz(xyz,matrix)); normals.append(np.asarray(cloud.normals)@matrix[:3,:3].T)
    xyz,normals=np.concatenate(points),np.concatenate(normals)
    if len(xyz)!=saved['points']: raise ValueError('Template point order/count differs')
    return xyz,normals,saved['h']


def batch_interpolate(full, query, anchors, configs, threads):
    import numpy as np
    from scipy.spatial import cKDTree
    tree=cKDTree(query)
    results={c['id']:np.empty(len(full),dtype=np.float64) for c in configs}
    grouped=defaultdict(list)
    for c in configs: grouped[min(c['method']['interpolation']['k'],len(query))].append(c)
    for k, group in grouped.items():
        for start in range(0,len(full),4096):
            stop=min(start+4096,len(full))
            distance,ids=tree.query(full[start:stop],k=list(range(1,k+1)),workers=threads)
            for c in group:
                cfg=c['method']['interpolation']; values=anchors[c['id']][ids]
                if cfg['power']==0: scores=np.cumsum(values,axis=1)[:,-1]/k
                else:
                    weights=1/np.maximum(distance,cfg['epsilon']) if cfg['power']==1 else np.maximum(distance,cfg['epsilon'])**(-cfg['power'])
                    weights/=weights.sum(axis=1,keepdims=True); scores=(values*weights).sum(axis=1)
                results[c['id']][start:stop]=scores
    return results


def case_predictions(case, record, library, configs, baseline, category, threads):
    import numpy as np
    import open3d as o3d
    from georeg3dad.geometry import transform_xyz
    from georeg3dad.scoring import residuals, anchor_scores
    first=method_from_dict(configs[0]['method'])
    cloud=o3d.io.read_point_cloud(case['test']); down=cloud.voxel_down_sample(first.features.voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=first.features.normal_radius*first.features.voxel,max_nn=first.features.normal_neighbors))
    matrix=np.asarray(record['transformation'])
    query=transform_xyz(np.asarray(down.points),matrix)
    normals=np.asarray(down.normals)@matrix[:3,:3].T
    full=transform_xyz(np.asarray(cloud.points),matrix)
    if len(full)!=case['points'] or len(query)!=record['anchor_points']: raise ValueError('Scan replay count differs')
    cache,anchors,unmatched={},{},{}
    for c in configs:
        matching=method_from_dict(c['method']).matching
        key=(matching.candidate_k,matching.radius_h)
        if key not in cache:
            cache[key]=residuals(query,normals,*library,matching,threads)
        scores,missing=anchor_scores(cache[key],matching)
        anchors[c['id']]=scores; unmatched[c['id']]=float(missing.mean())
    predictions=batch_interpolate(full,query,anchors,configs,threads)
    with np.load(baseline/'results'/category/record['score_file'],allow_pickle=False) as old:
        labels=old['labels'].copy(); reference=old['scores']
        error=float(np.max(np.abs(predictions[configs[0]['id']]-reference)))
        if error>1e-10: raise ValueError(f'Frozen baseline parity failure {case["sample"]}: {error}')
    if len(labels)!=(len(full) if case['point_gt_valid'] else 0): raise ValueError('Label coverage differs')
    if any(not np.isfinite(s).all() for s in predictions.values()): raise ValueError('Nonfinite scores')
    return predictions,labels,error,unmatched


def selected_cases(root,row,split):
    assignment=read_json(root/'splits.json')[row['category']]
    return [c for c in row['cases'] if split=='full' or assignment[c['sample']]==split]


def category_stage(root,stage,split,category,threads):
    configure_threads(threads)
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score
    from georeg3dad.metrics import point_metrics
    from georeg3dad.scoring import object_score
    _,baseline=checked_protocol(root)
    manifest=read_json(baseline/'dataset.json')
    row=next(r for r in manifest['categories'] if r['category']==category)
    check_hashes(row['input_sha256']); verify_category(baseline,row)
    configs=read_json(root/stage/split/'configs.json')
    cases=selected_cases(root,row,split)
    records={r['sample']:r for r in read_json(baseline/'results'/category/'cases.json')}
    library=frozen_library(baseline,row,method_from_dict(configs[0]['method']))
    output=root/stage/split/category; output.mkdir(parents=True,exist_ok=True)
    work=output/'work'; work.mkdir(exist_ok=True)
    n=sum(c['points'] for c in cases if c['point_gt_valid'])
    y=np.lib.format.open_memmap(work/'labels.npy',mode='w+',dtype=np.int8,shape=(n,))
    scores=np.lib.format.open_memmap(work/'scores.npy',mode='w+',dtype=np.float64,shape=(len(configs),n))
    objects=[];cursor=0;max_error=0.;files={};started=time.perf_counter()
    try:
        for index,case in enumerate(cases,1):
            predictions,labels,error,missing=case_predictions(case,records[case['sample']],library,configs,baseline,category,threads)
            max_error=max(max_error,error)
            if len(labels): y[cursor:cursor+len(labels)]=labels
            obj={'sample':case['sample'],'is_anomaly':case['is_anomaly'],'scores':{},'unmatched_fraction':missing}
            for ci,c in enumerate(configs):
                values=predictions[c['id']]
                if len(labels):scores[ci,cursor:cursor+len(labels)]=values
                obj['scores'][c['id']]=object_score(values,c['method']['object_top_fraction'])
            if split=='full':
                path=output/'predictions'/f'{case["sample"]}.npz'; path.parent.mkdir(exist_ok=True)
                temp=path.with_name(path.stem+'.tmp.npz')
                np.savez_compressed(temp,labels=labels,**predictions);replace(temp,path)
                with np.load(path,allow_pickle=False) as saved:
                    if not np.array_equal(saved['labels'],labels) or any(not np.array_equal(saved[k],v) for k,v in predictions.items()):
                        raise ValueError('Saved prediction mismatch')
                files[path.relative_to(output).as_posix()]=sha256(path)
            objects.append(obj);cursor+=len(labels)
            write_json(output/'progress.json',{'phase':'predict','done':index,'total':len(cases),'sample':case['sample']})
        if cursor!=n:raise ValueError('Incomplete pooled points')
        y.flush();scores.flush();metrics=[]
        assignment=read_json(root/'splits.json')[category]
        remainder=np.concatenate([np.full(c['points'],assignment[c['sample']]=='remainder',dtype=bool)
            for c in cases if c['point_gt_valid']]) if split=='full' else None
        for ci,c in enumerate(configs):
            write_json(output/'progress.json',{'phase':'metrics','done':len(cases),'total':len(cases),'config':c['id']})
            oy=[int(o['is_anomaly']) for o in objects];oscores=[o['scores'][c['id']] for o in objects]
            m={'config':c['id'],**point_metrics(y,scores[ci]),'i_auroc':float(roc_auc_score(oy,oscores)),
               'i_ap':float(average_precision_score(oy,oscores))}
            if remainder is not None:m['remainder']=point_metrics(y[remainder],scores[ci][remainder])
            metrics.append(m)
        write_json(output/'objects.json',objects);files['objects.json']=sha256(output/'objects.json')
    finally:
        y.flush();scores.flush();y._mmap.close();scores._mmap.close()
    (work/'labels.npy').unlink();(work/'scores.npy').unlink()
    checked_protocol(root)
    write_json(output/'summary.json',{'category':category,'source_split':row['source_split'],
        'test_scans':len(cases),'point_valid_scans':sum(c['point_gt_valid'] for c in cases),
        'config_sha256':sha256(root/stage/split/'configs.json'),'metrics':metrics,
        'max_control_error':max_error,'seconds':time.perf_counter()-started,'files_sha256':files})


def verify_stage_category(root,stage,split,row):
    output=root/stage/split/row['category'];summary=read_json(output/'summary.json')
    if summary['config_sha256']!=sha256(root/stage/split/'configs.json'):raise ValueError('Stage config changed')
    cases=selected_cases(root,row,split)
    if summary['test_scans']!=len(cases) or summary['point_valid_scans']!=sum(c['point_gt_valid'] for c in cases):
        raise ValueError('Stage category coverage differs')
    if summary['max_control_error']>1e-10:raise ValueError('Invalid parity record')
    check_hashes({str(output/k):v for k,v in summary['files_sha256'].items()})
    return summary


def aggregate(rows,configs):
    result={'categories':len(rows),'test_scans':sum(r['test_scans'] for r in rows),
            'point_valid_scans':sum(r['point_valid_scans'] for r in rows),'metrics':[],
            'max_control_error':max(r['max_control_error'] for r in rows)}
    for c in configs:
        entries=[next(m for m in r['metrics'] if m['config']==c['id']) for r in rows]
        row={'config':c['id'],**{key:statistics.mean(e[key] for e in entries) for key in METRICS}}
        if 'remainder' in entries[0]:row['remainder']={key:statistics.mean(e['remainder'][key] for e in entries) for key in ('p_auroc','p_ap')}
        result['metrics'].append(row)
    return result


def execute_stage(root,stage,split,configs,workers,threads):
    _,baseline=checked_protocol(root)
    manifest=read_json(baseline/'dataset.json')
    rows=[r for r in manifest['categories'] if split=='full' or r['source_split']=='pcd']
    output=root/stage/split;output.mkdir(parents=True,exist_ok=True)
    if (output/'configs.json').exists():
        if read_json(output/'configs.json')!=configs:raise ValueError('Resume grid differs')
    else:write_json(output/'configs.json',configs)
    def memory(row):
        cases=selected_cases(root,row,split);n=sum(c['points'] for c in cases if c['point_gt_valid'])
        return int(640*1024**2+n*(8*len(configs)+24)+max(c['points'] for c in cases)*200)
    pending=[];done=[];active={};failed=None
    for row in sorted(rows,key=memory,reverse=True):
        if (output/row['category']/'summary.json').exists():done.append(verify_stage_category(root,stage,split,row))
        else:pending.append(row)
    (output/'logs').mkdir(exist_ok=True)
    try:
        while pending or active:
            for name,(process,row,stream) in list(active.items()):
                if process.poll() is not None:
                    stream.close();del active[name]
                    if process.returncode:failed=f'{stage}/{split}/{name}: exit {process.returncode}'
                    else:done.append(verify_stage_category(root,stage,split,row))
            if failed:raise RuntimeError(failed)
            for row in list(pending):
                if len(active)>=workers:break
                reserved=sum(memory(item[1]) for item in active.values())
                if memory(row)>min(7*GIB-reserved,available_memory()-768*1024**2):continue
                stream=(output/'logs'/f'{row["category"]}.log').open('a',encoding='utf-8')
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'_category','--run-root',str(root),
                    '--stage',stage,'--split',split,'--category',row['category'],'--threads',str(threads)]
                process=subprocess.Popen(command,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                active[row['category']]=(process,row,stream);pending.remove(row)
            state='RUNNING' if active else 'WAITING_FOR_MEMORY' if pending else 'RUNNING'
            write_json(root/'status.json',{'state':state,'stage':stage,'split':split,'active':list(active),
                'completed_categories':sorted(r['category'] for r in done),'pending':[r['category'] for r in pending],'updated_at':now()})
            if active or pending:time.sleep(2)
    finally:
        for process,_,stream in active.values():stop_process_tree(process);stream.close()
    if len(done)!=len(rows):raise ValueError('Stage category coverage mismatch')
    result=aggregate(done,configs)
    if split=='full':result['official_pcd']=aggregate([r for r in done if r['source_split']=='pcd'],configs)
    write_json(output/'summary.json',result)
    return result


def shortlist(summary,incumbent,baseline,limit=6):
    rows=summary['metrics'];control=next(r for r in rows if r['config']==incumbent['id'])
    rankings=[sorted(rows,key=lambda r:r[key],reverse=True) for key in ('p_auroc','p_ap')]
    rankings.append(sorted(rows,key=lambda r:min(r[k]-control[k] for k in ('p_auroc','p_ap')),reverse=True))
    ids=[baseline['id'],incumbent['id']]
    for index in range(len(rows)):
        for ranking in rankings:
            if ranking[index]['config'] not in ids:ids.append(ranking[index]['config'])
            if len(set(ids))>=limit+len({baseline['id'],incumbent['id']}):return set(ids)
    return set(ids)


def choose(summary,incumbent):
    control=next(r for r in summary['metrics'] if r['config']==incumbent['id'])
    eligible=[r for r in summary['metrics'] if min(r[k]-control[k] for k in ('p_auroc','p_ap'))>=-1e-4
              and max(r[k]-control[k] for k in ('p_auroc','p_ap'))>1e-4]
    if not eligible:return incumbent['id']
    best=max(eligible,key=lambda r:(min(r[k]-control[k] for k in ('p_auroc','p_ap')),
        sum(r[k]-control[k] for k in ('p_auroc','p_ap')),r['config']))
    return best['config']


def write_report(root,history,baseline,selected,full=None):
    lines=['# ShapeNet 分阶段调参','',
        '范围：官方 40 类分组开发/验证选参；最终报告 52 类与官方 40 类。',
        '冻结最新基线的模板和扫描配准；同数字 ID 的变体不跨开发/验证/内部参考组。',
        '已观察过的基准；内部参考组不是未见测试集。每个扫描重算基线，要求误差 <=1e-10。','',
        '|阶段|开发配置数|验证配置数|选中配置|验证 P-AUROC|验证 P-AP|','|---|---:|---:|---|---:|---:|']
    for h in history:
        m=h['selected_validation']
        lines.append(f"|{h['stage']}|{h['development_count']}|{h['validation_count']}|{h['selected']}|{100*m['p_auroc']:.5f}%|{100*m['p_ap']:.5f}%|")
    lines+=['','当前选中方法：','```json',json.dumps(selected['method'],indent=2),'```']
    if full:
        for name,value in [('all_52',full),('official_40',full['official_pcd'])]:
            lines+=['',name,'','|配置|P-AUROC|P-AP|I-AUROC|I-AP|','|---|---:|---:|---:|---:|']
            for row in value['metrics']:lines.append('|'+row['config']+'|'+'|'.join(f'{100*row[k]:.5f}%' for k in METRICS)+'|')
            b=next(r for r in value['metrics'] if r['config']==baseline['id'])
            s=next(r for r in value['metrics'] if r['config']==selected['id'])
            lines+=['',f"相对基线：P-AUROC {(s['p_auroc']-b['p_auroc'])*100:+.5f} 个百分点；P-AP {(s['p_ap']-b['p_ap'])*100:+.5f} 个百分点，AP 相对变化 {(s['p_ap']/b['p_ap']-1)*100:+.3f}%。"]
    temp=root/'REPORT.tmp';temp.write_text('\n'.join(lines)+'\n',encoding='utf-8');replace(temp,root/'REPORT.md')


def run(root,baseline,workers,threads,split_path=None):
    if workers<1 or threads<1 or workers*threads>(os.cpu_count() or 1):raise ValueError('Invalid CPU allocation')
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'.run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=now()
    try:
        if read_json(baseline/'status.json')['state']!='COMPLETE' or read_json(baseline/'verification.json')['status']!='PASS':
            raise ValueError('A complete verified full baseline is required')
        manifest=read_json(baseline/'dataset.json')
        if manifest['dataset']!='shapenet' or manifest['smoke'] or len(manifest['categories'])!=52:raise ValueError('Expected full 52-category ShapeNet baseline')
        if source_hashes()!=read_json(baseline/'protocol.json')['source_sha256']:raise ValueError('Baseline core source differs')
        base=setting(read_json(baseline/'config.json')['method'])
        if (root/'protocol.json').exists():checked_protocol(root)
        else:
            if shutil.disk_usage(root).free<15*GIB:raise ValueError('At least 15 GiB free required')
            splits=make_splits(manifest)
            if split_path is not None and read_json(split_path)!=splits:raise ValueError('Predeclared split differs')
            write_json(root/'splits.json',splits)
            write_json(root/'protocol.json',{'started_at':started,'source_sha256':all_sources(),
                'baseline_root':str(baseline),'baseline_sha256':{name:sha256(baseline/name) for name in ('config.json','dataset.json','protocol.json','verification.json','summary.json')},
                'splits_sha256':sha256(root/'splits.json'),'stages':list(STAGES),
                'selection':'Official 40 only; dev shortlist by AUC/AP/max-min gain; validation max-min gain, neither metric may drop >1e-4; material gain >1e-4.',
                'geometry':'Replay saved template and case transforms; frozen features, h, seeds.',
                'scope':'Development on observed public benchmark, not untouched test evidence.'})
            snapshot=root/'source_snapshot';shutil.copytree(CODE/'georeg3dad',snapshot/'georeg3dad',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(__file__,snapshot/'tune_shapenet.py')
        selected=base;history=[]
        for stage in STAGES:
            configs=grid(stage,selected,base)
            development=execute_stage(root,stage,'development',configs,workers,threads)
            ids=shortlist(development,selected,base)
            candidates=[c for c in configs if c['id'] in ids]
            validation=execute_stage(root,stage,'validation',candidates,workers,threads)
            cid=choose(validation,selected);selected=next(c for c in candidates if c['id']==cid)
            record={'stage':stage,'development_count':len(configs),'validation_count':len(candidates),
                'selected':cid,'selected_validation':next(m for m in validation['metrics'] if m['config']==cid)}
            history.append(record);write_json(root/'history.json',history);write_json(root/'selected_config.json',selected)
            write_report(root,history,base,selected)
        full=execute_stage(root,'final','full',unique([base,selected]),workers,threads)
        checked_protocol(root)
        if full['test_scans']!=1723 or full['point_valid_scans']!=1718:raise ValueError('Final scan coverage differs')
        write_json(root/'full_summary.json',full)
        write_json(root/'config_shapenet_selected.json',{'dataset':'shapenet','description':'Validation-selected on observed benchmark; frozen-pose search. See protocol.json.', 'method':selected['method']})
        write_report(root,history,base,selected,full)
        write_json(root/'verification.json',{'status':'PASS','test_scans':1723,'point_valid_scans':1718,'categories':52,'max_control_error':full['max_control_error'],
            'checks':['baseline hashes','source hashes','fixed grouped split','all per-case baseline parity','final compressed array equality','coverage']})
        write_json(root/'status.json',{'state':'COMPLETE','started_at':read_json(root/'protocol.json')['started_at'],'finished_at':now(),'selected':selected['id']})
        print('SHAPENET_TUNING_COMPLETE',root,flush=True)
    except BaseException:
        write_json(root/'status.json',{'state':'FAILED','error':traceback.format_exc(),'updated_at':now()});raise
    finally:lock.unlink()


def self_test(baseline):
    configure_threads(1)
    import numpy as np
    from georeg3dad.scoring import interpolate
    method=read_json(baseline/'config.json')['method'];base=setting(method)
    configs=grid('interpolation',base,base)
    rng=np.random.default_rng(51);query=rng.normal(size=(137,3));full=np.r_[query[:2],rng.normal(size=(49,3))]
    values={c['id']:rng.random(len(query)) for c in configs}
    actual=batch_interpolate(full,query,values,configs,1)
    for c in configs:
        expected=interpolate(full,query,values[c['id']],method_from_dict(c['method']).interpolation,1)
        if not np.array_equal(actual[c['id']],expected):raise AssertionError('Shared interpolation differs')
    manifest=read_json(baseline/'dataset.json')
    count=0;maximum=0.
    for row in manifest['categories'][:2]:
        library=frozen_library(baseline,row,method_from_dict(method))
        records={r['sample']:r for r in read_json(baseline/'results'/row['category']/'cases.json')}
        for anomaly in (False,True):
            case=next(c for c in row['cases'] if c['is_anomaly']==anomaly and c['point_gt_valid'])
            _,_,error,_=case_predictions(case,records[case['sample']],library,configs,baseline,row['category'],1)
            maximum=max(maximum,error);count+=1
    print('SHAPENET_SEARCH_SELF_TEST_PASS',len(configs),'configs',count,'real cases max_error',maximum,flush=True)


def smoke(root,baseline):
    if root.exists():raise ValueError('Smoke output must be new')
    root.mkdir(parents=True)
    manifest=read_json(baseline/'dataset.json')
    base=setting(read_json(baseline/'config.json')['method'])
    write_json(root/'splits.json',{r['category']:{c['sample']:'development' for c in r['cases']} for r in manifest['categories']})
    write_json(root/'protocol.json',{'source_sha256':all_sources(),'baseline_root':str(baseline),
        'baseline_sha256':{name:sha256(baseline/name) for name in ('config.json','dataset.json','protocol.json','summary.json')},
        'splits_sha256':sha256(root/'splits.json'),'scope':'SMOKE ONLY; no selection or benchmark claims'})
    configs=unique([base]+variations(base,'matching','unmatched_penalty',[1.,4.]))
    result=execute_stage(root,'smoke','development',configs,2,1)
    if result['categories']!=2 or result['test_scans']!=4:raise ValueError('Expected two-class/four-scan smoke baseline')
    write_json(root/'verification.json',{'status':'PASS',**result})
    print('SHAPENET_SEARCH_SMOKE_PASS 2 categories 4 scans',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','self-test','smoke'):
        p=sub.add_parser(name);p.add_argument('--baseline-root',type=Path,required=True)
        if name=='run':
            p.add_argument('--run-root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=2)
            p.add_argument('--splits',type=Path)
        elif name=='smoke':p.add_argument('--run-root',type=Path,required=True)
    p=sub.add_parser('_category');p.add_argument('--run-root',type=Path,required=True)
    for name in ('stage','split','category'):p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,required=True)
    args=parser.parse_args();configure_threads(getattr(args,'threads',1))
    if args.command=='self-test':self_test(args.baseline_root.resolve())
    elif args.command=='smoke':smoke(args.run_root.resolve(),args.baseline_root.resolve())
    elif args.command=='_category':category_stage(args.run_root.resolve(),args.stage,args.split,args.category,args.threads)
    else:run(args.run_root.resolve(),args.baseline_root.resolve(),args.workers,args.threads,args.splits)


if __name__=='__main__':main()
