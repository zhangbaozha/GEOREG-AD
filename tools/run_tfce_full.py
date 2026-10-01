"""Server full-cohort continuation of the fixed TFCE experiment (no retuning)."""
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import run_tfce_experiment as base


def full_plan(args):
    output,source,data = args.output.resolve(),args.source.resolve(),args.data_root.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('New empty output required')
    config = base.read_json(source/'config.json'); manifest = base.read_json(source/'dataset.json')
    if config['dataset']!='real3dad' or base.read_json(source/'status.json')['state']!='COMPLETE' or base.read_json(source/'verification.json')['status']!='PASS':
        raise ValueError('Require verified completed Real3D archive')
    if config['method']['interpolation']['k']!=128 or config['method']['interpolation']['power']!=0:
        raise ValueError('Require original k128 p0 control')
    metadata={str(source/name):base.sha256(source/name) for name in ('config.json','dataset.json','status.json','verification.json')}
    rows=[]
    for original in manifest['categories'][:1] if args.smoke else manifest['categories']:
        folder=source/'results'/original['category']; previous_summary=base.read_json(folder/'summary.json')
        for name in ('templates.json','cases.json'):
            if base.sha256(folder/name)!=previous_summary['files_sha256'][name]:raise ValueError('Metadata hash mismatch')
            metadata[str(folder/name)]=base.sha256(folder/name)
        metadata[str(folder/'summary.json')]=base.sha256(folder/'summary.json')
        previous={r['sample']:r for r in base.read_json(folder/'cases.json')}
        selected=base.select(original['cases'],original['category'],1) if args.smoke else original['cases']
        def mapped(path):
            parts=path.replace('\\','/').split('/')
            return str(data/original['source_category']/parts[-2]/parts[-1])
        checks={mapped(p):original['input_sha256'][p] for p in original['templates']}
        cases=[]
        for raw in selected:
            c=deepcopy(raw);c['test']=mapped(raw['test']);c['gt']=mapped(raw['gt']) if raw['gt'] else None
            checks[c['test']]=original['input_sha256'][raw['test']]
            if c['gt']:checks[c['gt']]=original['input_sha256'][raw['gt']]
            rel=previous[c['sample']]['score_file'];checks[str(folder/rel)]=previous_summary['files_sha256'][rel]
            cases.append(c)
        templates=base.read_json(folder/'templates.json')
        for record in templates['registrations']:record['template']=mapped(record['template'])
        rows.append(dict(category=original['category'],cases=cases,folder=str(folder),
            records={c['sample']:previous[c['sample']] for c in cases},templates=templates,input_sha256=checks))
    code=base.hashes();code[str(Path(__file__).resolve())]=base.sha256(__file__)
    spec=dict(created_at=base.now(),source=str(source),source_metadata_sha256=metadata,code_sha256=code,
        dataset='real3dad',smoke=args.smoke,scope='smoke' if args.smoke else 'full',rows=rows,method=config['method'],
        variants=base.VARIANTS,tfce=dict(E=.5,H=2.,connectivity=26,extent='occupied voxel count',
            integration='exact continuous integral via merge tree',presmoothing=False,normalization='none'),
        mapping='Original Open3D scan grid min(full_xyz)-voxel/2 before registration; inherit voxel score.',
        selection='None. Complete original 1206-scan cohort; all four arms retained. No changes after pilot metrics.',
        primary='Class-macro pooled point AUROC/AP, original valid-GT mask.',
        secondary='Object AUROC/AP with archived Top 1%, all original scans including invalid-point-GT cases.',
        limitation='Observed benchmark and previously tuned geometry; descriptive full-cohort confirmation, not an independent holdout.',
        workers=2 if args.smoke else 8,threads_per_worker=2,process_start_method='spawn',replay_tolerance=1e-10,
        references=['https://pubmed.ncbi.nlm.nih.gov/18501637/','https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/randomise.html'])
    base.verify(metadata)
    if not args.smoke and (len(rows)!=12 or sum(len(r['cases']) for r in rows)!=1206 or sum(c['point_gt_valid'] for r in rows for c in r['cases'])!=1201):
        raise ValueError('Full cohort mismatch')
    # Large input files are verified by each worker before/after inference.
    base.write_json(output/'PLAN.json',spec)
    base.write_json(output/'status.json',dict(state='PLANNED',created_at=base.now(),categories=len(rows),scans=sum(len(r['cases']) for r in rows)))
    print(f'FULL_PLAN_FROZEN categories={len(rows)} scans={sum(len(r["cases"]) for r in rows)} sha256={base.sha256(output/"PLAN.json")}',flush=True)


def full_run(args):
    # Open3D initializes threads on import; Linux fork may inherit locked state.
    # Fresh interpreters match the already validated Windows execution semantics.
    base.ProcessPoolExecutor=partial(ProcessPoolExecutor,mp_context=get_context('spawn'))
    original_report=base.report
    def report(output,summaries,elapsed):
        result=original_report(output,summaries,elapsed)
        path=output/'REPORT.md'
        path.write_text(path.read_text(encoding='utf-8').replace('Real3D 先导实验','Real3D 服务器全量对照').replace('固定方案先导对照','固定方案全量对照'),encoding='utf-8')
        return result
    base.report=report
    base.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan')
    for name in ('source','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();full_plan(args) if args.command=='plan' else full_run(args)
