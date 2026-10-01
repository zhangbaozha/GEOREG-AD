"""Independent sklearn metric replay and summary plot for the frozen TFCE pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_category(job):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    root, plan, category = job
    methods = plan['variants']; errors = []; rows = []
    if category:
        folder = root/'categories'/category['category']
        y = np.load(folder/'point_labels.npy',mmap_mode='r')
        scores = np.load(folder/'point_scores.npy',mmap_mode='r')
        objects = read(folder/'objects.json')
        intended = next(r for r in plan['rows'] if r['category']==category['category'])
        assert [r['sample'] for r in objects] == [r['sample'] for r in intended['cases']]
        assert len(y)==sum(r['points'] for r in objects if r['point_gt_valid'])==category['points']
        assert scores.shape==(len(methods),len(y))
        cursor = 0
        for record in objects:
            previous = intended['records'][record['sample']]
            with np.load(Path(intended['folder'])/previous['score_file']) as old:
                old_labels = old['labels']
                if record['point_gt_valid']:
                    assert np.array_equal(y[cursor:cursor+len(old_labels)],old_labels)
                    assert np.max(np.abs(scores[0,cursor:cursor+len(old_labels)]-old['scores']))<=1e-10
                    cursor += len(old_labels)
        labels = [int(r['is_anomaly']) for r in objects]
        for i,method in enumerate(methods):
            values = [r['scores'][method] for r in objects]
            actual = dict(p_auroc=float(roc_auc_score(y,scores[i])),p_ap=float(average_precision_score(y,scores[i])),
                          i_auroc=float(roc_auc_score(labels,values)),i_ap=float(average_precision_score(labels,values)))
            reported = next(m for m in category['metrics'] if m['variant']==method)
            errors.extend(abs(actual[k]-reported[k]) for k in actual)
            rows.append(dict(category=category['category'],variant=method,**actual))
        print('INDEPENDENT_METRICS_PASS '+category['category'],flush=True)
    return rows, errors


def verify(root, workers=1):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    plan,summary = read(root/'PLAN.json'),read(root/'summary.json')
    assert read(root/'status.json')['state']=='COMPLETE'
    assert read(root/'verification.json')['status']=='PASS'
    methods = plan['variants']; errors = []; rows = []
    jobs = [(root, plan, category) for category in summary['categories_detail']]
    if workers == 1:
        results = list(map(verify_category, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            results = list(pool.map(verify_category, jobs))
    for category_rows, category_errors in results:
        rows.extend(category_rows); errors.extend(category_errors)
    max_error = max(errors)
    assert max_error < 1e-12, max_error
    for method in methods:
        expected = next(r for r in summary['aggregates'] if r['variant']==method)
        for metric in ('p_auroc','p_ap','i_auroc','i_ap'):
            actual = np.mean([r[metric] for r in rows if r['variant']==method])
            assert abs(actual-expected[metric]) < 1e-12
    baseline = next(r for r in summary['aggregates'] if r['variant']=='knn128_p0')
    raw = next(r for r in summary['aggregates'] if r['variant']=='voxel_raw')
    tfce = next(r for r in summary['aggregates'] if r['variant']=='voxel_tfce')
    deltas = {k:dict(vs_knn128=tfce[k]-baseline[k],vs_voxel_raw=tfce[k]-raw[k]) for k in ('p_auroc','p_ap','i_auroc','i_ap')}
    wins = {}
    for metric in ('p_auroc','p_ap'):
        differences = [next(m for m in r['metrics'] if m['variant']=='voxel_tfce')[metric]-next(m for m in r['metrics'] if m['variant']=='knn128_p0')[metric] for r in summary['categories_detail']]
        wins[metric] = dict(improved=sum(x>1e-12 for x in differences),declined=sum(x< -1e-12 for x in differences),tied=sum(abs(x)<=1e-12 for x in differences))
    result = dict(status='PASS',independent_metric_library='sklearn',categories=len(summary['categories_detail']),
        metric_rows=len(rows),max_metric_error=max_error,label_and_control_array_replay='PASS',
        tfce_deltas=deltas,tfce_category_changes_vs_knn128=wins,
        limitation='Descriptive paired observed-benchmark comparison; no hyperparameter or winner selection.')
    (root/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    files = [root/x for x in ('PLAN.json','summary.json','REPORT.md','verification.json','INDEPENDENT_VERIFICATION.json','metrics.csv')]
    receipt = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    s = read(root/'summary.json')
    scope = 'full benchmark' if read(root/'PLAN.json').get('scope') == 'full' else 'pilot'
    labels = ['k=128, p=0','k=3, p=1','Voxel raw','Voxel TFCE']
    colors = ['#475569','#94a3b8','#f59e0b','#0f766e']
    fig,axes = plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios':[1,1,1.6]})
    fig.suptitle(f'Fixed-default TFCE vs kNN smoothing | Real3D-AD {scope}',fontsize=16,fontweight='bold')
    for ax,metric,title in zip(axes[:2],['p_auroc','p_ap'],['Point AUROC','Point average precision']):
        values = [r[metric] for r in s['aggregates']]
        ax.barh(labels,values,color=colors,height=.6)
        ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+.015,i,f'{v:.4f}',va='center',fontsize=10)
    ax=axes[2];cats=s['categories_detail']
    differences = [100*(next(m for m in r['metrics'] if m['variant']=='voxel_tfce')['p_ap']-next(m for m in r['metrics'] if m['variant']=='knn128_p0')['p_ap']) for r in cats]
    ax.barh([r['category'] for r in cats],differences,color=['#0f766e' if v>=0 else '#b45309' for v in differences])
    ax.axvline(0,color='#334155',lw=1);ax.invert_yaxis();ax.set_title('TFCE - k128: point AP');ax.set_xlabel('Percentage points');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for panel in axes:
        panel.spines[['top','right']].set_visible(False)
    fig.text(.5,.025,f'{s["categories"]} classes / {s["scans"]} scans / {s["points"]:,} points | E=0.5, H=2, 26-connectivity | Observed benchmark',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg');plt.close(fig)
    print('PLOT_SAVED '+str(root/'comparison.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('verify','plot'));p.add_argument('root',type=Path)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();verify(a.root,a.workers) if a.command=='verify' else plot(a.root)
