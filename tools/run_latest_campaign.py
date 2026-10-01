"""Durable local queue: finish Real3D, run ShapeNet, then tune ShapeNet."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from georeg3dad.runtime import read_json,write_json,sha256,source_hashes,stop_process_tree
from georeg3dad.runner import now,verify_category


def verified(path):
    return ((path/'status.json').exists() and read_json(path/'status.json')['state']=='COMPLETE'
            and (path/'verification.json').exists() and read_json(path/'verification.json')['status']=='PASS')


def alive_pid(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000,False,pid)
        if not handle:return False
        try:return kernel.WaitForSingleObject(handle,0)==258
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--shape-data',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'.campaign.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    active=None;started=now()
    def status(state,stage,**details):
        write_json(root/'status.json',{'state':state,'stage':stage,'pid':os.getpid(),'started_at':started,'updated_at':now(),**details})
    def check_sources():
        expected=read_json(root/'campaign_protocol.json')
        actual={'core':source_hashes(),'tuning':sha256(CODE/'tools/tune_shapenet.py'),
                'queue':sha256(__file__),'shape_config':sha256(CODE/'configs/config_shapenet.json'),
                'splits':sha256(root/'shapenet_predeclared_splits.json')}
        if expected['sources']!=actual:raise ValueError('Campaign source/config changed')
    def run_command(stage,command):
        nonlocal active
        check_sources()
        with (root/(stage+'.log')).open('a',encoding='utf-8') as stream:
            active=subprocess.Popen(command,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
            while active.poll() is None:
                status('RUNNING',stage,child_pid=active.pid,command=command);time.sleep(5)
            code=active.returncode;active=None
            if code:raise RuntimeError(f'{stage} exited {code}; inspect {stage}.log')
    try:
        sources={'core':source_hashes(),'tuning':sha256(CODE/'tools/tune_shapenet.py'),
                 'queue':sha256(__file__),'shape_config':sha256(CODE/'configs/config_shapenet.json'),
                 'splits':sha256(root/'shapenet_predeclared_splits.json')}
        if (root/'campaign_protocol.json').exists():check_sources()
        else:write_json(root/'campaign_protocol.json',{'sources':sources,'started_at':started,
            'sequence':['real3dad_full','shapenet_full','shapenet_tuning'],
            'workers':args.workers,'threads':args.threads,'data':str(args.shape_data.resolve())})
        real=root/'real3dad_full'
        while not verified(real):
            if (real/'status.json').exists() and read_json(real/'status.json')['state']=='FAILED':raise RuntimeError('Real3D full failed; queue stopped')
            if (real/'.run.lock').exists() and not alive_pid(int((real/'.run.lock').read_text())):raise RuntimeError('Real3D runner no longer alive; queue stopped')
            status('RUNNING','real3dad_full',detail='Waiting for already launched full runner')
            time.sleep(5)
        check_sources()
        for row in read_json(real/'dataset.json')['categories']:verify_category(real,row)
        shape=root/'shapenet_full'
        if not verified(shape):
            command=[sys.executable,'-u','run.py','run','--config','configs/config_shapenet.json',
                '--data-root',str(args.shape_data.resolve()),'--scope','all','--run-root',str(shape),
                '--workers',str(args.workers),'--threads',str(args.threads)]
            if (shape/'config.json').exists():command.append('--resume')
            run_command('shapenet_full',command)
        if not verified(shape):raise ValueError('ShapeNet full verification missing')
        for row in read_json(shape/'dataset.json')['categories']:verify_category(shape,row)
        write_json(root/'full_comparison.json',{'real3dad':read_json(real/'summary.json'),'shapenet':read_json(shape/'summary.json')})
        lines=['# 最新代码全量实验','', '两次运行均完成且通过完整性检查。','',
            '|数据集/范围|类别|扫描|P-AUROC|P-AP|I-AUROC|I-AP|','|---|---:|---:|---:|---:|---:|---:|']
        for dataset,path in [('Real3D',real),('ShapeNet',shape)]:
            for group,row in read_json(path/'summary.json')['groups'].items():
                lines.append(f'|{dataset}/{group}|{row["categories"]}|{row["test_scans"]}|'+
                    '|'.join(f'{100*row["metrics"][k]:.5f}%' for k in ('p_auroc','p_ap','i_auroc','i_ap'))+'|')
        lines+=['','本次重新执行配准；历史固定配准调参结果见 REAL3D_TUNING_REVIEW.md。']
        (root/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        tuning=root/'shapenet_tuning'
        if not verified(tuning):
            run_command('shapenet_tuning',[sys.executable,'-u','tools/tune_shapenet.py','run',
                '--baseline-root',str(shape),'--run-root',str(tuning),'--workers',str(args.workers),'--threads',str(args.threads),
                '--splits',str(root/'shapenet_predeclared_splits.json')])
        if not verified(tuning):raise ValueError('Tuning verification missing')
        check_sources();status('COMPLETE','complete',finished_at=now())
    except BaseException:
        status('FAILED','queue_failed',error=traceback.format_exc());raise
    finally:
        if active is not None:stop_process_tree(active)
        lock.unlink()


if __name__=='__main__':main()
