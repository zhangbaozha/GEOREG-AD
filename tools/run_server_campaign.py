"""Run a verified ShapeNet baseline and its sequential search on one host."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from georeg3dad.runtime import read_json,write_json,sha256,source_hashes,stop_process_tree
from georeg3dad.runner import now,verify_category


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'.campaign.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    active=None;started=now()
    def status(state,stage,**extra):
        write_json(root/'status.json',{'state':state,'stage':stage,'pid':os.getpid(),
            'started_at':started,'updated_at':now(),**extra})
    def sources():
        return {'core':source_hashes(),'search':sha256(CODE/'tools/tune_shapenet.py'),
            'queue':sha256(__file__),'config':sha256(CODE/'configs/config_shapenet.json'),
            'splits':sha256(args.splits)}
    def verified(path):
        return ((path/'status.json').exists() and read_json(path/'status.json')['state']=='COMPLETE'
            and (path/'verification.json').exists() and read_json(path/'verification.json')['status']=='PASS')
    try:
        expected=sources()
        if (root/'protocol.json').exists():
            if read_json(root/'protocol.json')['sources']!=expected:raise ValueError('Campaign sources changed')
        else:write_json(root/'protocol.json',{'sources':expected,'started_at':started,'data_root':str(args.data_root.resolve()),
            'workers':args.workers,'threads':args.threads,'memory_budget_gib':os.environ.get('GEOREG_MEMORY_BUDGET_GIB','7')})
        baseline=root/'shapenet_full';tuning=root/'shapenet_tuning'
        commands=[('shapenet_full',baseline,[sys.executable,'-u','run.py','run','--config','configs/config_shapenet.json',
            '--data-root',str(args.data_root.resolve()),'--scope','all','--run-root',str(baseline),
            '--workers',str(args.workers),'--threads',str(args.threads)]),
            ('shapenet_tuning',tuning,[sys.executable,'-u','tools/tune_shapenet.py','run','--baseline-root',str(baseline),
            '--run-root',str(tuning),'--splits',str(args.splits.resolve()),'--workers',str(args.workers),'--threads',str(args.threads)])]
        for stage,output,command in commands:
            if sources()!=expected:raise ValueError('Campaign source changed')
            if not verified(output):
                if stage=='shapenet_full' and (output/'config.json').exists():command.append('--resume')
                with (root/(stage+'.log')).open('a',encoding='utf-8') as stream:
                    active=subprocess.Popen(command,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                    while active.poll() is None:
                        status('RUNNING',stage,child_pid=active.pid);time.sleep(5)
                    code=active.returncode;active=None
                    if code:raise RuntimeError(f'{stage} exited {code}; inspect its log')
            if not verified(output):raise ValueError('Missing verification: '+stage)
            if stage=='shapenet_full':
                for row in read_json(baseline/'dataset.json')['categories']:verify_category(baseline,row)
        if sources()!=expected:raise ValueError('Campaign source changed')
        status('COMPLETE','complete',finished_at=now())
    except BaseException:
        status('FAILED','queue_failed',error=traceback.format_exc());raise
    finally:
        if active is not None:stop_process_tree(active)
        lock.unlink()


if __name__=='__main__':main()
