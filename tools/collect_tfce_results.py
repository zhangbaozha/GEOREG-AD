"""Package completed TFCE metadata and frozen sources; leave point arrays remote."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(sys.argv[1])
assert read(root/'status.json')['state'] == 'COMPLETE'
assert read(root/'verification.json')['status'] == 'PASS'
assert read(root/'INDEPENDENT_VERIFICATION.json')['status'] == 'PASS'
plan = read(root/'PLAN.json')
original = read(Path(plan['source'])/'summary.json')['groups']['all_selected']['metrics']
current = next(r for r in read(root/'summary.json')['aggregates'] if r['variant'] == 'knn128_p0')
errors = {k: abs(current[k]-original[k]) for k in original}
assert errors['p_auroc'] < 1e-10 and errors['p_ap'] < 1e-7
assert errors['i_auroc'] < 1e-12 and errors['i_ap'] < 1e-12
(root/'SOURCE_CONTROL_CHECK.json').write_text(json.dumps(dict(status='PASS', archived_metrics=original,
    replayed_metrics=current, absolute_errors=errors), indent=2))
code = root.parent/'code_spawn'
for name, digest in plan['code_sha256'].items():
    source = Path(name)
    assert sha(source) == digest
    target = root/'source_snapshot'/source.relative_to(code)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
shutil.copyfile(root.parent/'audit_parallel.py', root/'source_snapshot'/'audit_parallel.py')
shutil.copyfile(__file__, root/'source_snapshot'/'collect_tfce_results.py')
for name, digest in read(root/'receipt.json').items():
    assert sha(root/name) == digest
files = [p for p in root.rglob('*') if p.is_file()
         and p.suffix in ('.json', '.md', '.csv', '.log', '.py') and p.name != 'download_receipt.json']
receipt = {p.relative_to(root).as_posix(): sha(p) for p in files}
(root/'download_receipt.json').write_text(json.dumps(receipt, indent=2))
archive = root.parent/'full_spawn_results.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in files+[root/'download_receipt.json']:
        tar.add(p, arcname=p.relative_to(root).as_posix())
print('COLLECTION_READY', archive, archive.stat().st_size, sha(archive), flush=True)
