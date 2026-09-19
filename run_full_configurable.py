"""Parameter-driven launcher for the unchanged historical baseline modules."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_BAD = {
    ("chicken", "705_bulge"), ("chicken", "706_sink"),
    ("seahorse", "265_bulge"), ("seahorse", "265_bulge_cut"),
    ("starfish", "433_bulge"),
}
CODE = Path(__file__).resolve().parent / "original"


def run_category(category, data_root, run_root, audit_file, env):
    cmd = [sys.executable, "-u", str(CODE / "raw_baseline.py"), "--data-root", str(data_root),
           "--category", category, "--output-dir", str(run_root / "results"),
           "--audit-result", str(audit_file)]
    with (run_root / "logs" / (category + ".log")).open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data_root, run_root = args.data_root.resolve(), args.run_root.resolve()
    if not data_root.is_dir():
        parser.error("Data root does not exist")
    if run_root.exists() and any(run_root.iterdir()):
        parser.error("Run root must be a new empty directory; the original code does not hash-check resumed cases")
    if args.workers < 1:
        parser.error("--workers must be positive")
    categories = sorted(p.name for p in data_root.iterdir() if (p / "test").is_dir())
    if len(categories) != 12 or any(len(list((data_root / c / "train").glob("*.pcd"))) != 4 for c in categories):
        parser.error("Expected 12 categories with four training PCD templates each")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir()
    (run_root / "results").mkdir()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    audit_file = run_root / "gt_coordinate_audit.json"
    with (run_root / "logs" / "audit.log").open("w", encoding="utf-8") as log:
        audit_status = subprocess.run([sys.executable, "-u", str(CODE / "audit_all.py"),
                                       "--data-root", str(data_root), "--output", str(audit_file)],
                                      stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if not audit_file.exists():
        raise RuntimeError("GT audit did not produce a manifest; see logs/audit.log")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    bad = {(Path(x["gt"]).parts[-3], Path(x["gt"]).stem) for x in audit["failures"]}
    if audit["total"] != 602 or audit["passed"] != 597 or bad != EXPECTED_BAD:
        raise RuntimeError("Dataset does not match the historical GT audit; inspect logs/audit.log")
    if audit_status not in (0, 1):
        raise RuntimeError("GT audit failed unexpectedly; inspect logs/audit.log")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_category, c, data_root, run_root, audit_file, env): c for c in categories}
        statuses = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
    print("CATEGORY_EXIT", statuses, flush=True)
    if any(statuses.values()):
        raise RuntimeError("At least one category failed; inspect logs and results/*/failures")
    with (run_root / "logs" / "summary.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(CODE / "summarize_raw_full.py"),
                        "--results", str(run_root / "results")],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    print("COMPLETE", str(run_root / "raw_full_summary.json"), flush=True)


if __name__ == "__main__":
    main()
