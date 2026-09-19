"""Run four CPU category workers and aggregate only after all scans finish."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = Path("/home/zhw/data/Real3D-AD-PCD")


def run_category(category):
    command = [sys.executable, "-u", str(CODE / "raw_baseline.py"),
               "--data-root", str(DATA), "--category", category,
               "--output-dir", str(ROOT / "results"),
               "--audit-result", str(CODE / "gt_coordinate_audit.json")]
    with (ROOT / "logs" / f"{category}.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(f"CATEGORY_EXIT {category} {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    categories = sorted(p.name for p in DATA.iterdir() if (p / "test").is_dir())
    if len(categories) != 12:
        raise ValueError(categories)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(run_category, categories))
    if any(statuses):
        raise RuntimeError(statuses)
    subprocess.run([sys.executable, "-u", str(CODE / "summarize_raw_full.py"),
                    "--results", str(ROOT / "results")], check=True)
    print("RAW_FULL_COMPLETE", flush=True)
