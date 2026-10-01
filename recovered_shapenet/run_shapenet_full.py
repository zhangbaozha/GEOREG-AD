"""Full CPU evaluation of GeoReg-AD on the read-only Anomaly-ShapeNet adapter."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=0.05)
    args = parser.parse_args()
    # This adapter carries only the GT-delimiter compatibility change.  The
    # geometry scoring, registration, interpolation, and aggregation logic are
    # otherwise the audited GeoReg-AD implementation copied alongside it.
    code_root = Path(__file__).resolve().parent
    baseline = code_root / "raw_baseline.py"
    audit_code = code_root / "audit_all.py"
    summary_code = Path(__file__).resolve().parent / "summarize_shapenet.py"
    data_root, run_root = args.data_root.resolve(), args.run_root.resolve()
    if not baseline.is_file() or not audit_code.is_file() or not summary_code.is_file():
        raise FileNotFoundError("expected original GeoReg code and ShapeNet summarizer")
    if not data_root.is_dir() or not args.layout_manifest.is_file():
        parser.error("data root or layout manifest is missing")
    if run_root.exists() and any(run_root.iterdir()):
        parser.error("run-root must be new or empty")
    if args.workers < 1 or args.voxel <= 0:
        parser.error("workers and voxel must be positive")

    manifest = json.loads(args.layout_manifest.read_text(encoding="utf-8"))
    categories = [row["adapter_category"] for row in manifest["categories"]]
    if set(categories) != {path.name for path in data_root.iterdir() if (path / "test").is_dir()}:
        raise ValueError("layout manifest and adapter directories differ")
    if any(len(list((data_root / category / "train").glob("*.pcd"))) != 4 for category in categories):
        raise ValueError("every category must have four normal templates")

    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    results = run_root / "results"
    logs.mkdir()
    results.mkdir()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

    audit_path = run_root / "gt_coordinate_audit.json"
    with (logs / "audit.log").open("w", encoding="utf-8") as log:
        audit_status = subprocess.run(
            [sys.executable, "-u", str(audit_code), "--data-root", str(data_root), "--output", str(audit_path)],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        ).returncode
    if not audit_path.exists() or audit_status not in (0, 1):
        raise RuntimeError("GT audit did not complete; inspect logs/audit.log")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_gt = sum(int(row["gt_files"]) for row in manifest["categories"])
    if audit["processed"] != audit["total"] or audit["total"] != expected_gt:
        raise RuntimeError("GT audit did not cover every annotation")
    (run_root / "audit_summary.json").write_text(json.dumps({
        "gt_files": expected_gt,
        "coordinate_aligned": audit["passed"],
        "point_metric_exclusions": audit["failed"],
        "failure_paths": [item["gt"] for item in audit["failures"]],
    }, indent=2), encoding="utf-8")

    def run_category(category: str) -> tuple[str, int]:
        command = [sys.executable, "-u", str(baseline), "--data-root", str(data_root),
                   "--category", category, "--output-dir", str(results),
                   "--audit-result", str(audit_path), "--voxel", str(args.voxel)]
        with (logs / f"{category}.log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return category, status

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = dict(pool.map(run_category, categories))
    if any(status.values()):
        raise RuntimeError(f"category process failure: {status}")
    incomplete = []
    for category in categories:
        config = json.loads((results / category / "config.json").read_text(encoding="utf-8"))
        cases = list((results / category / "cases").glob("*.json"))
        failures = list((results / category / "failures").glob("*.json"))
        if len(cases) != int(config["test_count"]) or failures:
            incomplete.append({"category": category, "expected": config["test_count"],
                               "cases": len(cases), "failures": [path.name for path in failures]})
    if incomplete:
        (run_root / "incomplete_categories.json").write_text(json.dumps(incomplete, indent=2), encoding="utf-8")
        raise RuntimeError("case failures recorded; do not summarize incomplete output")

    with (logs / "summary.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(summary_code), "--results", str(results),
                        "--layout-manifest", str(args.layout_manifest)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    run_manifest = {
        "data_root": str(data_root),
        "layout_manifest": str(args.layout_manifest.resolve()),
        "categories": len(categories),
        "workers": args.workers,
        "voxel": args.voxel,
        "cpu_only": True,
        "category_status": status,
        "audit": {"gt_files": expected_gt, "passed": audit["passed"], "failed": audit["failed"]},
    }
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("COMPLETE", run_root / "shapenet_full_summary.json", flush=True)


if __name__ == "__main__":
    main()
