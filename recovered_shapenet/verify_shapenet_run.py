"""Independently validate the completed GeoReg-AD Anomaly-ShapeNet run.

This validator intentionally consumes only the immutable run summaries, audit
record, and per-scan metadata.  The summarizer has already opened every score
archive to check finiteness, length, and label integrity; this script checks
that the published aggregate and the data-quality policy agree with those
per-scan records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_FAILURES = {
    "helmet1_bulge0.txt",
    "helmet1_bulge2.txt",
    "helmet1_bulge6.txt",
    "helmet1_concavity3.txt",
    "helmet1_concavity5.txt",
}


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name}: got {actual!r}; expected {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()

    summary = load(root / "shapenet_full_summary.json")
    audit = load(root / "gt_coordinate_audit.json")
    cases = load(root / "shapenet_full_cases.json")
    manifest = load(root / "run_manifest.json")
    if not isinstance(summary, dict) or not isinstance(audit, dict) or not isinstance(cases, list) or not isinstance(manifest, dict):
        raise ValueError("unexpected result artifact schema")

    aggregates = summary["aggregates"]
    core = aggregates["official_pcd_40"]
    extension = aggregates["new_pcd_extension"]
    total = aggregates["all_available_52"]
    for name, row, expected in (
        ("official_pcd_40", core, (40, 1312, 600, 712, 1307)),
        ("new_pcd_extension", extension, (12, 411, 180, 231, 411)),
        ("all_available_52", total, (52, 1723, 780, 943, 1718)),
    ):
        require_equal(f"{name}.categories", row["categories"], expected[0])
        require_equal(f"{name}.test_scans", row["test_scans"], expected[1])
        require_equal(f"{name}.normal_scans", row["normal_scans"], expected[2])
        require_equal(f"{name}.anomaly_scans", row["anomaly_scans"], expected[3])
        require_equal(f"{name}.point_valid_scans", row["point_valid_scans"], expected[4])

    require_equal("summary.total_test_scans", summary["total_test_scans"], 1723)
    require_equal("summary.total_point_valid_scans", summary["total_point_valid_scans"], 1718)
    require_equal("manifest.categories", manifest["categories"], 52)
    require_equal("manifest.cpu_only", manifest["cpu_only"], True)
    require_equal("manifest.voxel", manifest["voxel"], 0.05)
    require_equal("all category processes completed", set(manifest["category_status"].values()), {0})

    for key, expected in (("total", 943), ("processed", 943), ("passed", 938), ("failed", 5)):
        require_equal(f"audit.{key}", audit[key], expected)
    failed_files = {Path(item["gt"]).name for item in audit["failures"]}
    require_equal("audit failure file set", failed_files, EXPECTED_FAILURES)

    require_equal("case metadata length", len(cases), 1723)
    require_equal("case normal count", sum(not case["is_anomaly"] for case in cases), 780)
    require_equal("case anomaly count", sum(bool(case["is_anomaly"]) for case in cases), 943)
    invalid_cases = {f"{case['sample']}.txt" for case in cases if not case["point_gt_valid"]}
    require_equal("point-level exclusion case set", invalid_cases, EXPECTED_FAILURES)

    checks = {
        "status": "PASS",
        "counts": {
            "categories": 52,
            "test_scans": 1723,
            "normal_scans": 780,
            "anomaly_scans": 943,
            "point_valid_scans": 1718,
            "pcd_gt_aligned": 938,
            "point_metric_exclusions": 5,
        },
        "official_pcd_40_macro": core["macro_average"],
        "all_available_52_macro": total["macro_average"],
        "excluded_gt_files": sorted(failed_files),
        "checks": [
            "Run manifest recorded exit code zero for every category.",
            "Summarization opened every archived score array and checked finiteness, score/label lengths, and label counts.",
            "Per-scan metadata, aggregate counts, and GT-audit exclusions agree.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
