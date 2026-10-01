"""Create a read-only Real3D-AD manifest for GeoReg component ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows: list[dict[str, object]] = []
    for category in sorted(path for path in root.iterdir() if path.is_dir()):
        templates = sorted((category / "train").glob("*.pcd"))
        tests = sorted((category / "test").glob("*.pcd"))
        gt = sorted((category / "gt").glob("*.txt"))
        if len(templates) != 4 or not tests:
            raise ValueError(f"invalid category layout: {category}")
        normal = [path for path in tests if "good" in path.stem]
        anomaly = [path for path in tests if "good" not in path.stem]
        if len(gt) != len(anomaly):
            raise ValueError(f"annotation/test mismatch: {category}")
        rows.append({
            "adapter_category": category.name,
            "source_split": "real3dad",
            "templates": len(templates),
            "test_scans": len(tests),
            "normal_scans": len(normal),
            "anomaly_scans": len(anomaly),
            "gt_files": len(gt),
        })
    if len(rows) != 12:
        raise ValueError(f"expected 12 categories, found {len(rows)}")
    document = {
        "data_root": str(root),
        "categories": rows,
        "totals": {
            "categories": len(rows),
            "test_scans": sum(int(row["test_scans"]) for row in rows),
            "normal_scans": sum(int(row["normal_scans"]) for row in rows),
            "anomaly_scans": sum(int(row["anomaly_scans"]) for row in rows),
            "gt_files": sum(int(row["gt_files"]) for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
