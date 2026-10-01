"""Create a read-only GeoReg-AD adapter for Anomaly-ShapeNet v2.

The source dataset is never modified.  The adapter exposes the directory
contract used by the audited GeoReg scorer: lower-case ``train``, ``test``, and
``gt`` directories, with normal test clouds named ``*_good*.pcd``.  Original
Anomaly-ShapeNet normal test files use ``positive`` instead, so only their
adapter symlink names change.  Anomalous file names and their GT names remain
identical.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def is_normal_name(path: Path) -> bool:
    return "_positive" in path.stem


def symlink(source: Path, destination: Path, *, target_is_directory: bool = False) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source, destination, target_is_directory=target_is_directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Anomaly-ShapeNet-v2/dataset directory")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new, initially empty adapter directory")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error("output-root must be new or empty")
    if not source_root.is_dir():
        parser.error("source-root does not exist")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for split in ("pcd", "new_pcd"):
        split_root = source_root / split
        if not split_root.is_dir():
            parser.error(f"missing dataset split: {split_root}")
        for source_category in sorted(path for path in split_root.iterdir() if path.is_dir()):
            train = source_category / "train"
            test = source_category / "test"
            gt = source_category / "GT"
            templates = sorted(train.glob("*.pcd"))
            tests = sorted(test.glob("*.pcd"))
            if len(templates) != 4 or not tests or not gt.is_dir():
                raise ValueError(f"invalid category layout: {source_category}")

            key = f"{split}__{source_category.name}"
            destination = output_root / key
            destination.mkdir()
            symlink(train, destination / "train", target_is_directory=True)
            symlink(gt, destination / "gt", target_is_directory=True)
            test_destination = destination / "test"
            test_destination.mkdir()

            normal_count = 0
            anomaly_count = 0
            for cloud in tests:
                if is_normal_name(cloud):
                    name = cloud.name.replace("_positive", "_good")
                    normal_count += 1
                else:
                    name = cloud.name
                    anomaly_count += 1
                    annotation = gt / f"{cloud.stem}.txt"
                    if not annotation.is_file():
                        raise FileNotFoundError(annotation)
                symlink(cloud, test_destination / name)

            records.append({
                "adapter_category": key,
                "source_split": split,
                "source_category": source_category.name,
                "source_path": str(source_category),
                "templates": len(templates),
                "test_scans": len(tests),
                "normal_scans": normal_count,
                "anomaly_scans": anomaly_count,
                "gt_files": len(list(gt.glob("*.txt"))),
            })

    manifest = {
        "source_root": str(source_root),
        "normal_filename_rule": "source test stem contains '_positive'; adapter renames it to '_good'",
        "categories": records,
        "totals": {
            "categories": len(records),
            "test_scans": sum(int(row["test_scans"]) for row in records),
            "normal_scans": sum(int(row["normal_scans"]) for row in records),
            "anomaly_scans": sum(int(row["anomaly_scans"]) for row in records),
            "gt_files": sum(int(row["gt_files"]) for row in records),
        },
    }
    atomic_json(output_root / "layout_manifest.json", manifest)
    print(json.dumps(manifest["totals"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
