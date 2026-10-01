"""Full Real3D-AD PCD↔GT coordinate-integrity audit (read-only dataset)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pilot import align_gt_by_coordinates, xyz_from_pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all GT files")
    args = parser.parse_args()

    start = time.perf_counter()
    records = []
    failures = []
    gt_files = sorted(args.data_root.glob("*/gt/*.txt"))
    if args.limit:
        gt_files = gt_files[: args.limit]
    for index, gt_path in enumerate(gt_files, 1):
        pcd_path = gt_path.parent.parent / "test" / f"{gt_path.stem}.pcd"
        try:
            if not pcd_path.exists():
                raise FileNotFoundError(pcd_path)
            xyz = xyz_from_pcd(pcd_path)
            _, audit = align_gt_by_coordinates(xyz, gt_path)
            records.append({
                "category": gt_path.parent.parent.name,
                "sample": gt_path.stem,
                "points": len(xyz),
                **audit,
            })
        except Exception as exc:
            failures.append({"gt": str(gt_path), "error": repr(exc)})
        if index % 25 == 0 or index == len(gt_files):
            print(
                f"AUDIT {index}/{len(gt_files)} passed={len(records)} failed={len(failures)}",
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({
                    "processed": index,
                    "total": len(gt_files),
                    "passed": len(records),
                    "failed": len(failures),
                    "max_coordinate_error": max(
                        (x["max_sorted_coordinate_error"] for x in records),
                        default=None,
                    ),
                    "mean_rowwise_mismatch": (
                        sum(x["rowwise_mismatch_fraction"] for x in records) / len(records)
                        if records else None
                    ),
                    "seconds": time.perf_counter() - start,
                    "records": records,
                    "failures": failures,
                }, indent=2),
                encoding="utf-8",
            )
    if failures:
        raise SystemExit(f"{len(failures)} dataset integrity failures")


if __name__ == "__main__":
    main()
