"""Summarize recorded GeoReg per-scan timings without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    fields = sorted({key for case in cases for key in case["timing_seconds"]})
    result: dict[str, object] = {"scans": len(cases)}
    for field in fields:
        values = np.asarray([float(case["timing_seconds"].get(field, 0.0)) for case in cases])
        result[field] = {
            "median_seconds": float(np.median(values)),
            "p90_seconds": float(np.quantile(values, 0.9)),
            "mean_seconds": float(values.mean()),
            "sum_cpu_hours": float(values.sum() / 3600.0),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    groups: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        groups.setdefault(str(case["category"]), []).append(case)
    report = {
        "name": args.name,
        "interpretation": "Times are summed process CPU-seconds recorded by the inference code. They exclude template preparation, queueing, storage transfer, and parallel wall-clock scheduling.",
        "all_categories": summarize(cases),
        "per_category": {category: summarize(group) for category, group in sorted(groups.items())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "run_timing_summary.json"
    csv_path = args.output_dir / "run_timing_summary.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["category", "scans"]
    timing_fields = sorted(report["all_categories"].keys() - {"scans"})
    for timing in timing_fields:
        fields.extend([f"{timing}_median_seconds", f"{timing}_p90_seconds", f"{timing}_mean_seconds", f"{timing}_sum_cpu_hours"])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for category, stats in report["per_category"].items():
            row: dict[str, object] = {"category": category, "scans": stats["scans"]}
            for timing in timing_fields:
                for metric, value in stats[timing].items():
                    row[f"{timing}_{metric}"] = value
            writer.writerow(row)
    print(json.dumps(report["all_categories"], indent=2))


if __name__ == "__main__":
    main()
