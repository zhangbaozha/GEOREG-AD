"""Evaluate object-score aggregation choices from immutable GeoReg score archives.

The detector outputs full-resolution point scores once.  This script changes
only the object-level aggregation, so it can compare robust top-tail choices
without rerunning registration or reading any ground-truth point labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


AGGREGATIONS: tuple[tuple[str, float | None], ...] = (
    ("maximum", None),
    ("top_0_1pct", 0.001),
    ("top_1pct", 0.01),
    ("top_5pct", 0.05),
    ("mean_all_points", 1.0),
)


def object_scores(scores: np.ndarray) -> dict[str, float]:
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("score archive must contain a nonempty finite 1-D array")
    output: dict[str, float] = {}
    for name, fraction in AGGREGATIONS:
        if fraction is None:
            output[name] = float(scores.max())
            continue
        if fraction == 1.0:
            output[name] = float(scores.mean())
            continue
        count = max(1, math.ceil(len(scores) * fraction))
        # partition selects the tail without a full sort; it does not alter the
        # immutable archive because np.load returns a new in-memory array.
        tail = np.partition(scores, len(scores) - count)[-count:]
        output[name] = float(tail.mean())
    return output


def macro(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"categories": len(rows)}
    for name, _ in AGGREGATIONS:
        result[name] = {
            "object_auc": float(np.mean([float(row[f"{name}_object_auc"]) for row in rows])),
            "object_ap": float(np.mean([float(row[f"{name}_object_ap"]) for row in rows])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-name", default="shapenet_full_summary.json")
    parser.add_argument("--cases-name", default="shapenet_full_cases.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output_dir.resolve()
    summary = json.loads((root / args.summary_name).read_text(encoding="utf-8"))
    cases = json.loads((root / args.cases_name).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("unexpected case artifact")
    groups: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        groups.setdefault(str(case["category"]), []).append(case)

    expected_categories = {
        str(row.get("adapter_category", row.get("category"))): row
        for row in summary["categories"]
    }
    if set(groups) != set(expected_categories):
        raise ValueError("case and summary category sets differ")
    rows: list[dict[str, object]] = []
    for category in sorted(groups):
        case_rows = groups[category]
        labels = np.asarray([int(bool(case["is_anomaly"])) for case in case_rows])
        scores_by_aggregation = {name: [] for name, _ in AGGREGATIONS}
        for case in case_rows:
            with np.load(str(case["score_file"])) as archive:
                scores = archive["scores"]
            for name, value in object_scores(scores).items():
                scores_by_aggregation[name].append(value)
        source = expected_categories[category].get("source_split", "all")
        row: dict[str, object] = {
            "adapter_category": category,
            "source_split": source,
            "test_scans": len(case_rows),
            "normal_scans": int(np.sum(labels == 0)),
            "anomaly_scans": int(np.sum(labels == 1)),
        }
        for name, _ in AGGREGATIONS:
            values = np.asarray(scores_by_aggregation[name])
            row[f"{name}_object_auc"] = float(roc_auc_score(labels, values))
            row[f"{name}_object_ap"] = float(average_precision_score(labels, values))
        rows.append(row)
    source_splits = {str(row["source_split"]) for row in rows}
    buckets = {"all_categories": rows}
    if "pcd" in source_splits:
        buckets = {
            "official_pcd_40": [row for row in rows if row["source_split"] == "pcd"],
            "new_pcd_extension": [row for row in rows if row["source_split"] == "new_pcd"],
            "all_available_52": rows,
        }
    report = {
        "protocol": "Post-hoc object aggregation only. All variants use the same archived full-resolution GeoReg point scores, scans, and object labels; no registration or point-level score is rerun.",
        "aggregations": {name: ("maximum" if fraction is None else f"mean of largest ceil({fraction} * N) scores") for name, fraction in AGGREGATIONS},
        "aggregates": {name: macro(rows) for name, rows in buckets.items()},
        "categories": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "object_aggregation_sensitivity.json"
    csv_path = output / "object_aggregation_sensitivity.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"aggregates": report["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
