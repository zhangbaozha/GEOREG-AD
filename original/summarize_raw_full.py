"""Recompute category-pooled metrics from archived float64 raw point scores."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from raw_baseline import atomic_json


def summarize(root):
    summaries = []
    all_cases = []
    for cat in sorted(p for p in root.iterdir() if p.is_dir()):
        config = json.loads((cat / "config.json").read_text())
        cases = [json.loads(p.read_text()) for p in sorted((cat / "cases").glob("*.json"))]
        if len(cases) != config["test_count"] or config["limit"] is not None:
            raise ValueError(f"Incomplete category {cat.name}: {len(cases)}/{config['test_count']}")
        labels, scores = [], []
        invalid = []
        for item in cases:
            with np.load(item["score_file"]) as data:
                s, y = data["scores"], data["labels"]
            if not np.isfinite(s).all() or len(s) != item["full_points"]:
                raise ValueError(item["score_file"])
            if not item["point_gt_valid"]:
                if len(y):
                    raise ValueError("Invalid GT must be archived as an empty label array")
                invalid.append(item["sample"])
                continue
            if len(s) != len(y) or int(y.sum()) != item["positive_points"]:
                raise ValueError("Score/label integrity mismatch")
            labels.append(y)
            scores.append(s)
        y = np.concatenate(labels)
        s = np.concatenate(scores)
        del labels, scores
        object_y = np.array([int(x["is_anomaly"]) for x in cases])
        object_s = np.array([x["raw_top1pct"] for x in cases])
        row = {
            "category": cat.name,
            "test_scans": len(cases),
            "normal_scans": int(np.sum(object_y == 0)),
            "anomaly_scans": int(np.sum(object_y)),
            "point_valid_scans": len(cases) - len(invalid),
            "point_gt_exclusions": invalid,
            "total_evaluated_points": len(y),
            "positive_points": int(y.sum()),
            "point_prevalence": float(y.mean()),
            "pooled_point_auc_raw": float(roc_auc_score(y, s)),
            "pooled_point_ap_raw": float(average_precision_score(y, s)),
            "object_auc_raw": float(roc_auc_score(object_y, object_s)),
            "object_ap_raw": float(average_precision_score(object_y, object_s)),
            "mean_per_anomaly_point_auc_raw": float(np.mean([x["raw_point_auc"] for x in cases if x["raw_point_auc"] is not None])),
            "mean_per_anomaly_point_ap_raw": float(np.mean([x["raw_point_ap"] for x in cases if x["raw_point_ap"] is not None])),
            "mean_icp_fitness": float(np.mean([x["registration"]["icp_fitness"] for x in cases])),
            "mean_unmatched_fraction": float(np.mean([x["unmatched_fraction"] for x in cases])),
            "median_case_seconds": float(np.median([x["timing_seconds"]["total"] for x in cases])),
        }
        del y, s
        atomic_json(cat / "summary.json", row)
        summaries.append(row)
        all_cases.extend(cases)
        print(json.dumps(row), flush=True)
    keys = ["pooled_point_auc_raw", "pooled_point_ap_raw", "object_auc_raw", "object_ap_raw",
            "mean_per_anomaly_point_auc_raw", "mean_per_anomaly_point_ap_raw"]
    report = {
        "protocol": "All test files; per-category full-resolution points pooled across normal and anomalous scans; macro-average over 12 categories. Five invalid point GT files omitted only from point metrics. No per-scan score normalization. AP uses sklearn tie-aware average_precision_score.",
        "categories": summaries,
        "macro_average": {key: float(np.mean([x[key] for x in summaries])) for key in keys},
        "total_test_scans": sum(x["test_scans"] for x in summaries),
        "total_point_valid_scans": sum(x["point_valid_scans"] for x in summaries),
        "total_points": sum(x["total_evaluated_points"] for x in summaries),
        "seed_policy": "Open3D seed 0 before each category's template preparation; CRC32(test-relative-path) per test registration; not a multi-seed study",
    }
    atomic_json(root.parent / "raw_full_summary.json", report)
    atomic_json(root.parent / "raw_full_cases.json", all_cases)
    with (root.parent / "raw_full_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [x for x in summaries[0] if x != "point_gt_exclusions"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"macro_average": report["macro_average"], "total_scans": report["total_test_scans"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    summarize(parser.parse_args().results)
