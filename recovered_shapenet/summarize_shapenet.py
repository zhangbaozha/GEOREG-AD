"""Summarize archived GeoReg scores for the Anomaly-ShapeNet adapter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def atomic_json(path: Path, obj: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


METRICS = (
    "pooled_point_auc_raw",
    "pooled_point_ap_raw",
    "object_auc_raw",
    "object_ap_raw",
    "mean_per_anomaly_point_auc_raw",
    "mean_per_anomaly_point_ap_raw",
)


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "categories": len(rows),
        "test_scans": sum(int(row["test_scans"]) for row in rows),
        "normal_scans": sum(int(row["normal_scans"]) for row in rows),
        "anomaly_scans": sum(int(row["anomaly_scans"]) for row in rows),
        "point_valid_scans": sum(int(row["point_valid_scans"]) for row in rows),
        "total_evaluated_points": sum(int(row["total_evaluated_points"]) for row in rows),
        "positive_points": sum(int(row["positive_points"]) for row in rows),
        "macro_average": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    mapping = {
        row["adapter_category"]: row
        for row in json.loads(args.layout_manifest.read_text(encoding="utf-8"))["categories"]
    }
    rows: list[dict[str, object]] = []
    all_cases: list[dict[str, object]] = []
    for category in sorted(path for path in results.iterdir() if path.is_dir()):
        config = json.loads((category / "config.json").read_text(encoding="utf-8"))
        cases = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted((category / "cases").glob("*.json"))]
        failures = list((category / "failures").glob("*.json"))
        if failures or len(cases) != int(config["test_count"]):
            raise ValueError(f"incomplete category {category.name}: cases={len(cases)} failures={len(failures)}")
        point_labels: list[np.ndarray] = []
        point_scores: list[np.ndarray] = []
        invalid: list[str] = []
        for case in cases:
            with np.load(case["score_file"]) as score_archive:
                scores = score_archive["scores"]
                labels = score_archive["labels"]
            if not np.isfinite(scores).all() or len(scores) != int(case["full_points"]):
                raise ValueError(f"invalid score archive: {case['score_file']}")
            if not case["point_gt_valid"]:
                if len(labels):
                    raise ValueError("invalid point GT must be archived with empty labels")
                invalid.append(str(case["sample"]))
                continue
            if len(scores) != len(labels) or int(labels.sum()) != int(case["positive_points"]):
                raise ValueError(f"label integrity mismatch: {case['sample']}")
            point_labels.append(labels)
            point_scores.append(scores)
        labels = np.concatenate(point_labels)
        scores = np.concatenate(point_scores)
        object_labels = np.asarray([int(case["is_anomaly"]) for case in cases])
        object_scores = np.asarray([float(case["raw_top1pct"]) for case in cases])
        source = mapping[category.name]
        row = {
            "adapter_category": category.name,
            "source_split": source["source_split"],
            "source_category": source["source_category"],
            "test_scans": len(cases),
            "normal_scans": int(np.sum(object_labels == 0)),
            "anomaly_scans": int(np.sum(object_labels == 1)),
            "point_valid_scans": len(cases) - len(invalid),
            "point_gt_exclusions": invalid,
            "total_evaluated_points": len(labels),
            "positive_points": int(labels.sum()),
            "point_prevalence": float(labels.mean()),
            "pooled_point_auc_raw": float(roc_auc_score(labels, scores)),
            "pooled_point_ap_raw": float(average_precision_score(labels, scores)),
            "object_auc_raw": float(roc_auc_score(object_labels, object_scores)),
            "object_ap_raw": float(average_precision_score(object_labels, object_scores)),
            "mean_per_anomaly_point_auc_raw": float(np.mean([
                case["raw_point_auc"] for case in cases if case["raw_point_auc"] is not None
            ])),
            "mean_per_anomaly_point_ap_raw": float(np.mean([
                case["raw_point_ap"] for case in cases if case["raw_point_ap"] is not None
            ])),
            "mean_icp_fitness": float(np.mean([case["registration"]["icp_fitness"] for case in cases])),
            "mean_unmatched_fraction": float(np.mean([case["unmatched_fraction"] for case in cases])),
            "median_case_seconds": float(np.median([case["timing_seconds"]["total"] for case in cases])),
        }
        atomic_json(category / "summary.json", row)
        rows.append(row)
        all_cases.extend(cases)
        print(json.dumps(row), flush=True)

    original = [row for row in rows if row["source_split"] == "pcd"]
    extension = [row for row in rows if row["source_split"] == "new_pcd"]
    report = {
        "protocol": "Four normal templates per category; FPFH FGR plus point-to-plane ICP; pooled normal surface geometry cost; full-resolution 3-NN interpolation; top-1% object aggregate. Evaluation scores every adapter test PCD. P-AUROC/AP pool all full-resolution points per category; object AUROC/AP use all category test PCDs; reported aggregates are unweighted category means. No per-scan score normalization.",
        "voxel_policy": "fixed voxel width supplied by the run command; selected before scoring and independent of test labels",
        "categories": rows,
        "aggregates": {
            "official_pcd_40": aggregate(original),
            "new_pcd_extension": aggregate(extension),
            "all_available_52": aggregate(rows),
        },
        "total_test_scans": sum(int(row["test_scans"]) for row in rows),
        "total_point_valid_scans": sum(int(row["point_valid_scans"]) for row in rows),
        "seed_policy": "Open3D seed 0 before each category template preparation; CRC32(test-relative-path) before each test registration; one fixed configuration, not a multi-seed study.",
    }
    run_root = results.parent
    atomic_json(run_root / "shapenet_full_summary.json", report)
    atomic_json(run_root / "shapenet_full_cases.json", all_cases)
    with (run_root / "shapenet_full_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [key for key in rows[0] if key != "point_gt_exclusions"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"aggregates": report["aggregates"], "total_scans": report["total_test_scans"]}), flush=True)


if __name__ == "__main__":
    main()
