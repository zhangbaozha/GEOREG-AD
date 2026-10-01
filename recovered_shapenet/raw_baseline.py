"""Full Real3D-AD geometry-only baseline, preserving the pilot raw score.

Per-category process: four normal templates, every eligible test scan, no
counterfactual stage. Labels are used only after all anomaly scores are made.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.metrics import average_precision_score, roc_auc_score

from pilot import (
    align_gt_by_coordinates,
    interpolate_to_full,
    make_features,
    prepare_templates,
    register,
    retrieve,
    transform_xyz,
)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def atomic_scores(path: Path, scores: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, scores=scores, labels=labels)
    temp.replace(path)


def case(prepared: dict, test: Path, gt: Path | None, object_label: int,
         voxel: float, score_path: Path):
    begin = time.perf_counter()
    registration_seed = zlib.crc32(str(test.relative_to(test.parent.parent)).encode()) & 0x7fffffff
    o3d.utility.random.seed(registration_seed)
    full, down, feat, prep_seconds = make_features(test, voxel)
    transform, reg = register(
        down, feat, prepared["reference_down"], prepared["reference_feat"], voxel
    )
    query_xyz = transform_xyz(np.asarray(down.points), transform)
    query_normals = np.asarray(down.normals) @ transform[:3, :3].T
    matches = retrieve(
        query_xyz, query_normals, prepared["xyz"], prepared["normals"], prepared["h"]
    )
    full_xyz = np.asarray(full.points)
    full_aligned = transform_xyz(full_xyz, transform)
    scores, interp_seconds = interpolate_to_full(
        full_aligned, query_xyz, matches["raw"]
    )
    if gt is None:
        labels = np.zeros(len(full_xyz), dtype=np.int8)
        gt_audit = None
    else:
        labels, gt_audit = align_gt_by_coordinates(full_xyz, gt)
    if not np.isfinite(scores).all() or len(scores) != len(labels):
        raise ValueError(f"Invalid scores for {test}")
    count = max(1, int(np.ceil(0.01 * len(scores))))
    score_top = float(np.mean(np.partition(scores, -count)[-count:]))
    point_auc = point_ap = None
    point_gt_valid = gt is not None or object_label == 0
    if point_gt_valid and 0 < np.count_nonzero(labels) < len(labels):
        point_auc = float(roc_auc_score(labels, scores))
        point_ap = float(average_precision_score(labels, scores))
    atomic_scores(score_path, scores, labels if point_gt_valid else labels[:0])
    return {
        "category": test.parent.parent.name,
        "sample": test.stem,
        "test": str(test),
        "gt": str(gt) if gt else None,
        "is_anomaly": bool(object_label),
        "point_gt_valid": point_gt_valid,
        "registration_seed": registration_seed,
        "transformation": transform.tolist(),
        "full_points": len(scores),
        "positive_points": int(np.count_nonzero(labels)) if point_gt_valid else None,
        "anchor_points": len(query_xyz),
        "template_points": len(prepared["xyz"]),
        "spacing_h": prepared["h"],
        "registration": reg,
        "unmatched_fraction": float(np.mean(matches["unmatched"])),
        "raw_mean": float(np.mean(scores)),
        "raw_top1pct": score_top,
        "raw_point_auc": point_auc,
        "raw_point_ap": point_ap,
        "gt_audit": gt_audit,
        "score_file": str(score_path),
        "timing_seconds": {
            "preprocess": prep_seconds,
            "registration": reg["seconds"],
            "retrieval": matches["seconds"],
            "interpolation": interp_seconds,
            "total": time.perf_counter() - begin,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only")
    args = parser.parse_args()

    category_dir = args.data_root / args.category
    templates = sorted((category_dir / "train").glob("*.pcd"))
    tests = sorted((category_dir / "test").glob("*.pcd"))
    if len(templates) != 4 or not tests:
        raise ValueError(f"Expected four train templates and test scans: {category_dir}")
    audit = json.loads(args.audit_result.read_text(encoding="utf-8"))
    excluded = {x["gt"] for x in audit["failures"]}
    category_out = args.output_dir / args.category
    category_out.mkdir(parents=True, exist_ok=True)
    atomic_json(category_out / "config.json", {
        "category": args.category,
        "templates": [str(p) for p in templates],
        "voxel": args.voxel,
        "test_count": len(tests),
        "limit": args.limit,
        "invalid_point_gt": sorted(x for x in excluded if f"/{args.category}/" in x),
        "method": "pilot raw: FPFH FGR + point-to-plane ICP; 4 pooled normal templates; 8-NN geometry cost; 3-NN interpolation",
        "object_aggregation": "mean of top ceil(1% * N) full-resolution point scores",
    })
    eligible = []
    for test in tests:
        object_label = int("good" not in test.stem)
        if not object_label:
            gt = None
        else:
            gt = category_dir / "gt" / (test.stem + ".txt")
            if str(gt) in excluded:
                print(json.dumps({"point_gt_invalid": str(test), "reason": "documented GT/PCD mismatch"}), flush=True)
                gt = None
            elif not gt.exists():
                raise FileNotFoundError(gt)
        eligible.append((test, gt, object_label))
    if args.limit is not None:
        # Include both label types in a small smoke test.
        normal = [x for x in eligible if x[2] == 0]
        anomaly = [x for x in eligible if x[2] == 1]
        eligible = (normal[:1] + anomaly[: max(0, args.limit - 1)])[: args.limit]
    incomplete = [
        (test, gt, object_label) for test, gt, object_label in eligible
        if not ((category_out / "cases" / f"{test.stem}.json").exists()
                and (category_out / "scores" / f"{test.stem}.npz").exists())
    ]
    print(json.dumps({"category": args.category, "eligible": len(eligible), "pending": len(incomplete)}), flush=True)
    if not incomplete:
        return
    o3d.utility.random.seed(0)
    prepared = prepare_templates(templates, args.voxel)
    atomic_json(category_out / "template_preparation.json", {
        "templates": [str(p) for p in templates],
        "h": prepared["h"],
        "template_points": len(prepared["xyz"]),
        "registrations": prepared["registrations"],
        "seconds": prepared["seconds"],
    })
    print(json.dumps({"category": args.category, "template_seconds": prepared["seconds"]}), flush=True)
    for index, (test, gt, object_label) in enumerate(incomplete, 1):
        score_path = category_out / "scores" / f"{test.stem}.npz"
        case_path = category_out / "cases" / f"{test.stem}.json"
        try:
            result = case(prepared, test, gt, object_label, args.voxel, score_path)
            atomic_json(case_path, result)
            print(json.dumps({"done": index, "of": len(incomplete), "category": args.category,
                              "sample": test.stem, "seconds": result["timing_seconds"]["total"]}), flush=True)
        except Exception as exc:
            failure = {"category": args.category, "sample": test.stem,
                       "error": repr(exc), "traceback": traceback.format_exc()}
            atomic_json(category_out / "failures" / f"{test.stem}.json", failure)
            print(json.dumps(failure), flush=True)
    print(json.dumps({"category_finished": args.category}), flush=True)


if __name__ == "__main__":
    main()
