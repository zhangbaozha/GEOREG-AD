"""Controlled geometry-component ablation for GeoReg-AD.

All variants score the same PCDs with the same four normal templates.  Test
labels are loaded only after the score vector for every variant is complete.
The default full geometry configuration is fixed before this script runs;
the script never selects a replacement default from the reported metrics.

The parent invocation starts independent category processes because Open3D's
random generator is process-global.  A category invocation writes concise
per-scan score summaries plus reproducible category metrics; it does not emit
large duplicate full-resolution score archives for every ablation variant.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zlib

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

from pilot import (
    align_gt_by_coordinates,
    interpolate_to_full,
    make_features,
    prepare_templates,
    transform_xyz,
)


DEFAULT_VARIANT = "full_geometry"
VARIANTS: dict[str, dict[str, object]] = {
    "distance_only": {"weights": (1.0, 0.0, 0.0), "templates": 4, "registration": "fgr_icp"},
    "distance_plane": {"weights": (1.0, 0.5, 0.0), "templates": 4, "registration": "fgr_icp"},
    "distance_normal": {"weights": (1.0, 0.0, 0.5), "templates": 4, "registration": "fgr_icp"},
    "full_geometry": {"weights": (1.0, 0.5, 0.5), "templates": 4, "registration": "fgr_icp"},
    "templates_1": {"weights": (1.0, 0.5, 0.5), "templates": 1, "registration": "fgr_icp"},
    "templates_2": {"weights": (1.0, 0.5, 0.5), "templates": 2, "registration": "fgr_icp"},
    "registration_identity": {"weights": (1.0, 0.5, 0.5), "templates": 4, "registration": "identity"},
    "registration_fgr": {"weights": (1.0, 0.5, 0.5), "templates": 4, "registration": "fgr"},
}


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def score_hash(scores: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()


def geometry_cost(
    query_xyz: np.ndarray,
    query_normals: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_normals: np.ndarray,
    h: float,
    weights: tuple[float, float, float],
) -> np.ndarray:
    difference = query_xyz[..., None, :] - candidate_xyz
    euclidean = np.linalg.norm(difference, axis=-1) / h
    plane = np.abs(np.sum(difference * candidate_normals, axis=-1)) / h
    normal = 1.0 - np.abs(np.sum(query_normals[..., None, :] * candidate_normals, axis=-1))
    return weights[0] * euclidean + weights[1] * plane + weights[2] * normal


def retrieve_scores(
    query_xyz: np.ndarray,
    query_normals: np.ndarray,
    template_xyz: np.ndarray,
    template_normals: np.ndarray,
    h: float,
    weights: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    distances, ids = cKDTree(template_xyz).query(
        query_xyz, k=8, distance_upper_bound=8.0 * h, workers=4
    )
    valid = np.isfinite(distances) & (ids < len(template_xyz))
    safe_ids = np.where(valid, ids, 0)
    candidate_xyz = template_xyz[safe_ids]
    candidate_normals = template_normals[safe_ids]
    costs = geometry_cost(query_xyz, query_normals, candidate_xyz, candidate_normals, h, weights)
    costs[~valid] = np.inf
    raw = costs[np.arange(len(query_xyz)), np.argmin(costs, axis=1)]
    raw[~np.isfinite(raw)] = 16.0
    return raw, time.perf_counter() - start


def register_bundle(source_down, source_feat, target_down, target_feat, voxel: float):
    """Return identity, FGR, and FGR+ICP transforms from one FGR run."""
    registration = o3d.pipelines.registration
    started = time.perf_counter()
    try:
        coarse = registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_feat,
            target_feat,
            registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=2.5 * voxel,
                iteration_number=64,
            ),
        )
        fgr_transform = coarse.transformation
        fgr_fitness = float(coarse.fitness)
    except RuntimeError:
        fgr_transform = np.eye(4)
        fgr_fitness = 0.0
    fgr_seconds = time.perf_counter() - started
    icp_started = time.perf_counter()
    fine = registration.registration_icp(
        source_down,
        target_down,
        2.5 * voxel,
        fgr_transform,
        registration.TransformationEstimationPointToPlane(),
        registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return {
        "identity": np.eye(4),
        "fgr": fgr_transform,
        "fgr_icp": fine.transformation,
    }, {
        "fgr_fitness": fgr_fitness,
        "fgr_seconds": fgr_seconds,
        "icp_fitness": float(fine.fitness),
        "icp_inlier_rmse": float(fine.inlier_rmse),
        "icp_seconds": time.perf_counter() - icp_started,
        "total_seconds": time.perf_counter() - started,
    }


def build_libraries(templates: list[Path], voxel: float) -> dict[int, dict[str, object]]:
    libraries: dict[int, dict[str, object]] = {}
    for count in (1, 2, 4):
        # The reference is the first template for every library.  Resetting the
        # seed makes the four-template library use the same preparation policy
        # as the full GeoReg run.
        o3d.utility.random.seed(0)
        libraries[count] = prepare_templates(templates[:count], voxel)
    return libraries


def top_one_percent(scores: np.ndarray) -> float:
    count = max(1, int(math.ceil(0.01 * len(scores))))
    return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))


def point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if not (0 < int(np.count_nonzero(labels)) < len(labels)):
        return {"point_auc": None, "point_ap": None}
    return {
        "point_auc": float(roc_auc_score(labels, scores)),
        "point_ap": float(average_precision_score(labels, scores)),
    }


def run_case(
    *,
    category: str,
    test: Path,
    gt: Path | None,
    is_anomaly: bool,
    libraries: dict[int, dict[str, object]],
    voxel: float,
    reference_scores: Path | None,
) -> tuple[dict[str, object], dict[str, np.ndarray], np.ndarray | None]:
    started = time.perf_counter()
    registration_seed = zlib.crc32(str(test.relative_to(test.parent.parent)).encode()) & 0x7FFFFFFF
    o3d.utility.random.seed(registration_seed)
    full_cloud, test_down, test_feat, preprocessing_seconds = make_features(test, voxel)
    transformations, registration = register_bundle(
        test_down,
        test_feat,
        libraries[4]["reference_down"],
        libraries[4]["reference_feat"],
        voxel,
    )
    down_xyz = np.asarray(test_down.points)
    down_normals = np.asarray(test_down.normals)
    full_xyz = np.asarray(full_cloud.points)
    scores_by_variant: dict[str, np.ndarray] = {}
    retrieval_seconds: dict[str, float] = {}
    interpolation_seconds: dict[str, float] = {}
    for name, setting in VARIANTS.items():
        transformation = transformations[str(setting["registration"])]
        query_xyz = transform_xyz(down_xyz, transformation)
        query_normals = down_normals @ transformation[:3, :3].T
        library = libraries[int(setting["templates"])]
        anchors, retrieved = retrieve_scores(
            query_xyz,
            query_normals,
            library["xyz"],
            library["normals"],
            float(library["h"]),
            tuple(setting["weights"]),
        )
        full_aligned = transform_xyz(full_xyz, transformation)
        scores, interpolated = interpolate_to_full(full_aligned, query_xyz, anchors)
        if not np.isfinite(scores).all() or len(scores) != len(full_xyz):
            raise ValueError(f"invalid scores for {test} / {name}")
        scores_by_variant[name] = scores
        retrieval_seconds[name] = retrieved
        interpolation_seconds[name] = interpolated

    labels: np.ndarray | None = None
    if gt is not None:
        # All candidate scores have been generated before labels are read.
        labels, _ = align_gt_by_coordinates(full_xyz, gt)
    elif not is_anomaly:
        # Normal scans participate in pooled point metrics as all-zero labels,
        # exactly as in the primary full-file protocol.
        labels = np.zeros(len(full_xyz), dtype=np.int8)

    reference_delta: dict[str, float | bool] | None = None
    if reference_scores is not None:
        with np.load(reference_scores) as archive:
            reference = archive["scores"]
        current = scores_by_variant[DEFAULT_VARIANT]
        if len(reference) != len(current):
            raise ValueError(f"reference score length differs: {reference_scores}")
        delta = np.abs(reference.astype(np.float64) - current)
        reference_delta = {
            "max_absolute_difference": float(np.max(delta, initial=0.0)),
            "mean_absolute_difference": float(np.mean(delta)),
            "allclose_1e_minus_9": bool(np.allclose(reference, current, rtol=0.0, atol=1e-9)),
        }

    point_result = {
        name: point_metrics(labels, scores) if labels is not None else {"point_auc": None, "point_ap": None}
        for name, scores in scores_by_variant.items()
    }
    result: dict[str, object] = {
        "category": category,
        "sample": test.stem,
        "test": str(test),
        "gt": str(gt) if gt else None,
        "is_anomaly": bool(is_anomaly),
        "point_gt_valid": labels is not None,
        "full_points": int(len(full_xyz)),
        "anchor_points": int(len(down_xyz)),
        "registration_seed": registration_seed,
        "registration": registration,
        "reference_default_difference": reference_delta,
        "variants": {
            name: {
                "object_top_1pct": top_one_percent(scores),
                "score_sha256": score_hash(scores),
                **point_result[name],
            }
            for name, scores in scores_by_variant.items()
        },
        "timing_seconds": {
            "preprocess": preprocessing_seconds,
            "registration_bundle": registration["total_seconds"],
            "retrieval_by_variant": retrieval_seconds,
            "interpolation_by_variant": interpolation_seconds,
            "total": time.perf_counter() - started,
        },
    }
    return result, scores_by_variant, labels


def summarize_category(
    category: str,
    source_split: str,
    cases: list[dict[str, object]],
    labels: list[np.ndarray],
    scores: dict[str, list[np.ndarray]],
) -> dict[str, object]:
    object_labels = np.asarray([int(bool(case["is_anomaly"])) for case in cases])
    point_labels = np.concatenate(labels)
    summary: dict[str, object] = {
        "category": category,
        "source_split": source_split,
        "test_scans": len(cases),
        "normal_scans": int(np.sum(object_labels == 0)),
        "anomaly_scans": int(np.sum(object_labels == 1)),
        "point_valid_scans": int(sum(bool(case["point_gt_valid"]) for case in cases)),
        "total_evaluated_points": int(len(point_labels)),
        "positive_points": int(np.count_nonzero(point_labels)),
        "variants": {},
        "default_reference": {},
    }
    for name in VARIANTS:
        point_scores = np.concatenate(scores[name])
        object_scores = np.asarray([float(case["variants"][name]["object_top_1pct"]) for case in cases])
        summary["variants"][name] = {
            "pooled_point_auc": float(roc_auc_score(point_labels, point_scores)),
            "pooled_point_ap": float(average_precision_score(point_labels, point_scores)),
            "object_auc": float(roc_auc_score(object_labels, object_scores)),
            "object_ap": float(average_precision_score(object_labels, object_scores)),
        }
    deltas = [case["reference_default_difference"] for case in cases if case["reference_default_difference"]]
    if deltas:
        summary["default_reference"] = {
            "compared_cases": len(deltas),
            "max_absolute_difference": float(max(float(row["max_absolute_difference"]) for row in deltas)),
            "mean_absolute_difference": float(np.mean([float(row["mean_absolute_difference"]) for row in deltas])),
            "allclose_cases": int(sum(bool(row["allclose_1e_minus_9"]) for row in deltas)),
        }
    return summary


def category_run(args: argparse.Namespace, category: str, source_split: str) -> None:
    data_root = args.data_root.resolve()
    category_dir = data_root / category
    templates = sorted((category_dir / "train").glob("*.pcd"))
    tests = sorted((category_dir / "test").glob("*.pcd"))
    if len(templates) != 4 or not tests:
        raise ValueError(f"invalid category layout: {category_dir}")
    if args.limit is not None:
        normals = [path for path in tests if "good" in path.stem]
        anomalies = [path for path in tests if "good" not in path.stem]
        tests = (normals[:1] + anomalies[: max(0, args.limit - 1)])[: args.limit]
        if len(tests) != args.limit:
            raise ValueError(f"unable to create a {args.limit}-scan smoke test for {category}")
    audit = json.loads(args.audit_result.read_text(encoding="utf-8"))
    invalid_gt = {str(row["gt"]) for row in audit["failures"]}
    category_root = args.run_root.resolve() / "results" / category
    atomic_json(category_root / "config.json", {
        "category": category,
        "source_split": source_split,
        "templates": [str(path) for path in templates],
        "voxel": args.voxel,
        "default_variant": DEFAULT_VARIANT,
        "variants": VARIANTS,
        "protocol": "All variants score each PCD before any annotation is loaded. Labels are used after scoring only for metric calculation.",
    })
    libraries = build_libraries(templates, args.voxel)
    atomic_json(category_root / "template_libraries.json", {
        str(count): {
            "templates": [str(path) for path in library["paths"]],
            "anchors": int(len(library["xyz"])),
            "spacing_h": float(library["h"]),
            "seconds": float(library["seconds"]),
        }
        for count, library in libraries.items()
    })
    all_cases: list[dict[str, object]] = []
    labels: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in VARIANTS}
    for index, test in enumerate(tests, start=1):
        is_anomaly = "good" not in test.stem
        candidate_gt = category_dir / "gt" / f"{test.stem}.txt" if is_anomaly else None
        valid_gt = candidate_gt if candidate_gt is not None and str(candidate_gt) not in invalid_gt else None
        reference_score = None
        if args.reference_results is not None:
            candidate = args.reference_results / category / "scores" / f"{test.stem}.npz"
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            reference_score = candidate
        result, case_scores, case_labels = run_case(
            category=category,
            test=test,
            gt=valid_gt,
            is_anomaly=is_anomaly,
            libraries=libraries,
            voxel=args.voxel,
            reference_scores=reference_score,
        )
        atomic_json(category_root / "cases" / f"{test.stem}.json", result)
        all_cases.append(result)
        if case_labels is not None:
            labels.append(case_labels)
            for name in VARIANTS:
                scores[name].append(case_scores[name])
        print(json.dumps({"category": category, "done": index, "of": len(tests), "sample": test.stem}), flush=True)
    summary = summarize_category(category, source_split, all_cases, labels, scores)
    atomic_json(category_root / "summary.json", summary)
    print(json.dumps({"category_complete": category, "default": summary["variants"][DEFAULT_VARIANT]}), flush=True)


def aggregate_category_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {
        "categories": len(summaries),
        "test_scans": int(sum(int(row["test_scans"]) for row in summaries)),
        "normal_scans": int(sum(int(row["normal_scans"]) for row in summaries)),
        "anomaly_scans": int(sum(int(row["anomaly_scans"]) for row in summaries)),
        "point_valid_scans": int(sum(int(row["point_valid_scans"]) for row in summaries)),
        "variants": {},
    }
    for name in VARIANTS:
        output["variants"][name] = {
            metric: float(np.mean([float(row["variants"][name][metric]) for row in summaries]))
            for metric in ("pooled_point_auc", "pooled_point_ap", "object_auc", "object_ap")
        }
    reference = [row["default_reference"] for row in summaries if row.get("default_reference")]
    if reference:
        output["default_reference"] = {
            "compared_cases": int(sum(int(row["compared_cases"]) for row in reference)),
            "max_absolute_difference": float(max(float(row["max_absolute_difference"]) for row in reference)),
            "mean_absolute_difference": float(np.mean([float(row["mean_absolute_difference"]) for row in reference])),
            "allclose_cases": int(sum(int(row["allclose_cases"]) for row in reference)),
        }
    return output


def all_run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.layout_manifest.read_text(encoding="utf-8"))
    selected = [row for row in manifest["categories"] if row["source_split"] == args.source_split]
    if not selected:
        raise ValueError(f"no categories for source split {args.source_split}")
    root = args.run_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run root must be new or empty: {root}")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    atomic_json(root / "run_manifest.json", {
        "data_root": str(args.data_root.resolve()),
        "layout_manifest": str(args.layout_manifest.resolve()),
        "audit_result": str(args.audit_result.resolve()),
        "reference_results": str(args.reference_results.resolve()) if args.reference_results else None,
        "source_split": args.source_split,
        "workers": args.workers,
        "voxel": args.voxel,
        "default_variant": DEFAULT_VARIANT,
        "variants": VARIANTS,
        "test_label_policy": "Scores for all variants are produced before labels are loaded. The fixed default variant is not changed using test metrics.",
    })
    script = Path(__file__).resolve()

    def child(row: dict[str, object]) -> tuple[str, int]:
        category = str(row["adapter_category"])
        command = [sys.executable, "-u", str(script), "--data-root", str(args.data_root),
                   "--layout-manifest", str(args.layout_manifest), "--run-root", str(root),
                   "--audit-result", str(args.audit_result), "--category", category,
                   "--source-split", str(row["source_split"]), "--voxel", str(args.voxel)]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.reference_results:
            command.extend(["--reference-results", str(args.reference_results)])
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        with (root / "logs" / f"{category}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
        return category, code

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = dict(pool.map(child, selected))
    if any(status.values()):
        raise RuntimeError(f"category failures: {status}")
    summaries = [json.loads((root / "results" / str(row["adapter_category"]) / "summary.json").read_text(encoding="utf-8")) for row in selected]
    aggregate = aggregate_category_summaries(summaries)
    report = {
        "protocol": "Shared-scan component ablation. Scores are created before annotation labels are loaded. The fixed default is full geometry, four templates, and FGR+ICP; metrics do not choose a new default.",
        "variants": VARIANTS,
        "default_variant": DEFAULT_VARIANT,
        "source_split": args.source_split,
        "category_status": status,
        "aggregate": aggregate,
        "categories": summaries,
    }
    atomic_json(root / "component_ablation_summary.json", report)
    with (root / "component_ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["variant", "pooled_point_auc", "pooled_point_ap", "object_auc", "object_ap"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, metrics in aggregate["variants"].items():
            writer.writerow({"variant": name, **metrics})
    print(json.dumps({"COMPLETE": str(root), "aggregate": aggregate}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--category", default="all")
    parser.add_argument("--source-split", default="pcd")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--limit", type=int, help="smoke-test scan count; never use for a complete run")
    args = parser.parse_args()
    if args.voxel <= 0 or args.workers < 1 or (args.limit is not None and args.limit < 2):
        parser.error("workers and voxel must be positive")
    if not args.data_root.is_dir() or not args.layout_manifest.is_file() or not args.audit_result.is_file():
        parser.error("data root, layout manifest, or audit result is missing")
    if args.category == "all":
        all_run(args)
    else:
        category_run(args, args.category, args.source_split)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        raise SystemExit(f"FAILED: {error}")
