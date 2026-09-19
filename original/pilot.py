"""Small, reproducible geometry-only Real3D-AD pilot.

This is a feasibility / mechanism pilot, NOT the final 12-class result. It
compares raw retrieval cost with a same-candidate annular counterfactual under
one shared registration, sampling, and interpolation pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def xyz_from_pcd(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid XYZ in {path}")
    return xyz


def align_gt_by_coordinates(xyz: np.ndarray, gt_path: Path, tol: float = 1e-5):
    """Map unordered GT rows to PCD rows; fail closed on ambiguous coordinates."""
    gt = np.loadtxt(gt_path, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] < 4 or len(gt) != len(xyz):
        raise ValueError(f"GT shape/count mismatch: {gt_path}")
    ref = gt[:, :3]
    labels = gt[:, 3].astype(np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Nonbinary GT labels: {gt_path}")
    order_xyz = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    order_gt = np.lexsort((ref[:, 2], ref[:, 1], ref[:, 0]))
    error = np.linalg.norm(xyz[order_xyz] - ref[order_gt], axis=1)
    if np.max(error, initial=0.0) > tol:
        raise ValueError(
            f"Coordinate sort mismatch {gt_path}: max={error.max():.8g} > {tol}"
        )
    # Duplicate or near-duplicate points with conflicting labels are not
    # safely distinguishable by coordinates alone.
    sorted_ref = ref[order_gt]
    neighbor_gap = np.linalg.norm(np.diff(sorted_ref, axis=0), axis=1)
    ambiguous = (neighbor_gap <= tol) & (np.diff(labels[order_gt]) != 0)
    if np.any(ambiguous):
        raise ValueError(f"Conflicting labels at duplicate/nearby GT points: {gt_path}")
    aligned = np.empty(len(xyz), dtype=np.int8)
    aligned[order_xyz] = labels[order_gt]
    rowwise_mismatch = float(np.mean(np.linalg.norm(xyz - ref, axis=1) > tol))
    return aligned, {
        "max_sorted_coordinate_error": float(np.max(error, initial=0.0)),
        "rowwise_mismatch_fraction": rowwise_mismatch,
        "positive_points": int(np.count_nonzero(aligned)),
    }


def make_features(path: Path, voxel: float):
    start = time.perf_counter()
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise ValueError(f"Empty point cloud: {path}")
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.5 * voxel, max_nn=32)
    )
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100),
    )
    return cloud, down, feat, time.perf_counter() - start


def register(source_down, source_feat, target_down, target_feat, voxel: float):
    reg = o3d.pipelines.registration
    begin = time.perf_counter()
    try:
        coarse = reg.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_feat,
            target_feat,
            reg.FastGlobalRegistrationOption(
                maximum_correspondence_distance=2.5 * voxel,
                iteration_number=64,
            ),
        )
        init = coarse.transformation
        coarse_fitness = float(coarse.fitness)
    except RuntimeError:
        init = np.eye(4)
        coarse_fitness = 0.0
    fine = reg.registration_icp(
        source_down,
        target_down,
        2.5 * voxel,
        init,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=50),
    )
    return fine.transformation, {
        "coarse_fitness": coarse_fitness,
        "icp_fitness": float(fine.fitness),
        "icp_inlier_rmse": float(fine.inlier_rmse),
        "seconds": time.perf_counter() - begin,
    }


def transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def geometry_cost(p, n, q, nq, h):
    diff = p[..., None, :] - q
    euclidean = np.linalg.norm(diff, axis=-1) / h
    plane = np.abs(np.sum(diff * nq, axis=-1)) / h
    normal = 1 - np.abs(np.sum(n[..., None, :] * nq, axis=-1))
    return euclidean + 0.5 * plane + 0.5 * normal


def retrieve(query_xyz, query_normals, template_xyz, template_normals, h):
    begin = time.perf_counter()
    tree = cKDTree(template_xyz)
    distances, ids = tree.query(
        query_xyz, k=8, distance_upper_bound=8 * h, workers=4
    )
    valid = np.isfinite(distances) & (ids < len(template_xyz))
    safe_ids = np.where(valid, ids, 0)
    cand_xyz = template_xyz[safe_ids]
    cand_normals = template_normals[safe_ids]
    costs = geometry_cost(query_xyz, query_normals, cand_xyz, cand_normals, h)
    costs[~valid] = np.inf
    best = np.argmin(costs, axis=1)
    raw = costs[np.arange(len(query_xyz)), best]
    unmatched = ~np.isfinite(raw)
    # Fixed geometry-only pilot fallback, not a learned/calibrated anomaly score.
    raw[unmatched] = 16.0
    matched_xyz = cand_xyz[np.arange(len(query_xyz)), best]
    return {
        "raw": raw,
        "matched_xyz": matched_xyz,
        "candidate_xyz": cand_xyz,
        "candidate_normals": cand_normals,
        "valid": valid,
        "unmatched": unmatched,
        "seconds": time.perf_counter() - begin,
    }


def counterfactual(query_xyz, query_normals, matches, h):
    begin = time.perf_counter()
    tree = cKDTree(query_xyz)
    inner, outer, eta, epsilon = 2 * h, 8 * h, 2 * h, 3 * h
    raw = matches["raw"]
    disp = matches["matched_xyz"] - query_xyz
    result = raw.copy()
    supported = np.zeros(len(raw), dtype=bool)
    purity = np.zeros(len(raw), dtype=np.float32)
    correction_norm = np.zeros(len(raw), dtype=np.float32)
    for i, neighbors in enumerate(tree.query_ball_point(query_xyz, outer, workers=1)):
        if matches["unmatched"][i]:
            continue
        annulus = [
            j for j in neighbors
            if j != i and not matches["unmatched"][j]
            and np.linalg.norm(query_xyz[j] - query_xyz[i]) >= inner
            and abs(np.dot(query_normals[j], query_normals[i])) >= 0.7
        ]
        if len(annulus) < 8:
            continue
        vectors = disp[annulus]
        center = np.median(vectors, axis=0)
        keep = np.zeros(len(annulus), dtype=bool)
        for _ in range(2):
            keep = np.linalg.norm(vectors - center, axis=1) <= eta
            if np.count_nonzero(keep) < 8:
                break
            center = np.median(vectors[keep], axis=0)
        purity[i] = np.count_nonzero(keep) / len(annulus)
        if np.count_nonzero(keep) < 8 or purity[i] < 0.6:
            continue
        magnitude = np.linalg.norm(center)
        if magnitude > epsilon:
            center *= epsilon / magnitude
        correction_norm[i] = np.linalg.norm(center)
        valid = matches["valid"][i]
        if not np.any(valid):
            continue
        corrected = geometry_cost(
            query_xyz[i] + center,
            query_normals[i],
            matches["candidate_xyz"][i],
            matches["candidate_normals"][i],
            h,
        )
        result[i] = min(raw[i], np.min(corrected[valid]))
        supported[i] = True
    return result, supported, purity, correction_norm, time.perf_counter() - begin


def interpolate_to_full(full_xyz, anchor_xyz, anchor_scores):
    begin = time.perf_counter()
    tree = cKDTree(anchor_xyz)
    distances, ids = tree.query(full_xyz, k=3, workers=4)
    weights = 1.0 / np.maximum(distances, 1e-8)
    weights /= np.sum(weights, axis=1, keepdims=True)
    scores = np.sum(anchor_scores[ids] * weights, axis=1)
    return scores, time.perf_counter() - begin


def auc_ap(labels, scores):
    positives = int(np.count_nonzero(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None, None
    ranks = rankdata(scores, method="average")
    auc = (np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    ap = np.sum(precision[sorted_labels == 1]) / positives
    return float(auc), float(ap)


def unique_base_samples(paths: list[Path]) -> list[Path]:
    """Avoid counting a scan and its *_cut/*_copy variant as two scans."""
    seen = set()
    selected = []
    for path in paths:
        base = path.stem.split("_", 1)[0]
        if base not in seen:
            seen.add(base)
            selected.append(path)
    return selected


def prepare_templates(template_paths: list[Path], voxel: float):
    """Align normal training scans to the first scan and pool their anchors."""
    begin = time.perf_counter()
    _, reference_down, reference_feat, _ = make_features(template_paths[0], voxel)
    reference_xyz = np.asarray(reference_down.points)
    nearest_spacing = cKDTree(reference_xyz).query(
        reference_xyz, k=2, workers=4
    )[0][:, 1]
    h = float(np.median(nearest_spacing))
    all_xyz = [reference_xyz]
    all_normals = [np.asarray(reference_down.normals)]
    registrations = []
    for path in template_paths[1:]:
        _, down, feat, _ = make_features(path, voxel)
        transform, registration = register(
            down, feat, reference_down, reference_feat, voxel
        )
        all_xyz.append(transform_xyz(np.asarray(down.points), transform))
        all_normals.append(np.asarray(down.normals) @ transform[:3, :3].T)
        registrations.append({"template": str(path), **registration})
    return {
        "paths": template_paths,
        "reference_down": reference_down,
        "reference_feat": reference_feat,
        "xyz": np.concatenate(all_xyz),
        "normals": np.concatenate(all_normals),
        "h": h,
        "registrations": registrations,
        "seconds": time.perf_counter() - begin,
    }


def run_case(prepared, test_path: Path, gt_path: Path | None, voxel: float):
    full_cloud, test_down, test_feat, t_test = make_features(test_path, voxel)
    transform, registration = register(
        test_down, test_feat,
        prepared["reference_down"], prepared["reference_feat"], voxel
    )
    query_xyz = transform_xyz(np.asarray(test_down.points), transform)
    query_normals = np.asarray(test_down.normals) @ transform[:3, :3].T
    template_xyz = prepared["xyz"]
    template_normals = prepared["normals"]
    h = prepared["h"]
    matches = retrieve(query_xyz, query_normals, template_xyz, template_normals, h)
    filtered, supported, purity, correction_norm, t_filter = counterfactual(
        query_xyz, query_normals, matches, h
    )
    full_xyz = np.asarray(full_cloud.points)
    full_aligned = transform_xyz(full_xyz, transform)
    raw_full, t_interp = interpolate_to_full(
        full_aligned, query_xyz, matches["raw"]
    )
    filtered_full, _ = interpolate_to_full(full_aligned, query_xyz, filtered)
    if gt_path is None:
        labels = np.zeros(len(full_xyz), dtype=np.int8)
        gt_audit = None
    else:
        labels, gt_audit = align_gt_by_coordinates(full_xyz, gt_path)
    raw_auc, raw_ap = auc_ap(labels, raw_full)
    cf_auc, cf_ap = auc_ap(labels, filtered_full)
    mixture_sensitivity = []
    for cf_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = (1.0 - cf_weight) * raw_full + cf_weight * filtered_full
        auc, ap = auc_ap(labels, scores)
        mixture_sensitivity.append({
            "counterfactual_weight": cf_weight,
            "point_auc": auc,
            "point_ap": ap,
        })
    positive = labels == 1
    negative = ~positive
    top = max(1, int(np.ceil(0.01 * len(full_xyz))))
    return {
        "templates": [str(path) for path in prepared["paths"]],
        "test": str(test_path),
        "gt": str(gt_path) if gt_path else None,
        "full_points": len(full_xyz),
        "anchor_points": len(query_xyz),
        "template_points": len(template_xyz),
        "spacing_h": h,
        "registration": registration,
        "unmatched_fraction": float(np.mean(matches["unmatched"])),
        "supported_fraction": float(np.mean(supported)),
        "mean_support_purity": float(np.mean(purity[supported])) if np.any(supported) else None,
        "mean_correction_norm": float(np.mean(correction_norm[supported])) if np.any(supported) else None,
        "score_reduction_normal_mean": float(np.mean((raw_full - filtered_full)[negative])),
        "score_reduction_anomaly_mean": float(np.mean((raw_full - filtered_full)[positive])) if np.any(positive) else None,
        "raw_mean": float(np.mean(raw_full)),
        "counterfactual_mean": float(np.mean(filtered_full)),
        "raw_top1pct": float(np.mean(np.partition(raw_full, -top)[-top:])),
        "counterfactual_top1pct": float(
            np.mean(np.partition(filtered_full, -top)[-top:])
        ),
        "raw_point_auc": raw_auc,
        "counterfactual_point_auc": cf_auc,
        "raw_point_ap": raw_ap,
        "counterfactual_point_ap": cf_ap,
        "mixture_sensitivity": mixture_sensitivity,
        "gt_audit": gt_audit,
        "timing_seconds": {
            "template_preprocess_shared": prepared["seconds"],
            "test_preprocess": t_test,
            "registration": registration["seconds"],
            "retrieval": matches["seconds"],
            "counterfactual": t_filter,
            "one_interpolation": t_interp,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--voxel", type=float, default=0.2)
    parser.add_argument("--normal-count", type=int, default=1)
    parser.add_argument("--abnormal-count", type=int, default=1)
    parser.add_argument("--template-count", type=int, default=1)
    parser.add_argument("--unique-test-base", action="store_true")
    parser.add_argument(
        "--audit-result", type=Path,
        help="Exclude samples with a documented PCD/GT integrity failure",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    category = args.data_root / args.category
    templates = sorted((category / "train").glob("*.pcd"))
    normals = sorted(p for p in (category / "test").glob("*.pcd") if "good" in p.stem)
    abnormal = sorted(p for p in (category / "test").glob("*.pcd") if "good" not in p.stem)
    if args.audit_result:
        audit = json.loads(args.audit_result.read_text(encoding="utf-8"))
        excluded = {x["gt"] for x in audit["failures"]}
        abnormal = [
            p for p in abnormal
            if str(category / "gt" / f"{p.stem}.txt") not in excluded
        ]
    if args.unique_test_base:
        normals = unique_base_samples(normals)
        abnormal = unique_base_samples(abnormal)
    if len(templates) < args.template_count or not normals or not abnormal:
        raise ValueError(f"Missing train/test files in {category}")
    if len(normals) < args.normal_count or len(abnormal) < args.abnormal_count:
        raise ValueError(f"Too few eligible test cases in {category}")
    selected = normals[: args.normal_count] + abnormal[: args.abnormal_count]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_templates(templates[: args.template_count], args.voxel)
    print(json.dumps({"template_preparation": {
        "paths": [str(path) for path in prepared["paths"]],
        "registrations": prepared["registrations"],
        "seconds": prepared["seconds"],
    }}), flush=True)
    for test in selected:
        gt = category / "gt" / (test.stem + ".txt")
        gt_path = gt if gt.exists() else None
        if "good" not in test.stem and gt_path is None:
            raise FileNotFoundError(gt)
        result = run_case(prepared, test, gt_path, args.voxel)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
