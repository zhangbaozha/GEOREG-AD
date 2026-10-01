#!/usr/bin/env python3
"""Interpolation-bandwidth diagnostic for GeoReg3DAD.

The anchor -> full-resolution interpolation exposes two free parameters,
``interpolation.k`` and ``interpolation.power``.  On Real3D-AD the validated
configuration uses k=128, power=0 while the untuned baseline uses k=3,
power=1, and ShapeNet prefers k=16, power=0.5.  That spread suggests the pair
acts as a *spatial smoothing bandwidth* rather than a model parameter, and
that its optimum is matched to the annotation granularity of each benchmark.

This tool tests that reading in two independent ways.

Q1 -- geometry only, no scoring, no labels on the k side
    How wide is the neighbourhood spanned by the k nearest anchors, measured
    in units of the reference spacing ``h``?  Compare it with the physical
    size of the ground-truth defect blobs.  If the validated k lands on the
    blob scale while the baseline k does not, the curve is being fitted to
    the annotation, not to the surface.

Q2 -- requires labels and scoring
    Sweep (k, power) over the *same* anchor scores and report pooled
    P-AUROC / P-AP.  A broad, flat optimum means smoothing carries real
    signal.  A narrow spike means the value is specific to this benchmark.

Both parts call ``georeg3dad.scoring`` and ``georeg3dad.metrics`` so the
numbers are produced by the same code as the main pipeline.

Usage (real data, run from the directory that holds ``georeg3dad/``):

    python tools/interp_bandwidth_diagnostic.py ^
        --data-root "D:/GeoReg3DAD-Data/datasets/Real3D-AD-PCD/Real3D-AD-PCD" ^
        --config configs/config_real3dad.json ^
        --category airplane ^
        --normal-scans 10 --anomaly-scans 10 ^
        --k-list 3,8,16,32,64,128,256 --power-list 0,0.5,1 ^
        --out "D:/GeoReg3DAD-Data/reports/interp_diag_airplane"

Self test (no dataset, no Open3D, no labels on disk):

    python tools/interp_bandwidth_diagnostic.py --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Pure geometry helpers (numpy / scipy only, so they can be tested anywhere)
# --------------------------------------------------------------------------

def nn_spacing(xyz, threads=1):
    """Median nearest-neighbour distance, the same definition as template ``h``."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        raise ValueError("Need at least two points to measure spacing")
    return float(np.median(cKDTree(xyz).query(xyz, k=2, workers=threads)[0][:, 1]))


def knn_radius_profile(points, ks, threads=1, sample=3000, seed=0):
    """Median radius that contains the k nearest neighbours, for each k.

    On a surface with anchor density rho the count inside radius r grows like
    rho * pi * r^2, so this is the quantity that turns "k neighbours" into a
    physical smoothing width.
    """
    points = np.asarray(points, dtype=np.float64)
    ks = sorted({int(k) for k in ks})
    kmax = ks[-1]
    if len(points) <= kmax:
        raise ValueError(f"Only {len(points)} anchors; cannot measure k={kmax}")
    if sample and len(points) > sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(points), size=sample, replace=False))
    else:
        idx = np.arange(len(points))
    dist = cKDTree(points).query(points[idx], k=list(range(1, kmax + 1)), workers=threads)[0]
    return {k: float(np.median(dist[:, k - 1])) for k in ks}


def blob_stats(blob):
    """Size of a ground-truth defect blob around its own centroid."""
    blob = np.asarray(blob, dtype=np.float64)
    if len(blob) == 0:
        return None
    centre = blob.mean(axis=0)
    radius = np.linalg.norm(blob - centre, axis=1)
    return {
        "n_points": int(len(blob)),
        "r_gyration": float(np.sqrt(np.mean(radius ** 2))),
        "r_p90": float(np.quantile(radius, 0.90)),
        "r_max": float(radius.max()),
    }


# --------------------------------------------------------------------------
# Interpolation sweep
# --------------------------------------------------------------------------

def interpolate_many(full_xyz, anchor_xyz, values, ks, powers,
                     chunk_size=4096, epsilon=1e-8, threads=1):
    """Interpolate one set of anchor scores at several (k, power).

    A single KD-tree query per chunk already returns the kmax nearest anchors
    in order, so every smaller k is a prefix of that same array.  The whole
    grid therefore costs about one interpolation instead of len(ks)*len(powers).

    Semantics are identical to ``georeg3dad.scoring.interpolate``; the self
    test asserts bitwise agreement with it.
    """
    full_xyz = np.asarray(full_xyz, dtype=np.float64)
    anchor_xyz = np.asarray(anchor_xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(anchor_xyz) == 0 or len(values) != len(anchor_xyz):
        raise ValueError("Interpolation requires one score per non-empty anchor")
    if len(full_xyz) == 0:
        raise ValueError("No full-resolution points to score")
    usable = sorted({int(k) for k in ks if 1 <= int(k) <= len(anchor_xyz)})
    if not usable:
        raise ValueError("No requested k fits inside the anchor count")
    powers = sorted({float(p) for p in powers})
    kmax = usable[-1]
    result = {(k, p): np.empty(len(full_xyz), dtype=np.float64) for k in usable for p in powers}
    tree = cKDTree(anchor_xyz)
    for start in range(0, len(full_xyz), chunk_size):
        stop = min(start + chunk_size, len(full_xyz))
        dist, ids = tree.query(full_xyz[start:stop], k=list(range(1, kmax + 1)), workers=threads)
        for k in usable:
            neighbour_scores = values[ids[:, :k]]
            for power in powers:
                if power == 0:
                    # Arithmetic mean, matching the archived k=128 tuning runs.
                    result[(k, power)][start:stop] = np.cumsum(neighbour_scores, axis=1)[:, -1] / k
                else:
                    weights = np.maximum(dist[:, :k], epsilon) ** (-power)
                    weights /= weights.sum(axis=1, keepdims=True)
                    result[(k, power)][start:stop] = (neighbour_scores * weights).sum(axis=1)
    return result, usable, powers


def pooled_metrics(labels, scores):
    from georeg3dad.metrics import point_metrics
    return point_metrics(np.asarray(labels), np.asarray(scores))


# --------------------------------------------------------------------------
# Real-data driver
# --------------------------------------------------------------------------

def anchor_pass(model, path, canonical, transform=None):
    """Mirror of ``GeoReg3DAD.predict`` up to the anchor scores.

    Kept here rather than in the package so the diagnostic can inspect the
    intermediate anchor field that ``predict`` consumes internally.
    """
    import open3d as o3d
    from georeg3dad.geometry import make_features, register, registration_seed, transform_xyz
    from georeg3dad.scoring import anchor_scores, residuals

    cfg, lib = model.config, model.library
    seed = registration_seed(canonical)
    o3d.utility.random.seed(seed)
    full, down, feature = make_features(path, cfg.features)
    if transform is None:
        transform, info = register(down, feature, lib.reference, lib.feature,
                                   cfg.features.voxel, cfg.registration)
    else:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Replay transform must be a finite 4x4 matrix")
        info = {"replayed": True}

    anchor_xyz = transform_xyz(np.asarray(down.points), transform)
    anchor_normals = np.asarray(down.normals) @ transform[:3, :3].T
    components = residuals(anchor_xyz, anchor_normals, lib.xyz, lib.normals,
                           lib.h, cfg.matching, model.threads)
    scores, unmatched = anchor_scores(components, cfg.matching)
    full_points = np.asarray(full.points)
    return {
        # GT association works in the original frame, interpolation in the
        # registered frame; a rigid transform preserves both, but keep the
        # two apart so label alignment matches the runner exactly.
        "full_xyz": full_points,
        "full_xyz_registered": transform_xyz(full_points, transform),
        "anchor_xyz": anchor_xyz,
        "anchor_scores": scores,
        "unmatched_fraction": float(unmatched.mean()),
        "registration": info,
        "seed": seed,
    }


def pick(items, count, key):
    if count <= 0 or count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def run_real(args):
    from georeg3dad.config import load_config, method_from_dict
    from georeg3dad.datasets import align_labels, canonical_filename, inspect_dataset
    from georeg3dad.geometry import GeoReg3DAD

    settings, method = load_config(args.config, args.preset, args.set)
    if settings["dataset"] != "real3dad" and args.scope is None:
        args.scope = "pcd"
    manifest = inspect_dataset(settings["dataset"], args.data_root,
                               [args.category], args.scope or "all")
    row = manifest["categories"][0]
    model = GeoReg3DAD(method, threads=args.threads)
    library = model.prepare(row["templates"])
    h = library.h

    normals = [c for c in row["cases"] if not c["is_anomaly"]]
    anomalies = [c for c in row["cases"] if c["is_anomaly"] and c["point_gt_valid"]]
    chosen = pick(normals, args.normal_scans, "test") + pick(anomalies, args.anomaly_scans, "test")
    if not any(c["is_anomaly"] for c in chosen):
        raise SystemExit("No anomalous scan selected; nothing to diagnose")

    profiles, pooled_labels, pooled_scores = [], [], []
    pooled = {(k, p): [] for k in [int(x) for x in args.k_list.split(",")]
              for p in [float(x) for x in args.power_list.split(",")]}
    per_scan = []
    rng = np.random.default_rng(args.subsample_seed)

    for index, case in enumerate(chosen, 1):
        print(f"[{index}/{len(chosen)}] {case['sample']}"
              f"{' (anomaly)' if case['is_anomaly'] else ''} ...", flush=True)
        got = anchor_pass(model, case["test"], case["sample"] + ".pcd")
        full_orig, full_reg = got["full_xyz"], got["full_xyz_registered"]
        anchor_xyz = got["anchor_xyz"]
        if case["gt"]:
            labels = align_labels(full_orig, case["gt"], settings["dataset"], args.threads)
        else:
            labels = np.zeros(len(full_orig), dtype=np.int8)
        full_xyz = full_reg
        if args.max_full_points and len(full_xyz) > args.max_full_points:
            idx = np.sort(rng.choice(len(full_xyz), size=args.max_full_points, replace=False))
            full_xyz, labels = full_xyz[idx], labels[idx]

        per_scan.append({"sample": case["sample"], "is_anomaly": bool(case["is_anomaly"]),
                         "anchors": int(len(anchor_xyz)), "full_points": int(len(full_xyz)),
                         "blob": blob_stats(full_xyz[labels == 1]) if case["is_anomaly"] else None})

        if case["is_anomaly"] and labels.any():
            profiles.append((case["sample"],
                             knn_radius_profile(anchor_xyz, [int(x) for x in args.k_list.split(",")],
                                                threads=args.threads)))

        scores, usable, powers = interpolate_many(
            full_xyz, anchor_xyz, got["anchor_scores"],
            [int(x) for x in args.k_list.split(",")],
            [float(x) for x in args.power_list.split(",")],
            chunk_size=args.chunk_size, threads=args.threads)
        for key, value in scores.items():
            pooled[key].append(value)
        pooled_labels.append(labels)

    labels_all = np.concatenate(pooled_labels)
    metrics = {}
    for key, chunks in pooled.items():
        if key not in scores:
            continue
        scores_all = np.concatenate(chunks)
        metrics[f"{key[0]}|{key[1]}"] = pooled_metrics(labels_all, scores_all)

    blob_radii = [s["blob"]["r_gyration"] for s in per_scan if s["blob"]]
    blob_radius = float(np.median(blob_radii)) if blob_radii else float("nan")
    ks = sorted({int(x) for x in args.k_list.split(",")})
    radius_by_k = {}
    for _, profile in profiles:
        for k, value in profile.items():
            radius_by_k.setdefault(k, []).append(value)
    radius_by_k = {k: float(np.median(v)) for k, v in radius_by_k.items()}

    return {
        "category": args.category,
        "dataset": settings["dataset"],
        "h": h,
        "scans": per_scan,
        "point_level": {
            "k": int(len(labels_all)),
            "positives": int(labels_all.sum()),
            "prevalence": float(labels_all.mean()),
            "subsampled": bool(args.max_full_points),
        },
        "blob_radius_median": blob_radius,
        "blob_radius_over_h": blob_radius / h if np.isfinite(blob_radius) else None,
        "knn_radius_by_k": radius_by_k,
        "knn_radius_over_h": {k: v / h for k, v in radius_by_k.items()},
        "knn_radius_over_blob": {k: v / blob_radius for k, v in radius_by_k.items()}
        if np.isfinite(blob_radius) else {},
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# Self test: synthetic surface with one known defect blob
# --------------------------------------------------------------------------

def _shared_roughness(u, v):
    """Deterministic surface texture shared by every surface of the scene.

    Because template and test share it, it cancels in the residual and keeps
    the normal region genuinely normal; only the injected noise survives.
    """
    return 0.004 * (np.sin(9.0 * np.pi * u) * np.cos(11.0 * np.pi * v)
                    + 0.5 * np.sin(23.0 * np.pi * u * v))


def _surface(n, centre, sigma, height, noise, seed):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(axis, axis)
    d2 = (u - centre[0]) ** 2 + (v - centre[1]) ** 2
    z = height * np.exp(-d2 / (2.0 * sigma ** 2)) + _shared_roughness(u, v)
    du = np.gradient(z, axis, axis=1)
    dv = np.gradient(z, axis, axis=0)
    normal = np.stack([-du, -dv, np.ones_like(z)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    xyz = np.stack([u, v, z], axis=-1).reshape(-1, 3)
    nrm = normal.reshape(-1, 3)
    if noise:
        xyz = xyz + rng.normal(0.0, noise, xyz.shape)
    pure_bump = height * np.exp(-d2 / (2.0 * sigma ** 2))
    return xyz, nrm, z.reshape(-1), pure_bump.reshape(-1)


def self_test(verbose=True):
    from georeg3dad.config import Interpolation, Matching
    from georeg3dad.scoring import anchor_scores, interpolate, residuals

    ok = []

    # --- 1. nn_spacing on a regular lattice equals the lattice step ---------
    step = 0.05
    grid = np.stack(np.meshgrid(np.arange(6) * step, np.arange(6) * step), axis=-1).reshape(-1, 2)
    grid = np.c_[grid, np.zeros(len(grid))]
    measured = nn_spacing(grid)
    ok.append(("nn_spacing matches lattice step", abs(measured - step) < 1e-12, measured, step))

    # --- 2. interpolate_many reproduces scoring.interpolate exactly ---------
    rng = np.random.default_rng(7)
    anchors = rng.normal(size=(300, 3)) * 0.1
    full = rng.normal(size=(2500, 3)) * 0.3
    values = rng.normal(size=len(anchors))
    fast, usable, powers = interpolate_many(full, anchors, values, [1, 3, 16, 64, 300], [0, 0.5, 1],
                                            chunk_size=333)
    worst = 0.0
    for k in usable:
        for p in powers:
            reference = interpolate(full, anchors, values,
                                    Interpolation(k=k, power=p, chunk_size=333, epsilon=1e-8))
            worst = max(worst, float(np.abs(reference - fast[(k, p)]).max()))
    ok.append(("interpolate_many == scoring.interpolate", worst < 1e-12, worst, 0.0))

    # --- 3. k=1 is nearest-neighbour and power never changes the k=1 case ---
    n1, _, _ = interpolate_many(full, anchors, values, [1], [0, 1, 3])
    agree = float(np.abs(n1[(1, 0.0)] - n1[(1, 3.0)]).max())
    ok.append(("k=1 independent of power", agree < 1e-12, agree, 0.0))

    # --- 4. bandwidth vs blob scale, and the resulting k curve --------------
    n, sigma, height, noise = 100, 0.04, 0.05, 0.006
    clean_xyz, clean_nrm, _, _ = _surface(n, (0.5, 0.5), sigma, 0.0, noise, 0)
    test_xyz, test_nrm, _, pure_bump = _surface(n, (0.5, 0.5), sigma, height, noise, 1)
    anchors = test_xyz.reshape(n, n, 3)[::2, ::2].reshape(-1, 3)
    anchor_nrm = test_nrm.reshape(n, n, 3)[::2, ::2].reshape(-1, 3)
    h = nn_spacing(anchors)

    matching = Matching(candidate_k=8, radius_h=8.0, distance_weight=1.0,
                        plane_weight=0.5, normal_weight=0.5, unmatched_penalty=16.0)
    components = residuals(anchors, anchor_nrm, clean_xyz, clean_nrm, h, matching)
    anchor_values, _ = anchor_scores(components, matching)

    labels = (pure_bump > 0.25 * height).astype(np.int8)
    ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    sweep, usable, _ = interpolate_many(test_xyz, anchors, anchor_values, ks, [0.0])
    curve = {k: pooled_metrics(labels, sweep[(k, 0.0)]) for k in usable}

    radius = knn_radius_profile(anchors, ks)
    centroid = test_xyz[labels == 1].mean(axis=0)
    blob_r = float(np.sqrt(np.mean(np.linalg.norm(test_xyz[labels == 1] - centroid, axis=1) ** 2)))

    best_k = max(curve, key=lambda k: curve[k]["p_auroc"])
    smallest, largest = usable[0], usable[-1]
    ok.append(("synthetic sweep produces a finite curve",
               bool(np.isfinite([curve[k]["p_auroc"] for k in usable]).all()), best_k, None))
    ok.append(("smoothing helps below the optimum",
               curve[smallest]["p_auroc"] < curve[best_k]["p_auroc"],
               curve[smallest]["p_auroc"], curve[best_k]["p_auroc"]))
    ok.append(("oversmoothing hurts above the optimum",
               curve[largest]["p_auroc"] < curve[best_k]["p_auroc"],
               curve[largest]["p_auroc"], curve[best_k]["p_auroc"]))
    ok.append(("optimum lies strictly inside the swept range",
               best_k not in (smallest, largest), best_k, None))

    if verbose:
        print("\n[SELF TEST] synthetic surface, one Gaussian bump")
        print(f"  spacing h = {h:.5f}   blob r_gyration = {blob_r:.5f}   blob/h = {blob_r / h:.2f}")
        print(f"  {'k':>5} {'r_k/h':>8} {'r_k/blob':>9} {'P-AUROC':>9} {'P-AP':>8}")
        for k in usable:
            print(f"  {k:>5} {radius[k] / h:>8.2f} {radius[k] / blob_r:>9.2f} "
                  f"{curve[k]['p_auroc']:>9.4f} {curve[k]['p_ap']:>8.4f}")
        print(f"  best k = {best_k} (r_k/blob = {radius[best_k] / blob_r:.2f})")
        print("\n  checks:")
        for name, passed, got, want in ok:
            tail = "" if want is None else f"  (got {got:.6g}, want {want:.6g})"
            print(f"    [{'PASS' if passed else 'FAIL'}] {name}{tail}")
    return all(passed for _, passed, _, _ in ok), {"curve": curve, "radius": radius, "h": h, "blob_r": blob_r}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the raw numbers first: a formatting problem must never discard
    # the expensive part of the run.
    (out_dir / "interp_bandwidth.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    k_radius = result["knn_radius_over_h"]
    k_blob = result["knn_radius_over_blob"]
    lines = [f"# 插值带宽诊断 — {result['category']}", "",
             f"- 数据集：{result['dataset']}",
             f"- 模板间距 h（最近邻中位值）：{result['h']:.6f}",
             f"- GT 缺陷块半径（回转半径中位值）：{result['blob_radius_median']:.6f}"
             f"（= {result['blob_radius_over_h']:.2f} h）" if result["blob_radius_over_h"] else "",
             f"- 点级样本：{result['point_level']['k']:,} 个点，"
             f"{result['point_level']['positives']:,} 个正例"
             f"（正例率 {result['point_level']['prevalence']:.4%}）", "",
             "## Q1 每个 k 覆盖的物理半径", "",
             "| k | r_k / h | r_k / 缺陷块半径 |", "|---:|---:|---:|"]
    for k in sorted(k_radius, key=lambda x: int(x)):
        blob = f"{k_blob[k]:.2f}" if k in k_blob else "—"
        lines.append(f"| {k} | {k_radius[k]:.2f} | {blob} |")

    lines += ["", "## Q2 k 扫描曲线（点级 pooled）", "",
              "| k | power | P-AUROC | P-AP |", "|---:|---:|---:|---:|"]
    for key, value in sorted(result["metrics"].items(), key=lambda kv: (int(kv[0].split('|')[0]),
                                                                       float(kv[0].split('|')[1]))):
        k, p = key.split("|")
        lines.append(f"| {k} | {p} | {value['p_auroc']:.4f} | {value['p_ap']:.4f} |")

    best = max(result["metrics"].items(), key=lambda kv: kv[1]["p_auroc"]) if result["metrics"] else None
    if best:
        lines += ["", f"**P-AUROC 最优**：k={best[0].split('|')[0]}, power={best[0].split('|')[1]}"
                      f"（{best[1]['p_auroc']:.4f}）"]
    lines += ["", "> 说明：本诊断只在所选类别、所选扫描子集上 pooled 计算，数值与论文全量口径不可直接比较，",
              "> 关注的是曲线形状（峰宽）而非绝对值。"]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--category")
    parser.add_argument("--scope", choices=("all", "pcd", "new_pcd"))
    parser.add_argument("--normal-scans", type=int, default=10)
    parser.add_argument("--anomaly-scans", type=int, default=10)
    parser.add_argument("--k-list", default="3,8,16,32,64,128,256")
    parser.add_argument("--power-list", default="0,0.5,1")
    parser.add_argument("--max-full-points", type=int, default=0,
                        help="uniformly subsample full-resolution points per scan (0 = all)")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        passed, _ = self_test()
        print("\nSELF TEST:", "PASS" if passed else "FAIL")
        return 0 if passed else 1
    missing = [n for n in ("data_root", "config", "category", "out") if getattr(args, n) in (None, "")]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    result = run_real(args)
    out_dir = report(result, args.out)
    print(f"\nQ1  r_k/h: " + ", ".join(f"{k}->{v:.2f}" for k, v in sorted(result['knn_radius_over_h'].items(), key=lambda x: int(x[0]))))
    if result["blob_radius_over_h"]:
        print(f"    blob/h = {result['blob_radius_over_h']:.2f}")
    print("Q2  P-AUROC by (k,power):")
    for key, value in sorted(result["metrics"].items(),
                             key=lambda kv: (int(kv[0].split('|')[0]), float(kv[0].split('|')[1]))):
        print(f"    k={key.split('|')[0]:>4} power={key.split('|')[1]:>4}  "
              f"P-AUROC={value['p_auroc']:.4f}  P-AP={value['p_ap']:.4f}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
