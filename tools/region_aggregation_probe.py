#!/usr/bin/env python3
"""Region-aggregation probe: replace the k-NN smoothing bandwidth with
connectivity-based anomaly-region aggregation.

Motivation
----------
``interpolation.k`` behaves as a spatial low-pass bandwidth.  The diagnostic
showed the optimum k is set per dataset and even flips sign across categories,
so it is not a physical quantity.  What the smoothing is really buying is
*spatial coherence*: Real3D-AD defects are contiguous regions, and a blurred
score field ranks the whole region highly while suppressing isolated noise
spikes.

Connectivity can supply that coherence directly, without a bandwidth.  For an
anchor ``p`` define

    region(p) = |C(p, s(p))| = number of anchors that are both connected to p
                and have a score >= s(p)

i.e. the size of the anomalous region p sits in, at p's own level.  A point
inside a contiguous defect accumulates a large region; an isolated spike
accumulates 1.  Ranking by this quantity is the "group the anomalous points
together" operator.

Implementation note: inserting anchors in descending score order and unioning
with already-inserted neighbours makes the component containing p at insertion
time exactly ``{q : s(q) >= s(p)}`` reachable from p.  One pass therefore
yields every point's region size - this is the max-tree / component-tree
sweep, not a per-threshold relabelling.

Only the connectivity radius remains, and it is a *physical* length tied to
the template spacing h.  Because the aggregation itself does the denoising,
interpolation drops back to a small k, so k stops being a tuned knob.

Two stages
----------
    --cache    run the pipeline on one category and save anchors + scores
                + labels (expensive: template prep ~50 s)
    --analyze  load the cache and compare aggregation schemes (cheap, so the
                scheme grid can be iterated freely)

Usage:
    python tools/region_aggregation_probe.py --cache  --category airplane ...
    python tools/region_aggregation_probe.py --analyze --cache-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from interp_bandwidth_diagnostic import interpolate_many, pooled_metrics


# --------------------------------------------------------------------------
# Core: component size at each point's own level, single descending pass
# --------------------------------------------------------------------------

def neighbour_arrays(xyz, radius, max_degree=32, threads=1):
    """Symmetric neighbour lists via a radius graph, degree capped by distance.

    A cap keeps the union-find pass linear when the radius is generous; the
    nearest neighbours are the ones retained, so the cap does not change
    connectivity inside dense regions.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    dist, idx = cKDTree(xyz).query(xyz, k=min(max_degree + 1, len(xyz)),
                                   distance_upper_bound=radius, workers=threads)
    n = len(xyz)
    src, dst = [], []
    for i in range(n):
        row_d, row_i = dist[i], idx[i]
        keep = np.isfinite(row_d) & (row_i < n) & (row_i != i)
        src.append(np.full(int(keep.sum()), i, dtype=np.int64))
        dst.append(row_i[keep].astype(np.int64))
    src = np.concatenate(src) if src else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst) if dst else np.empty(0, dtype=np.int64)
    return src, dst


def region_size(scores, src, dst, n):
    """|C(p, s(p))| for every p, by descending insertion + union-find."""
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    active = np.zeros(n, dtype=bool)
    region = np.ones(n, dtype=np.int64)

    # CSR of the neighbour graph for fast inner iteration
    order_e = np.argsort(src, kind="stable")
    src_s, dst_s = src[order_e], dst[order_e]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    indptr = np.cumsum(indptr)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    start = 0
    while start < n:
        stop = start + 1
        # Insert one full tie level together so equal scores are one level.
        while stop < n and s_sorted[stop] == s_sorted[start]:
            stop += 1
        batch = order[start:stop]
        active[batch] = True
        for p in batch:
            for e in range(indptr[p], indptr[p + 1]):
                q = int(dst_s[e])
                if active[q]:
                    ra, rb = find(int(p)), find(q)
                    if ra != rb:
                        if size[ra] < size[rb]:
                            ra, rb = rb, ra
                        parent[rb] = ra
                        size[ra] += size[rb]
        for p in batch:
            region[p] = size[find(int(p))]
        start = stop
    return region


def rankdata(values):
    """Average ranks, 0-based, ties shared (no scipy dependency)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_v = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_v[stop] == sorted_v[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    return ranks


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    return np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)


# --------------------------------------------------------------------------
# Schemes
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def build_cache(args):
    from georeg3dad.config import load_config
    from georeg3dad.datasets import align_labels, inspect_dataset
    from georeg3dad.geometry import GeoReg3DAD
    from interp_bandwidth_diagnostic import anchor_pass, pick

    settings, method = load_config(args.config, args.preset, args.set)
    manifest = inspect_dataset(settings["dataset"], args.data_root, [args.category],
                               args.scope or "all")
    row = manifest["categories"][0]
    model = GeoReg3DAD(method, threads=args.threads)
    library = model.prepare(row["templates"])
    print(f"h = {library.h:.5f}  template anchors = {len(library.xyz)}", flush=True)

    normals = [c for c in row["cases"] if not c["is_anomaly"]]
    anomalies = [c for c in row["cases"] if c["is_anomaly"] and c["point_gt_valid"]]
    chosen = pick(normals, args.normal_scans, "test") + pick(anomalies, args.anomaly_scans, "test")
    rng = np.random.default_rng(args.subsample_seed)

    anchors, scores, fulls, labels_all, meta = [], [], [], [], []
    for i, case in enumerate(chosen, 1):
        print(f"  [{i}/{len(chosen)}] {case['sample']}", flush=True)
        got = anchor_pass(model, case["test"], case["sample"] + ".pcd")
        labels = (align_labels(got["full_xyz"], case["gt"], settings["dataset"], args.threads)
                  if case["gt"] else np.zeros(len(got["full_xyz"]), dtype=np.int8))
        full = got["full_xyz_registered"]
        if args.max_full_points and len(full) > args.max_full_points:
            idx = np.sort(rng.choice(len(full), size=args.max_full_points, replace=False))
            full, labels = full[idx], labels[idx]
        anchors.append(got["anchor_xyz"])
        scores.append(got["anchor_scores"])
        fulls.append(full)
        labels_all.append(labels)
        meta.append({"sample": case["sample"], "is_anomaly": bool(case["is_anomaly"]),
                     "anchors": int(len(got["anchor_xyz"])), "full_points": int(len(full)),
                     "positives": int(labels.sum())})

    out = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"cache_{args.category}.npz",
        h=library.h,
        n_scan=len(chosen),
        anchors=np.concatenate(anchors),
        anchor_offsets=np.cumsum([0] + [len(a) for a in anchors]),
        scores=np.concatenate(scores),
        fulls=np.concatenate(fulls),
        full_offsets=np.cumsum([0] + [len(f) for f in fulls]),
        labels=np.concatenate(labels_all))
    (out / f"meta_{args.category}.json").write_text(
        json.dumps({"category": args.category, "h": library.h, "scans": meta},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"cached -> {out/f'cache_{args.category}.npz'}", flush=True)


# --------------------------------------------------------------------------
# Analyze
# --------------------------------------------------------------------------

def analyze(args):
    path = Path(args.cache_dir) / f"cache_{args.category}.npz"
    blob = np.load(path)
    h = float(blob["h"])
    anchors_all, scores_all = blob["anchors"], blob["scores"]
    fulls_all, labels_all = blob["fulls"], blob["labels"]
    aoff, foff = blob["anchor_offsets"], blob["full_offsets"]
    n_scan = int(blob["n_scan"])
    meta = json.loads((Path(args.cache_dir) / f"meta_{args.category}.json").read_text(encoding="utf-8"))

    mults = [float(x) for x in args.radius_mults.split(",")]
    k_probe = [int(x) for x in args.k_list.split(",")]

    # scheme name -> per-radius function(anchors, scores) -> values
    def mk(size_pow, use_size, blend):
        def fn(a, s, region):
            if not use_size:
                return s
            if blend == "product":
                return s * (region ** size_pow)
            if blend == "ranksum":
                return rankdata(s) + rankdata(region) * size_pow
            return region.astype(np.float64)
        return fn

    results = []  # (scan_name, scheme, k, metrics)
    for si in range(n_scan):
        a = anchors_all[aoff[si]:aoff[si + 1]]
        s = scores_all[aoff[si]:aoff[si + 1]]
        f = fulls_all[foff[si]:foff[si + 1]]
        y = labels_all[foff[si]:foff[si + 1]]
        name = meta["scans"][si]["sample"]
        print(f"[{si+1}/{n_scan}] {name}  anchors={len(a)} full={len(f)}", flush=True)
        results.append({"scan": name, "is_anomaly": meta["scans"][si]["is_anomaly"],
                        "anchors": len(a), "full": len(f), "labels": y,
                        "scores": s, "anchor_xyz": a, "full_xyz": f})

    # ---- baseline: raw scores at several k -------------------------------
    curves = {}
    for k in k_probe:
        ys, ss = [], []
        for r in results:
            sweep, _, _ = interpolate_many(r["full_xyz"], r["anchor_xyz"], r["scores"], [k], [0.0],
                                           chunk_size=args.chunk_size, threads=args.threads)
            ys.append(r["labels"]); ss.append(sweep[(k, 0.0)])
        curves[f"raw|k={k}"] = pooled_metrics(np.concatenate(ys), np.concatenate(ss))

    # ---- aggregation schemes (interpolated with the small k) -------------
    for mult in mults:
        radii = [mult * h]
        agg = {}
        for r in results:
            a, s = r["anchor_xyz"], r["scores"]
            src, dst = neighbour_arrays(a, mult * h, max_degree=args.max_degree, threads=args.threads)
            region = region_size(s, src, dst, len(a))
            r["region"] = region
            r["graph_edges"] = len(src) // 2
        for label, fn in (
            ("size", lambda a, s, reg: reg.astype(np.float64)),
            ("score*size", lambda a, s, reg: s * reg),
            ("score*sqrt(size)", lambda a, s, reg: s * np.sqrt(reg)),
            ("rank(s)+rank(size)", lambda a, s, reg: rankdata(s) + rankdata(reg)),
            ("score+norm(size)", lambda a, s, reg: normalize(s) + normalize(reg)),
        ):
            ys, ss = [], []
            for r in results:
                values = fn(r["anchor_xyz"], r["scores"], r["region"])
                sweep, _, _ = interpolate_many(r["full_xyz"], r["anchor_xyz"], values,
                                               [args.interp_k], [0.0],
                                               chunk_size=args.chunk_size, threads=args.threads)
                ys.append(r["labels"]); ss.append(sweep[(args.interp_k, 0.0)])
            agg[label] = pooled_metrics(np.concatenate(ys), np.concatenate(ss))
        curves[f"r={mult:g}h"] = agg

    return {"category": args.category, "h": h, "interp_k": args.interp_k,
            "scans": len(results), "curves": curves,
            "graph_edges": results[0]["graph_edges"] if results else None,
            "positives": int(sum(int(r["labels"].sum()) for r in results)),
            "points": int(sum(len(r["labels"]) for r in results))}


def print_report(res):
    print(f"\n=== {res['category']}  (h={res['h']:.5f}, {res['scans']} scans, "
          f"{res['points']:,} pts, {res['positives']:,} pos) ===")
    print("\n-- baseline: raw scores, interpolated with different k --")
    print(f"  {'scheme':<26}{'P-AUROC':>10}{'P-AP':>10}")
    for key, value in res["curves"].items():
        if isinstance(value, dict) and "p_auroc" in value:
            print(f"  {key:<26}{value['p_auroc']:>10.4f}{value['p_ap']:>10.4f}")
    print(f"\n-- region aggregation, interpolated with k={res['interp_k']} --")
    for key, value in res["curves"].items():
        if not isinstance(value, dict) or "p_auroc" in value:
            continue
        print(f"  [{key}]")
        for scheme, m in value.items():
            print(f"    {scheme:<24}{m['p_auroc']:>10.4f}{m['p_ap']:>10.4f}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--preset")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--category", required=True)
    p.add_argument("--scope", choices=("all", "pcd", "new_pcd"))
    p.add_argument("--normal-scans", type=int, default=5)
    p.add_argument("--anomaly-scans", type=int, default=5)
    p.add_argument("--max-full-points", type=int, default=40000)
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--k-list", default="3,16,64,128,256")
    p.add_argument("--radius-mults", default="0.5,1,2,3,5")
    p.add_argument("--max-degree", type=int, default=32)
    p.add_argument("--interp-k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cache:
        if not args.data_root or not args.config:
            raise SystemExit("--cache needs --data-root and --config")
        build_cache(args)
        return 0
    if args.analyze:
        res = analyze(args)
        print_report(res)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"region_probe_{args.category}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0
    raise SystemExit("Pass --cache or --analyze")


if __name__ == "__main__":
    raise SystemExit(main())
