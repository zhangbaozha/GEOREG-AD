"""Independent archive checks; does not import the experiment's scoring code."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def grouped_metrics(y, s):
    positives = int(y.sum())
    negatives = len(y) - positives
    ranks = rankdata(s, method="average")
    auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-s)
    sorted_scores, sorted_y = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(s) - 1]
    tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
    ap = np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives
    return float(auc), float(ap)


def verify(root, data):
    evidence = []
    for category in sorted(p for p in data.iterdir() if (p / "test").is_dir()):
        case_dir = root / "results" / category.name / "cases"
        cases = [json.loads(p.read_text()) for p in sorted(case_dir.glob("*.json"))]
        expected = {p.stem for p in (category / "test").glob("*.pcd")}
        actual = {x["sample"] for x in cases}
        if expected != actual or len(cases) != len(actual):
            raise ValueError(f"Coverage mismatch: {category.name}")
        normal = next(x for x in cases if not x["is_anomaly"])
        anomaly = next(x for x in cases if x["is_anomaly"] and x["point_gt_valid"])
        with np.load(normal["score_file"]) as archive:
            assert not np.any(archive["labels"])
            assert len(archive["labels"]) == len(archive["scores"])
        with np.load(anomaly["score_file"]) as archive:
            labels, scores = archive["labels"], archive["scores"]
        assert scores.dtype == np.float64
        assert np.isin(labels, [0, 1]).all()
        xyz = np.asarray(o3d.io.read_point_cloud(anomaly["test"]).points)
        gt = np.loadtxt(anomaly["gt"])
        assert np.isin(gt[:, 3], [0, 1]).all()
        distances, ids = cKDTree(gt[:, :3]).query(xyz, k=1, workers=1)
        assert distances.max() < 1e-5
        assert np.array_equal(labels, gt[ids, 3])
        auc, ap = grouped_metrics(labels, scores)
        assert abs(auc - anomaly["raw_point_auc"]) < 1e-12
        assert abs(ap - anomaly["raw_point_ap"]) < 1e-12
        invalid = [x for x in cases if not x["point_gt_valid"]]
        for item in invalid:
            with np.load(item["score_file"]) as archive:
                assert len(archive["labels"]) == 0
                assert len(archive["scores"]) == item["full_points"]
            assert item["is_anomaly"]
        evidence.append({"category": category.name, "coverage_scans": len(cases),
                         "normal_spotcheck": normal["sample"],
                         "anomaly_spotcheck": anomaly["sample"],
                         "score_dtype": str(scores.dtype),
                         "gt_nearest_neighbor_max_error": float(distances.max()),
                         "point_auc_independent": auc, "point_ap_independent": ap,
                         "invalid_point_gt_scans": [x["sample"] for x in invalid]})
    result = {"status": "PASS", "checks": evidence,
              "scope": "All test-filename coverage; one normal and one anomalous archive per category; independent nearest-neighbor GT alignment and rank/grouped-threshold AUROC/AP on the anomalous spotchecks; all invalid-GT archive checks."}
    (root / "raw_archive_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_root, args.data_root)
