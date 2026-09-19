"""Check the local summary and case archive without remote point-score files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_BAD = {
    ("chicken", "705_bulge"),
    ("chicken", "706_sink"),
    ("seahorse", "265_bulge"),
    ("seahorse", "265_bulge_cut"),
    ("starfish", "433_bulge"),
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def roc_auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    check(n_pos > 0 and n_neg > 0, "AUROC requires both classes")
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + j - i - 1) / 2
        pos_rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(labels, scores):
    pairs = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    check(positives > 0, "AP requires a positive class")
    tp = 0
    ap = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        new_pos = sum(label for _, label in pairs[i:j])
        tp += new_pos
        ap += new_pos * (tp / j) / positives
        i = j
    return ap


def main(root):
    source = root / "02_results" / "original"
    report = json.loads((source / "raw_full_summary.json").read_text(encoding="utf-8"))
    cases = json.loads((source / "raw_full_cases.json").read_text(encoding="utf-8"))
    run = json.loads((root / "03_audit/original/raw_full_run_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "03_audit/original/gt_coordinate_audit.json").read_text(encoding="utf-8"))
    archive_check = json.loads((root / "03_audit/original/raw_archive_verification.json").read_text(encoding="utf-8"))
    rows = report["categories"]
    check(len(rows) == 12 and len({r["category"] for r in rows}) == 12, "12 categories")
    check(len(cases) == 1206 and len({(c["category"], c["sample"]) for c in cases}) == 1206, "1206 unique cases")
    check(sum(not c["is_anomaly"] for c in cases) == 604, "604 normal")
    check(sum(c["is_anomaly"] for c in cases) == 602, "602 anomaly")
    bad = {(c["category"], c["sample"]) for c in cases if not c["point_gt_valid"]}
    check(bad == EXPECTED_BAD, "exactly five invalid point GT cases")
    check(sum(c["point_gt_valid"] for c in cases) == 1201, "1201 point-valid")
    check(all(c["gt"] is None and c["positive_points"] is None for c in cases if not c["point_gt_valid"]), "invalid labels are absent")
    check(sum(r["test_scans"] for r in rows) == 1206, "category test count")
    check(sum(r["normal_scans"] for r in rows) == 604, "category normal count")
    check(sum(r["anomaly_scans"] for r in rows) == 602, "category anomaly count")
    check(sum(r["point_valid_scans"] for r in rows) == 1201, "category point-valid count")
    check(sum(r["total_evaluated_points"] for r in rows) == report["total_points"] == 189006793, "pooled point count")
    check(run["expected_object_test_scans"] == 1206 and run["expected_point_valid_test_scans"] == 1201, "run manifest count")
    check(audit["total"] == audit["processed"] == 602 and audit["passed"] == 597 and audit["failed"] == 5, "GT audit count")
    check({(Path(f["gt"]).parts[-3], Path(f["gt"]).stem) for f in audit["failures"]} == EXPECTED_BAD, "GT audit exclusions")
    check(archive_check["status"] == "PASS" and len(archive_check["checks"]) == 12, "archive spotcheck record")

    csv_rows = list(csv.DictReader((source / "raw_full_summary.csv").open(encoding="utf-8-sig", newline="")))
    check(len(csv_rows) == 12, "original CSV row count")
    by_cat = {r["category"]: r for r in rows}
    for csv_row in csv_rows:
        r = by_cat[csv_row["category"]]
        for key in csv_row:
            if key == "category":
                continue
            check(abs(float(csv_row[key]) - float(r[key])) < 1e-12, f"CSV/JSON {r['category']} {key}")
    for r in rows:
        subset = [c for c in cases if c["category"] == r["category"]]
        check(len(subset) == r["test_scans"], "per-category case count")
        check(sum(c["is_anomaly"] for c in subset) == r["anomaly_scans"], "per-category anomaly count")
        check(sum(c["point_gt_valid"] for c in subset) == r["point_valid_scans"], "per-category point-valid count")
        labels = [int(c["is_anomaly"]) for c in subset]
        scores = [float(c["raw_top1pct"]) for c in subset]
        check(abs(roc_auc(labels, scores) - r["object_auc_raw"]) < 1e-12, "independent object AUROC")
        check(abs(average_precision(labels, scores) - r["object_ap_raw"]) < 1e-12, "independent object AP")
    for key, target in {"pooled_point_auc_raw": 0.7967, "pooled_point_ap_raw": 0.2752,
                        "object_auc_raw": 0.8217, "object_ap_raw": 0.7865}.items():
        mean = sum(r[key] for r in rows) / 12
        check(abs(mean - report["macro_average"][key]) < 1e-12, f"macro {key}")
        check(round(mean, 4) == target, f"rounded macro {key}")
    for filename, digest in run["code_sha256"].items():
        check(sha256(root / "04_code/original" / filename) == digest, f"original code hash {filename}")
    print("PASS: 12 categories; 1206=604+602; 1201 point-valid; 5 documented exclusions")
    print("PASS: original CSV/JSON consistency; four macro values; independent object metrics; code SHA256")
    print("LIMIT: point AUROC/AP are checked against stored category values, not recomputed from absent remote NPZ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().archive_root.resolve())
