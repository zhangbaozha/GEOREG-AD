"""Summarize paired category effects from a completed GeoReg component run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("pooled_point_auc", "pooled_point_ap", "object_auc", "object_ap")


def bootstrap_interval(values: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    indices = generator.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    default = str(report["default_variant"])
    rows = report["categories"]
    generator = np.random.default_rng(20260918)
    effects: list[dict[str, object]] = []
    for variant in report["variants"]:
        item: dict[str, object] = {"variant": variant, "reference": default}
        for metric in METRICS:
            delta = np.asarray([
                float(row["variants"][variant][metric]) - float(row["variants"][default][metric])
                for row in rows
            ])
            low, high = bootstrap_interval(delta, generator)
            item.update({
                f"{metric}_mean_delta": float(delta.mean()),
                f"{metric}_median_delta": float(np.median(delta)),
                f"{metric}_improved_categories": int(np.sum(delta > 0.0)),
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        effects.append(item)
    output = {
        "input": str(args.input.resolve()),
        "n_categories": len(rows),
        "protocol": (
            "Paired unweighted-category deltas against the fixed default. "
            f"The 95% percentile intervals resample the {len(rows)} category deltas with a fixed RNG seed; "
            "they describe category variation, not independent physical-item uncertainty."
        ),
        "default_variant": default,
        "metrics": report["aggregate"]["variants"],
        "effects": effects,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "component_effects.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (args.output_dir / "component_effects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
