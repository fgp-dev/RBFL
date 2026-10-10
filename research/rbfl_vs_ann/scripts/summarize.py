#!/usr/bin/env python3
"""Build per-seed, ablation, and aggregate tables from raw benchmark CSVs."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "train_exact", "train_bit", "val_exact", "val_bit", "test_exact",
    "test_bit", "shift_exact", "shift_bit", "gap_bit", "training_ms",
    "ns_per_sequence", "ns_per_step", "model_bytes",
    "temporary_inference_bytes", "total_inference_bytes", "parameter_bits",
    "accepted", "candidates", "inactive_rewires", "q_changes",
)


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_dicts(path, rows, fieldnames):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing raw CSVs")
    root = parser.parse_args().results

    rows = read_csv(root / "benchmark.csv")
    if not rows:
        parser.error("benchmark.csv has no data rows")
    for row in rows:
        row["total_inference_bytes"] = str(
            int(row["model_bytes"]) + int(row["temporary_inference_bytes"])
        )

    fieldnames = list(rows[0])
    write_dicts(root / "seeds.csv", rows, fieldnames)
    ablation = [
        row for row in rows
        if row["task"] == "parity" and row["model"].startswith("rbfl")
    ]
    write_dicts(root / "ablation.csv", ablation, fieldnames)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["model"])].append(row)

    aggregate = []
    for (task, model), group in sorted(groups.items()):
        result = {"task": task, "model": model, "n": len(group)}
        for field in FIELDS:
            # An absent partition has NA, not a measured zero. Keep it absent
            # in summary.csv as well, instead of averaging placeholder values.
            values = [
                float(row[field]) for row in group
                if row[field] not in ("", "NA")
            ]
            mean, sd = mean_sd(values) if values else (None, None)
            result[f"{field}_mean"] = f"{mean:.6f}" if mean is not None else "NA"
            result[f"{field}_sd"] = f"{sd:.6f}" if sd is not None else "NA"
        aggregate.append(result)

    names = ["task", "model", "n"] + [
        f"{field}_{suffix}" for field in FIELDS for suffix in ("mean", "sd")
    ]
    write_dicts(root / "summary.csv", aggregate, names)

    historical = read_csv(root / "historical_reproduction.csv")
    with (root / "historical_summary.csv").open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("model", "n", "train_mean", "train_sd", "test_mean", "test_sd"))
        for model in sorted({row["model"] for row in historical}):
            group = [row for row in historical if row["model"] == model]
            train_mean, train_sd = mean_sd([float(row["train_accuracy"]) for row in group])
            test_mean, test_sd = mean_sd([float(row["test_accuracy"]) for row in group])
            writer.writerow((model, len(group), train_mean, train_sd, test_mean, test_sd))


if __name__ == "__main__":
    main()
