"""Summarize saved paper results or outputs from train.py; no model inference."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
METRICS = (
    "test_labeled_f1_doc_macro",
    "test_unlabeled_f1_doc_macro",
    "test_labeled_f1_doc_macro_temporal",
    "test_unlabeled_f1_doc_macro_temporal",
)


def validate_rows(rows):
    if not rows:
        raise ValueError("No results to summarize.")
    seen = set()
    parsed = []
    for raw in rows:
        row = dict(raw)
        if row.get("phase") not in ("baseline", "iterative"):
            raise ValueError("Unknown phase.")
        row["seed"] = int(row["seed"])
        key = row["phase"], row["seed"]
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid F1 for {key}: {metric}={value}")
            row[metric] = value
        parsed.append(row)
    phases = {r["phase"] for r in parsed}
    if len(phases) == 2:
        seeds = {p: {r["seed"] for r in parsed if r["phase"] == p} for p in phases}
        if seeds["baseline"] != seeds["iterative"]:
            raise ValueError("Paired comparisons require identical seed sets.")
    return sorted(parsed, key=lambda r: (r["phase"], r["seed"]))


def load_runs(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tdp-paper-runs/v1":
        raise ValueError("Unsupported run manifest.")
    if manifest.get("status") != "complete":
        raise ValueError("The requested run batch is incomplete.")
    rows = []
    for run in manifest["runs"]:
        if run["status"] != "complete":
            raise ValueError(f"Incomplete run: {run['run_id']}")
        with (manifest_path.parent / run["final_test_csv"]).open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        if len(saved) != 1:
            raise ValueError(f"Expected one final result for {run['run_id']}")
        if saved[0].get("timestamp") != run["run_id"]:
            raise ValueError(f"Result identity differs from manifest: {run['run_id']}")
        rows.append(dict(saved[0], phase=run["phase"], seed=run["seed"]))
    return validate_rows(rows)


def summarize(rows):
    rows = validate_rows(rows)
    summary = []
    for phase in ("baseline", "iterative"):
        selected = [r for r in rows if r["phase"] == phase]
        if not selected:
            continue
        for metric in METRICS:
            values = [r[metric] * 100 for r in selected]
            summary.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "n": len(values),
                    "mean_percent": mean(values),
                    "sample_std_percent": stdev(values) if len(values) > 1 else None,
                }
            )
    baseline = {r["seed"]: r for r in rows if r["phase"] == "baseline"}
    iterative = {r["seed"]: r for r in rows if r["phase"] == "iterative"}
    paired = [
        {
            "seed": seed,
            **{
                metric: 100 * (iterative[seed][metric] - baseline[seed][metric])
                for metric in METRICS
            },
        }
        for seed in sorted(baseline.keys() & iterative.keys())
    ]
    return {
        "summary": summary,
        "paired_deltas_percentage_points": paired,
        "mean_paired_deltas_percentage_points": (
            {metric: mean(r[metric] for r in paired) for metric in METRICS}
            if paired
            else {}
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--runs", type=Path, help="manifest.json from train.py")
    source.add_argument("--csv", type=Path, default=ROOT / "results/paper_per_seed.csv")
    parser.add_argument(
        "--output", type=Path, help="Optional JSON output; otherwise print only"
    )
    args = parser.parse_args(argv)
    try:
        if args.runs:
            rows = load_runs(args.runs)
        else:
            with args.csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        result = summarize(rows)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    for row in result["summary"]:
        spread = (
            "n/a"
            if row["sample_std_percent"] is None
            else f"{row['sample_std_percent']:.2f}"
        )
        print(
            f"{row['phase']:9s} {row['metric']:43s} {row['mean_percent']:.2f} ± {spread} (n={row['n']})"
        )
    for metric, delta in result["mean_paired_deltas_percentage_points"].items():
        print(f"Delta {metric}: {delta:+.2f} pp")
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
