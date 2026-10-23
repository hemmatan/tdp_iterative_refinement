"""Generate paper tables, diagnostics and the DEV figure from saved results."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

import summarize

ROOT = Path(__file__).resolve().parent
SEEDS = (30, 31, 32)
MODES = ("labeled", "unlabeled")
RELATIONS = ("before", "included", "Depend-on", "overlap", "after")
BINS = ("same", "adjacent", "far2plus")
DIAGNOSTICS = {
    "iterations": ("paper_dev_iterations.csv", "per_iter_slice_metrics_fair.csv"),
    "stability": ("paper_dev_stability.csv", "per_iter_slice_metrics_fair_stability.csv"),
    "distance": ("paper_dev_distance.csv", "per_iter_slice_metrics_fair_distance_binned.csv"),
    "relations": ("paper_dev_relations.csv", "per_iter_slice_metrics_fair_per_label.csv"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(results_dir=None, runs=None):
    data = {}
    hashes = {}
    if runs is None:
        root = (results_dir or ROOT / "results").resolve()
        provenance = root / "provenance.json"
        saved = json.loads(provenance.read_text())
        paths = {"main": "paper_per_seed.csv", **{k: v[0] for k, v in DIAGNOSTICS.items()}}
        for kind, name in paths.items():
            path = root / name
            checksum = digest(path)
            if checksum != saved["export_sha256"][name]:
                raise ValueError(f"Archived checksum differs: {name}")
            data[kind] = read_csv(path)
            hashes[name] = checksum
        hashes["provenance.json"] = digest(provenance)
        origin = "Archived paper runs; no new model execution."
    else:
        runs = runs.resolve()
        root = runs.parent
        data["main"] = summarize.load_runs(runs)
        manifest = json.loads(runs.read_text())
        hashes[runs.name] = digest(runs)
        data.update({k: [] for k in DIAGNOSTICS})
        for run in manifest["runs"]:
            final_path = (root / run["final_test_csv"]).resolve()
            final_path.relative_to(root)
            hashes[str(final_path.relative_to(root))] = digest(final_path)
            if run["phase"] != "iterative":
                continue
            saved = read_csv(final_path)[0]
            folders = [p for p in (final_path.parent / "per_iter").glob("*")
                       if p.is_dir() and p.name.endswith("_" + saved["model_name"])]
            if len(folders) != 1:
                raise ValueError(f"Expected one diagnostic directory for {run['run_id']}")
            metadata_path = folders[0] / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if (metadata["model_name"] != saved["model_name"] or
                    folders[0].name != str(metadata["hp_timestamp"]) + "_" + saved["model_name"]):
                raise ValueError(f"Diagnostic identity differs for {run['run_id']}")
            hashes[str(metadata_path.relative_to(root))] = digest(metadata_path)
            for kind, (_, filename) in DIAGNOSTICS.items():
                path = folders[0] / filename
                data[kind].extend(dict(row, phase=run["phase"], seed=run["seed"])
                                  for row in read_csv(path))
                hashes[str(path.relative_to(root))] = digest(path)
        origin = "Completed runs supplied through train.py's manifest."
    return data, {"origin": origin, "input_sha256": hashes}


def index_diagnostic(rows, dimensions, values, metric, name):
    expected = set(itertools.product(SEEDS, *values))
    indexed = {}
    for row in rows:
        if row["phase"] not in ("baseline", "iterative"):
            raise ValueError(f"Unknown phase in {name}")
        if row["phase"] == "baseline":
            continue
        key = (int(row["seed"]), *(int(row[d]) if d == "iteration" else row[d]
                                   for d in dimensions))
        if key in indexed:
            raise ValueError(f"Duplicate {name} row: {key}")
        value = float(row[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {name} value: {key}")
        indexed[key] = value
    if indexed.keys() != expected:
        raise ValueError(f"Incomplete or unexpected {name} coverage: "
                         f"missing {len(expected - indexed.keys())}, "
                         f"extra {len(indexed.keys() - expected)}")
    return indexed


def stats(values, scale=1):
    numbers = [value * scale for value in values]
    return {"n": len(numbers), "mean": mean(numbers), "sample_std": stdev(numbers)}


def build_report(data):
    main = summarize.validate_rows(data["main"])
    if {(r["phase"], r["seed"]) for r in main} != set(itertools.product(("baseline", "iterative"), SEEDS)):
        raise ValueError("A paper report requires both models and seeds 30, 31, 32.")
    iterations = index_diagnostic(data["iterations"], ("iteration", "scope", "mode"),
                                  (range(4), ("all", "temporal"), MODES), "f1_doc_macro", "iterations")
    stability = index_diagnostic(data["stability"], ("iteration", "mode"),
                                 (range(1, 4), MODES), "jaccard_micro", "stability")
    distance = index_diagnostic(data["distance"], ("iteration", "mode", "bin"),
                                (range(4), MODES, BINS), "f1_micro", "distance")
    relations = index_diagnostic(data["relations"], ("iteration", "label"),
                                 (range(4), RELATIONS), "f1_micro", "relations")
    result = {
        "schema": "tdp-paper-report/v1",
        "seeds": list(SEEDS),
        "metrics": {
            "main_and_dev_iterations": "document-level macro F1, percent",
            "distance_and_relations": "micro F1, percent",
            "self_jaccard": "consecutive predicted edge sets, micro, fraction",
            "gains": "paired iteration 3 minus 0, percentage points",
            "spread": "sample standard deviation across seeds, ddof=1",
        },
        "main": summarize.summarize(main),
        "dev_iterations": [], "dev_gains": [], "self_jaccard": [],
        "distance_iterations": [], "distance_gains": [],
        "relation_iterations": [], "relation_gains": [],
    }
    for scope, mode in itertools.product(("all", "temporal"), MODES):
        for i in range(4):
            result["dev_iterations"].append(dict(iteration=i, scope=scope, mode=mode,
                **stats((iterations[s, i, scope, mode] for s in SEEDS), 100)))
        result["dev_gains"].append(dict(scope=scope, mode=mode,
            **stats((iterations[s, 3, scope, mode] - iterations[s, 0, scope, mode] for s in SEEDS), 100)))
    for i, mode in itertools.product(range(1, 4), MODES):
        result["self_jaccard"].append(dict(iteration=i, mode=mode,
            **stats(stability[s, i, mode] for s in SEEDS)))
    for mode, bin_name in itertools.product(MODES, BINS):
        for i in range(4):
            result["distance_iterations"].append(dict(iteration=i, mode=mode, bin=bin_name,
                **stats((distance[s, i, mode, bin_name] for s in SEEDS), 100)))
        result["distance_gains"].append(dict(mode=mode, bin=bin_name,
            **stats((distance[s, 3, mode, bin_name] - distance[s, 0, mode, bin_name] for s in SEEDS), 100)))
    for label in RELATIONS:
        for i in range(4):
            result["relation_iterations"].append(dict(iteration=i, label=label,
                **stats((relations[s, i, label] for s in SEEDS), 100)))
        result["relation_gains"].append(dict(label=label,
            **stats((relations[s, 3, label] - relations[s, 0, label] for s in SEEDS), 100)))
    return result


def main_table(result):
    rows = []
    for phase, label in (("baseline", "Baseline (T=0)"), ("iterative", "Iterative (T=3)")):
        cells = {r["metric"]: f"{r['mean_percent']:.2f} ± {r['sample_std_percent']:.2f}"
                 for r in result["main"]["summary"] if r["phase"] == phase}
        rows.append([label, *(cells[m] for m in summarize.METRICS)])
    rows.append(["Paired gain", *(f"{result['main']['mean_paired_deltas_percentage_points'][m]:+.2f}"
                                  for m in summarize.METRICS)])
    return rows


def comparison_table(result):
    dev = next(r["mean"] for r in result["dev_iterations"]
               if r["iteration"] == 3 and r["scope"] == "all" and r["mode"] == "labeled")
    test = next(r["mean_percent"] for r in result["main"]["summary"]
                if r["phase"] == "iterative" and r["metric"] == summarize.METRICS[0])
    # Literature values transcribed from the paper; only our row is computed.
    return [["NLI-based TDG (avg.)", "66.0", "74.0"], ["DocTime", "69.0", "77.0"],
            ["Ours", f"{dev:.2f}", f"{test:.2f}"]]


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def latex_table(headers, rows):
    def cell(value):
        text = str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        return text.replace("±", r"$\pm$")
    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\hline"]
    lines += [" & ".join(cell(x) for x in row) + r" \\" for row in [headers, *rows]]
    return "\n".join([*lines, r"\hline", r"\end{tabular}", ""])


def render_markdown(result):
    sections = ["# Paper results", result["provenance"]["origin"],
                "F1 values are percentages; gains are percentage points. Spreads are sample standard deviations across three seeds (ddof=1). Gains are paired within each seed before aggregation.",
                "## TEST: document-level macro F1",
                markdown_table(["Model", "All labeled", "All unlabeled", "Temporal labeled", "Temporal unlabeled"], main_table(result)),
                "## DEV: refinement progression (document-level macro F1)",
                markdown_table(["Iteration", "Scope", "Mode", "Mean ± std"],
                    [[r['iteration'], r['scope'], r['mode'], f"{r['mean']:.2f} ± {r['sample_std']:.2f}"] for r in result['dev_iterations']]),
                "## DEV: gain from iteration 0 to 3",
                markdown_table(["Scope", "Mode", "Mean ± std"],
                    [[r['scope'], r['mode'], f"{r['mean']:+.2f} ± {r['sample_std']:.2f}"] for r in result['dev_gains']]),
                "## DEV: consecutive-prediction self-Jaccard (micro)",
                markdown_table(["Transition", "Mode", "Mean ± std"],
                    [[f"{r['iteration']-1} → {r['iteration']}", r['mode'], f"{r['mean']:.4f} ± {r['sample_std']:.4f}"] for r in result['self_jaccard']]),
                "## DEV: distance gains (micro F1, iteration 3 minus 0)",
                "Sentence distance: same = 0, adjacent = 1, far2plus = 2 or more.",
                markdown_table(["Mode", "Distance", "Mean ± std"],
                    [[r['mode'], r['bin'], f"{r['mean']:+.2f} ± {r['sample_std']:.2f}"] for r in result['distance_gains']]),
                "## DEV: relation gains (micro F1, iteration 3 minus 0)",
                markdown_table(["Relation", "Mean ± std"],
                    [[r['label'], f"{r['mean']:+.2f} ± {r['sample_std']:.2f}"] for r in result['relation_gains']]),
                "## Contextual comparison",
                markdown_table(["Model", "DEV", "TEST"], comparison_table(result)),
                "NLI-based TDG: Yao et al. (2023), paper citation key `yao-etal-2023-textual`; DocTime: Mathur et al. (2022), `mathur-etal-2022-doctime`. These two rows are literature values transcribed from the paper, not executions of this repository. The protocols differ; this is contextual reference, not a matched comparison."]
    return "\n\n".join(sections) + "\n"


def make_figure(result):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        raise ValueError("Plotting requires requirements-report.txt; use --no-plot for tables only.") from error
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for mode, color, marker in (("labeled", "#2166ac", "o"), ("unlabeled", "#b35806", "s")):
        rows = [r for r in result["dev_iterations"] if r["scope"] == "all" and r["mode"] == mode]
        ax.errorbar([r["iteration"] for r in rows], [r["mean"] for r in rows],
                    yerr=[r["sample_std"] for r in rows], label=f"{mode.capitalize()} F1",
                    color=color, marker=marker, capsize=4, linewidth=1.8)
        for row in rows:
            ax.annotate(f"{row['mean']:.2f}", (row['iteration'], row['mean']),
                        xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="Refinement iteration", ylabel="Document-level macro F1 (%)",
           title="DEV progression · T=3 · mean ± sample std over 3 seeds", xticks=range(4))
    ax.margins(x=0.08, y=0.18)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="center right", frameon=False)
    return figure


def write_report(result, output_dir, plot=True):
    if output_dir.exists():
        raise ValueError("Report output directory must be new; existing files are never overwritten.")
    figure = make_figure(result) if plot else None
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
    (output_dir / "main_results.tex").write_text(latex_table(
        ["Model", "All L", "All U", "Temporal L", "Temporal U"], main_table(result)), encoding="utf-8")
    (output_dir / "comparison.tex").write_text(latex_table(
        ["Model", "DEV", "TEST"], comparison_table(result)), encoding="utf-8")
    for name in ("dev_iterations", "dev_gains", "self_jaccard", "distance_iterations", "distance_gains", "relation_iterations", "relation_gains"):
        rows = result[name]
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if figure is not None:
        figure.savefig(output_dir / "dev_iterations.pdf")
        figure.savefig(output_dir / "dev_iterations.png", dpi=180)
    # Written last so a failed export does not look like a complete report.
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results-dir", type=Path, help="Archived CSV directory with provenance.json")
    source.add_argument("--runs", type=Path, help="Completed six-run manifest.json from train.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for reports")
    parser.add_argument("--no-plot", action="store_true", help="Generate tables and JSON/CSV without matplotlib")
    args = parser.parse_args(argv)
    try:
        data, provenance = load_inputs(args.results_dir, args.runs)
        result = build_report(data)
        result["provenance"] = provenance
        write_report(result, args.output_dir, plot=not args.no_plot)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Paper report written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
