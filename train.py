import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DRIVER = "tdp.experiment"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT / name for name in ("train.py", "summarize.py", "report.py")]
    for directory in ("tdp", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def make_plan(mode, seeds, data_dir, output_dir, gpu):
    if mode not in ("baseline", "iterative", "both"):
        raise ValueError("Unknown model mode.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a nonempty list of distinct seeds.")
    if any(seed not in (30, 31, 32) for seed in seeds):
        raise ValueError("This paper release supports seeds 30, 31 and 32.")
    if gpu < 0:
        raise ValueError("GPU index must be nonnegative.")
    for split in ("train", "dev", "test"):
        if not (data_dir / f"{split}.txt").is_file():
            raise ValueError(f"Missing data file: {data_dir / (split + '.txt')}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(
            "Output directory must be absent or empty; existing results are never overwritten."
        )
    phases = ("baseline", "iterative") if mode == "both" else (mode,)
    plan = []
    for phase in phases:
        for seed in seeds:
            run_id = f"taln_{phase}_s{seed}"
            run_dir = output_dir / run_id
            command = [
                sys.executable,
                "-m",
                DRIVER,
                "--config",
                str(run_dir / "runtime_config.json"),
                "--seed",
                str(seed),
                "--gpu_id",
                str(gpu),
                "--details",
                f"{phase}, seed={seed}",
                "--run_timestamp",
                run_id,
                "--phase",
                phase,
            ]
            plan.append(
                {
                    "phase": phase,
                    "seed": seed,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "command": command,
                    "final_test_csv": f"{run_id}/logs/{run_id}/final_test_metrics.csv",
                }
            )
    return plan


def environment_record():
    versions = {}
    for package in (
        "torch",
        "dgl",
        "transformers",
        "graph4nlp",
        "numpy",
        "pandas",
        "scikit-learn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED_at_launch": os.environ.get("PYTHONHASHSEED"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the baseline and iterative models.")
    parser.add_argument(
        "--mode", choices=("baseline", "iterative", "both"), default="both",
        help="Model to train (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.txt, dev.txt and test.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for run outputs",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="CUDA index visible to this process"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[30, 31, 32],
        help="Seeds to run, selected from 30, 31 and 32 (default: all three)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without importing ML libraries or writing files",
    )
    args = parser.parse_args(argv)
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    try:
        plan = make_plan(args.mode, args.seeds, data_dir, output_dir, args.gpu)
    except ValueError as error:
        parser.error(str(error))
    for run in plan:
        print(shlex.join(run["command"]))
    if args.dry_run:
        return 0

    from scripts.check_data import check_data

    check_data(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": "tdp-paper-runs/v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": environment_record(),
        "data_sha256": {
            s: sha256(data_dir / f"{s}.txt") for s in ("train", "dev", "test")
        },
        "source_sha256": source_hashes(),
        "config_sha256": {
            p.name: sha256(p) for p in sorted((ROOT / "configs").glob("*.json"))
        },
        "status": "running",
        "runs": [dict(run, status="pending") for run in plan],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for record in manifest["runs"]:
        run = record
        run_dir = Path(run["run_dir"])
        run_dir.mkdir()
        runtime = json.loads((ROOT / "configs/runtime.json").read_text())
        runtime.update(
            {f"{s}_file": str(data_dir / f"{s}.txt") for s in ("train", "dev", "test")}
        )
        (run_dir / "runtime_config.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["TDP_OUTPUT_DIR"] = str(run_dir)
        env["TDP_EXPECTED_CONFIG"] = str(ROOT / "configs" / f"{run['phase']}.json")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        record["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            result = subprocess.run(run["command"], cwd=run_dir, env=env, check=False)
        except (KeyboardInterrupt, OSError) as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["status"] = manifest["status"] = status
            if isinstance(error, OSError):
                record["error"] = str(error)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        final_csv = output_dir / run["final_test_csv"]
        record["returncode"] = result.returncode
        record["status"] = (
            "complete" if result.returncode == 0 and final_csv.is_file() else "failed"
        )
        if record["status"] == "complete":
            from summarize import validate_rows

            try:
                with final_csv.open(newline="") as handle:
                    saved = list(csv.DictReader(handle))
                if len(saved) != 1 or saved[0].get("timestamp") != run["run_id"]:
                    raise ValueError("Unexpected final result identity or row count.")
                validate_rows([dict(saved[0], phase=run["phase"], seed=run["seed"])])
            except (ValueError, KeyError, OSError) as error:
                record["status"] = "failed"
                record["validation_error"] = str(error)
        if record["status"] != "complete":
            manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if record["status"] != "complete":
            raise RuntimeError(
                f"Run {run['run_id']} failed or did not produce final_test_metrics.csv; inspect its logs."
            )
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Completed {len(plan)} runs. Summarize with: python summarize.py --runs {shlex.quote(str(manifest_path))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
