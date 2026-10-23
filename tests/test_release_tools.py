"""Standard-library checks for launch safety, result aggregation and provenance."""

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
import summarize
from scripts.check_data import check_data
from tdp.release_support import validate_paper_config, validate_supported_model


class ReleaseToolsTests(unittest.TestCase):

    def test_paper_architectures_and_removed_options(self):
        for phase in ("baseline", "iterative"):
            combo = json.loads((ROOT / "configs" / f"{phase}.json").read_text())["param_combination"]
            validate_supported_model(SimpleNamespace(**combo), "BERT")
            for key, value in {
                "MAX_ITERATIONS": 5,
                "ENCODER_TYPE": "ROBERTA",
                "GNN_TYPE": "GUNET",
                "SPARSIFIER_TYPE": "ENTMAX",
                "freeze_encoder": "FREEZE_ALL",
                "graph_scorer_type": "bilinear",
                "edge_scorer_type": "bilinear",
                "UPDATE_ADJ_STRATEGY": "ANCHOR",
                "USE_EXTRA_FEATURES": False,
                "BPTT": False,
                "GUMBEL_TEMP_SCHEDULE": "exp",
            }.items():
                with self.subTest(phase=phase, option=key):
                    changed = dict(combo, **{key: value})
                    with self.assertRaises(ValueError):
                        validate_supported_model(SimpleNamespace(**changed), "BERT")
        # The baseline's <X> marker evaluates to true.
        baseline = json.loads((ROOT / "configs/baseline.json").read_text())["param_combination"]
        baseline["BINARIZE"] = False
        with self.assertRaises(ValueError):
            validate_supported_model(SimpleNamespace(**baseline), "BERT")
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for split in ("train", "dev", "test"):
            (self.data / f"{split}.txt").write_text("test fixture\n")
        self.output = self.root / "outputs"
        with (ROOT / "results/paper_per_seed.csv").open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def test_dry_run_never_launches_or_writes(self):
        before = set(sys.modules)
        with (
            patch("train.subprocess.run") as child,
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            result = train.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--output-dir",
                    str(self.output),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        child.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(printed.getvalue().splitlines()), 6)
        self.assertFalse({"torch", "dgl", "transformers"} & (set(sys.modules) - before))
        plan = train.make_plan("both", [30, 31, 32], self.data, self.output, 0)
        self.assertEqual(len({r["run_dir"] for r in plan}), 6)
        for run in plan:
            command = run["command"]
            self.assertEqual(
                command[command.index("--phase") + 1], run["phase"],
            )
            self.assertNotIn("--fixed_max_iterations", command)
            self.assertNotIn("--combo_json", command)

    def test_existing_outputs_and_invalid_inputs_rejected(self):
        self.output.mkdir()
        (self.output / "existing.txt").write_text("keep")
        with self.assertRaises(ValueError):
            train.make_plan("both", [30], self.data, self.output, 0)
        self.assertEqual((self.output / "existing.txt").read_text(), "keep")
        for seeds in ([], [30, 30], [29]):
            with self.assertRaises(ValueError):
                train.make_plan("both", seeds, self.data, self.root / "new", 0)
        (self.data / "dev.txt").unlink()
        with self.assertRaises(ValueError):
            train.make_plan("both", [30], self.data, self.root / "new", 0)

    def test_source_hashes_exclude_environments_and_outputs(self):
        included = [
            "train.py", "summarize.py", "report.py", "tdp/model.py",
            "tdp/data_handler/features.py", "scripts/check_data.py", "tests/test_example.py",
        ]
        excluded = [
            ".venv/lib/python3.10/site-packages/package.py", "outputs/copy.py",
            "scratch.py", ".git/example.py",
        ]
        for name in included + excluded:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        with patch.object(train, "ROOT", self.root):
            hashes = train.source_hashes()
        self.assertEqual(set(hashes), set(included))
        for name in included:
            self.assertEqual(hashes[name], hashlib.sha256(name.encode()).hexdigest())

    def test_interrupted_and_failed_launches_update_manifest(self):
        cases = [
            (KeyboardInterrupt(), "interrupted"),
            (OSError("Unable to start process"), "failed"),
        ]
        for error, status in cases:
            with self.subTest(status=status):
                output = self.root / status
                with (
                    patch("scripts.check_data.check_data"),
                    patch("train.environment_record", return_value={}),
                    patch("train.subprocess.run", side_effect=error) as child,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(type(error)),
                ):
                    train.main([
                        "--data-dir", str(self.data), "--output-dir", str(output),
                    ])
                child.assert_called_once()
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], status)
                self.assertEqual(manifest["runs"][0]["status"], status)
                self.assertTrue(all(r["status"] == "pending" for r in manifest["runs"][1:]))
                if isinstance(error, OSError):
                    self.assertEqual(manifest["runs"][0]["error"], str(error))

    def test_data_mismatch_stops_before_creating_outputs(self):
        with self.assertRaisesRegex(ValueError, "Checksum differs"):
            check_data(self.data)
        with (
            patch("train.subprocess.run") as child,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "Checksum differs"):
                train.main(
                    ["--data-dir", str(self.data), "--output-dir", str(self.output)]
                )
        child.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_archived_results_match_paper(self):
        result = summarize.summarize(self.rows)
        means = [round(r["mean_percent"], 2) for r in result["summary"]]
        spreads = [round(r["sample_std_percent"], 2) for r in result["summary"]]
        self.assertEqual(
            means, [66.56, 73.97, 60.99, 71.53, 68.95, 76.06, 63.98, 73.96]
        )
        self.assertEqual(spreads, [0.19, 0.31, 0.16, 0.44, 0.80, 0.81, 0.71, 0.77])
        self.assertEqual(
            [
                round(v, 2)
                for v in result["mean_paired_deltas_percentage_points"].values()
            ],
            [2.38, 2.09, 2.99, 2.43],
        )
        one = summarize.summarize(self.rows[:1])
        self.assertIsNone(one["summary"][0]["sample_std_percent"])

    def test_invalid_statistics_rejected(self):
        with self.assertRaises(ValueError):
            summarize.summarize(self.rows + self.rows[:1])
        with self.assertRaises(ValueError):
            summarize.summarize(self.rows[:-1])
        for invalid in ("nan", "inf", "-0.1", "1.1"):
            rows = copy.deepcopy(self.rows)
            rows[0][summarize.METRICS[0]] = invalid
            with self.assertRaises(ValueError):
                summarize.summarize(rows)

    def test_manifest_completeness_and_result_identity(self):
        row = dict(self.rows[0])
        run_id = "taln_baseline_s30"
        row["timestamp"] = run_id
        csv_path = self.root / "final.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        record = {
            "run_id": run_id,
            "phase": "baseline",
            "seed": 30,
            "status": "complete",
            "final_test_csv": "final.csv",
        }
        manifest = {
            "schema": "tdp-paper-runs/v1",
            "status": "running",
            "runs": [record],
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            summarize.load_runs(path)
        manifest["status"] = "complete"
        path.write_text(json.dumps(manifest))
        self.assertEqual(len(summarize.load_runs(path)), 1)
        record["run_id"] = "wrong_run"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            summarize.load_runs(path)
        record["status"] = "pending"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize.load_runs(path)

    def test_fixed_config_rejects_changed_settings(self):
        for phase in ("baseline", "iterative"):
            path = ROOT / "configs" / f"{phase}.json"
            combo = json.loads(path.read_text())["param_combination"]
            with patch.dict("os.environ", {"TDP_EXPECTED_CONFIG": str(path)}):
                validate_paper_config(combo)
                wrong = dict(combo, gamma=2)
                with self.assertRaises(ValueError):
                    validate_paper_config(wrong)
                wrong = dict(combo, alphas=[1] * 5)
                with self.assertRaises(ValueError):
                    validate_paper_config(wrong)

    def test_archived_export_hashes(self):
        manifest = json.loads((ROOT / "results/provenance.json").read_text())
        self.assertEqual(len(manifest["runs"]), 6)
        for name, expected in manifest["export_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest(),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
