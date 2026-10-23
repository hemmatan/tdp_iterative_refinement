"""Check paper claims, complete diagnostic coverage and report input isolation."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import report


class PaperReportTests(unittest.TestCase):
    def setUp(self):
        self.data, self.provenance = report.load_inputs()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_paper_diagnostic_claims(self):
        result = report.build_report(self.data)
        curves = [round(r['mean'], 2) for r in result['dev_iterations'] if r['scope'] == 'all']
        self.assertEqual(curves, [58.29, 58.60, 59.03, 59.24, 66.64, 66.93, 67.22, 67.45])
        gains = [(round(r['mean'], 2), round(r['sample_std'], 2))
                 for r in result['dev_gains'] if r['scope'] == 'all']
        self.assertEqual(gains, [(0.95, 0.28), (0.81, 0.23)])
        self.assertEqual([round(r['mean'], 4) for r in result['self_jaccard'] if r['mode'] == 'labeled'],
                         [0.8368, 0.9168, 0.9439])
        self.assertEqual([round(r['mean'], 2) for r in result['distance_gains']],
                         [2.30, -0.89, 0.31, 1.63, -0.29, 0.43])
        self.assertEqual([round(r['mean'], 2) for r in result['relation_gains']],
                         [1.37, -0.27, 0.11, 1.48, 2.51])
        self.assertEqual(report.comparison_table(result)[-1], ['Ours', '59.24', '68.95'])

    def test_missing_duplicate_and_invalid_diagnostics_fail(self):
        for kind, metric in [('iterations', 'f1_doc_macro'), ('stability', 'jaccard_micro'),
                             ('distance', 'f1_micro'), ('relations', 'f1_micro')]:
            for failure in ['missing', 'duplicate', 'nan', 'inf', '-0.01', '1.01', 'seed', 'iteration']:
                with self.subTest(kind=kind, failure=failure):
                    data = copy.deepcopy(self.data)
                    index = next(i for i, row in enumerate(data[kind]) if row['phase'] == 'iterative')
                    if failure == 'missing':
                        data[kind].pop(index)
                    elif failure == 'duplicate':
                        data[kind].append(dict(data[kind][index]))
                    elif failure == 'seed':
                        data[kind][index]['seed'] = '33'
                    elif failure == 'iteration':
                        data[kind][index]['iteration'] = '4'
                    else:
                        data[kind][index][metric] = failure
                    with self.assertRaises(ValueError):
                        report.build_report(data)
        self.data['main'] = self.data['main'][:1]
        with self.assertRaisesRegex(ValueError, 'both models'):
            report.build_report(self.data)

    def write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make_run_fixture(self):
        runs = []
        for row in self.data['main']:
            phase, seed = row['phase'], row['seed']
            run_id = f'taln_{phase}_s{seed}'
            final = self.root / run_id / 'logs' / run_id / 'final_test_metrics.csv'
            self.write_csv(final, [dict(row, timestamp=run_id)])
            runs.append(dict(phase=phase, seed=int(seed), run_id=run_id,
                             status='complete', final_test_csv=str(final.relative_to(self.root))))
            if phase == 'iterative':
                folder = final.parent / 'per_iter' / ('fixture_' + row['model_name'])
                folder.mkdir(parents=True)
                (folder / 'run_metadata.json').write_text(json.dumps(
                    dict(model_name=row['model_name'], hp_timestamp='fixture')))
                for kind, (_, name) in report.DIAGNOSTICS.items():
                    selected = [{k: v for k, v in x.items() if k not in ('phase', 'seed')}
                                for x in self.data[kind] if x['phase'] == phase and x['seed'] == seed]
                    self.write_csv(folder / name, selected)
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(dict(schema='tdp-paper-runs/v1', status='complete', runs=runs)))
        return path

    def test_manifest_path_matches_archived_aggregation(self):
        path = self.make_run_fixture()
        data, provenance = report.load_inputs(runs=path)
        self.assertEqual(report.build_report(data), report.build_report(self.data))
        self.assertNotEqual(provenance['origin'], self.provenance['origin'])
        self.assertTrue(all(not Path(p).is_absolute() for p in provenance['input_sha256']))
        manifest = json.loads(path.read_text())
        manifest['status'] = 'failed'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            report.load_inputs(runs=path)

    def test_missing_diagnostics_and_wrong_identity_fail(self):
        path = self.make_run_fixture()
        metadata = next(self.root.glob('*/logs/*/per_iter/*/run_metadata.json'))
        original = metadata.read_text()
        metadata.write_text(json.dumps(dict(model_name='wrong', hp_timestamp='fixture')))
        with self.assertRaisesRegex(ValueError, 'identity'):
            report.load_inputs(runs=path)
        metadata.write_text(original)
        (metadata.parent / report.DIAGNOSTICS['distance'][1]).unlink()
        with self.assertRaises(FileNotFoundError):
            report.load_inputs(runs=path)

    def test_cli_exports_without_model_libraries_and_rejects_overwrite(self):
        before = set(sys.modules)
        out = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(['--output-dir', str(out), '--no-plot']), 0)
        self.assertFalse({'torch', 'dgl', 'transformers', 'matplotlib'} & (set(sys.modules) - before))
        saved = (out / 'report.json').read_bytes()
        self.assertEqual(json.loads(saved)['provenance']['input_sha256'], self.provenance['input_sha256'])
        self.assertIn('66.56 $\\pm$ 0.19', (out / 'main_results.tex').read_text())
        self.assertEqual(len(list(out.iterdir())), 11)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main(['--output-dir', str(out), '--no-plot'])
        self.assertEqual((out / 'report.json').read_bytes(), saved)

    def test_changed_archived_input_fails_before_output(self):
        archive = self.root / 'results'
        archive.mkdir()
        for p in (ROOT / 'results').glob('*'):
            (archive / p.name).write_bytes(p.read_bytes())
        with (archive / 'paper_dev_distance.csv').open('a') as handle:
            handle.write('\n')
        out = self.root / 'report'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main(['--results-dir', str(archive), '--output-dir', str(out), '--no-plot'])
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
