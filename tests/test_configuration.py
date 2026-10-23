"""Fixed configuration and initialization regression checks."""

import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdp.configuration import advance_setup_rng, load_parameters


class ConfigurationTests(unittest.TestCase):
    def test_setup_rng_matches_paper_initialization(self):
        expected = {30: 'a66e9b8d18dd342b5e6d1a21141757191a218f8e47e6a2f1edef7d2c5286fe0f', 31: 'fc293ebeb5473f8a774dc4c4ea35659599c04e586a72adcaaf6b4a7bb7dbaae3', 32: '88a652ec5b37960de5ea83fe5098fdfb0e20dfe109ad54d40a9d784be54b79a5'}
        original = random.getstate()
        self.addCleanup(random.setstate, original)
        for seed, digest in expected.items():
            with self.subTest(seed=seed):
                random.seed(seed)
                advance_setup_rng()
                actual = hashlib.sha256(json.dumps(random.getstate()).encode()).hexdigest()
                self.assertEqual(actual, digest)

    def test_fixed_parameters_are_independent(self):
        for phase, iterations in (("baseline", 0), ("iterative", 3)):
            first = load_parameters(phase)
            self.assertEqual(first["MAX_ITERATIONS"], iterations)
            self.assertEqual(first["gamma"], 3)
            self.assertEqual(first["num_epochs"], 35)
            first["alphas"][0] = -1
            self.assertGreater(load_parameters(phase)["alphas"][0], 0)
        with self.assertRaises(ValueError):
            load_parameters("../runtime")


if __name__ == "__main__":
    unittest.main()
