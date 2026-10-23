"""Fixed model settings and setup RNG state."""

import json
from pathlib import Path
import random

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_parameters(phase):
    if phase not in ("baseline", "iterative"):
        raise ValueError("Choose baseline or iterative.")
    config = json.loads((CONFIG_DIR / f"{phase}.json").read_text())
    if config["schema"] != "tdp-paper-config/v1" or config["phase"] != phase:
        raise ValueError("Invalid paper configuration.")
    return config["param_combination"]


def advance_setup_rng():
    """Advance to the setup RNG state used by the paper experiments.
    """
    random.sample(range(6), 5)
    random.sample(range(1), 1)
    indices = list(range(6))
    random.shuffle(indices)
