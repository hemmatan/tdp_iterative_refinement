"""Validate the fixed paper configurations."""

import json
import math
import os
from pathlib import Path


def validate_paper_config(actual):
    path = os.environ.get("TDP_EXPECTED_CONFIG")
    if not path:
        return
    expected = json.loads(Path(path).read_text())["param_combination"]
    differences = {}
    for key, target in expected.items():
        value = actual.get(key)
        if key == "alphas":
            if hasattr(value, "tolist"):
                value = value.tolist()
            equal = (
                isinstance(value, list)
                and len(value) == len(target)
                and all(
                    math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)
                    for a, b in zip(value, target)
                )
            )
        else:
            equal = value == target
        if not equal:
            differences[key] = {"expected": target, "actual": str(value)}
    if differences:
        raise ValueError(
            f"Settings differ from the paper configuration: {differences}"
        )


def validate_supported_model(args, encoder_type):
    """Check the supported T=0 and T=3 model settings."""
    if encoder_type != "BERT" or getattr(args, "ENCODER_TYPE", None) != "BERT":
        raise ValueError("The paper configurations require BERT-base-cased.")
    iterations = getattr(args, "MAX_ITERATIONS", None)
    if iterations not in (0, 3):
        raise ValueError("The paper configurations use T=0 or T=3.")
    expected = {
        "freeze_encoder": "NO_FREEZING",
        "graph_scorer_type": "biaffine",
        "edge_scorer_type": "biaffine",
        "USE_EXTRA_FEATURES": True,
        "SPARSIFIER_TYPE": "GUMBEL",
        "GUMBEL_TEMP_SCHEDULE": "cosine",
        "BPTT": True,
        "EDGE_LOSS_AGG": "weighted_mean",
        "GRAPH_LOSS_AGG": "weighted_mean",
        "BINARIZE": "<X>" if iterations == 0 else False,
        "GNN_TYPE": "<X>" if iterations == 0 else "GCN",
        "UPDATE_ADJ_STRATEGY": "<X>" if iterations == 0 else "EMA",
    }
    differences = {
        key: {"expected": value, "actual": getattr(args, key, None)}
        for key, value in expected.items()
        if getattr(args, key, None) != value
    }
    if differences:
        raise ValueError(f"Unsupported paper-model settings: {differences}")
