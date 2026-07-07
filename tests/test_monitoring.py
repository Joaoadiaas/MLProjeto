import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.monitoring import compute_drift_report, _psi_for_column


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=1000))
    psi = _psi_for_column(s, s)
    assert psi < 1e-6


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(loc=0, scale=1, size=2000))
    shifted = pd.Series(rng.normal(loc=3, scale=1, size=2000))
    psi = _psi_for_column(reference, shifted)
    assert psi > 0.2


def test_compute_drift_report_flags_only_shifted_columns():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({
        "stable_feature": rng.normal(size=1000),
        "drifting_feature": rng.normal(size=1000),
        "Churn": rng.integers(0, 2, size=1000),
    })
    current = reference.copy()
    current["drifting_feature"] = rng.normal(loc=5, size=1000)

    report = compute_drift_report(reference, current)
    assert "drifting_feature" in report["flagged_features"]
    assert "stable_feature" not in report["flagged_features"]
    assert "Churn" not in report["features"]
    assert report["retrain_recommended"] is True
