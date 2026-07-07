"""
monitoring.py
-------------
Lightweight data-drift monitoring using the Population Stability Index (PSI),
the standard metric credit-risk and ML teams use to decide when a production
model needs retraining - no extra dependencies beyond numpy/pandas.

PSI interpretation (industry rule of thumb):
    < 0.1  -> no significant drift
    0.1-0.2 -> moderate drift, watch closely
    > 0.2  -> significant drift, investigate / retrain

Usage:
    python src/monitoring.py --reference data/processed/train.csv \
                              --current data/processed/test.csv \
                              --out models/drift_report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _psi_for_column(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()

    # binary / low-cardinality columns: treat each unique value as its own bucket
    if reference.nunique() <= bins:
        categories = sorted(set(reference.unique()) | set(current.unique()))
        ref_counts = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        cur_counts = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
        ref_binned = pd.cut(reference, bins=edges, include_lowest=True)
        cur_binned = pd.cut(current, bins=edges, include_lowest=True)
        ref_counts = ref_binned.value_counts(normalize=True, sort=False)
        cur_counts = cur_binned.value_counts(normalize=True, sort=False)

    eps = 1e-4
    ref_pct = ref_counts.values + eps
    cur_pct = cur_counts.reindex(ref_counts.index, fill_value=0).values + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame, exclude=("Churn",)) -> dict:
    cols = [c for c in reference.columns if c in current.columns and c not in exclude]
    report = {}
    for col in cols:
        psi = _psi_for_column(reference[col], current[col])
        if psi < 0.1:
            status = "stable"
        elif psi < 0.2:
            status = "moderate_drift"
        else:
            status = "significant_drift"
        report[col] = {"psi": round(psi, 4), "status": status}

    flagged = [c for c, v in report.items() if v["status"] != "stable"]
    return {
        "features": report,
        "flagged_features": flagged,
        "retrain_recommended": len(flagged) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/test.csv")
    parser.add_argument("--out", default="models/drift_report.json")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    report = compute_drift_report(reference, current)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"Checked {len(report['features'])} features.")
    if report["flagged_features"]:
        print(f"Drift flagged in: {report['flagged_features']}")
    else:
        print("No significant drift detected.")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
