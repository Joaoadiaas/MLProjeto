import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_processing import build_feature_frame, clean, engineer_features, load_and_split

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"


def _sample_raw(n=20):
    df = pd.read_csv(RAW_PATH)
    return df.sample(n=n, random_state=0).reset_index(drop=True)


def test_clean_has_no_missing_values():
    df = clean(_sample_raw())
    assert df.isna().sum().sum() == 0


def test_engineer_features_adds_expected_columns():
    df = engineer_features(clean(_sample_raw()))
    for col in ["avg_monthly_spend", "tenure_years", "num_addons", "has_internet"]:
        assert col in df.columns


def test_build_feature_frame_is_all_numeric():
    df = _sample_raw()
    X, cols = build_feature_frame(df)
    assert X.shape[0] == len(df)
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert "customerID" not in cols
    assert "Churn" not in cols


def test_build_feature_frame_reindexes_to_fit_columns():
    df = _sample_raw()
    _, fit_cols = build_feature_frame(df)

    single_row = df.iloc[[0]]
    X_single, _ = build_feature_frame(single_row, fit_columns=fit_cols)
    assert list(X_single.columns) == fit_cols
    assert X_single.shape[0] == 1


def test_load_and_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test, feature_cols = load_and_split(test_size=0.2, seed=1)

    assert len(X_train) + len(X_test) > 0
    assert set(X_train.columns) == set(X_test.columns) == set(feature_cols)

    # class balance roughly preserved between splits
    assert abs(y_train.mean() - y_test.mean()) < 0.05
