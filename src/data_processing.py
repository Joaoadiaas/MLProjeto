"""
data_processing.py
-------------------
Cleaning + feature engineering for the churn dataset.

This module is imported by both train.py (offline, batch) and api/main.py
(online, single-record) so that training and serving always share the exact
same transformation logic - avoiding train/serve skew.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DIR = Path("data/processed")
FEATURE_LIST_PATH = Path("models/feature_list.json")

TARGET = "Churn"
ID_COL = "customerID"

BINARY_YESNO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]
MULTI_CATEGORICAL = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges sometimes arrives as blank strings for tenure==0 in the
    # original IBM dataset - coerce defensively either way.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df = df.dropna(subset=NUMERIC + BINARY_YESNO + MULTI_CATEGORICAL)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a handful of derived features that tend to help tree models."""
    df = df.copy()
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["tenure_years"] = df["tenure"] / 12.0
    df["num_addons"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    return df


def build_feature_frame(df: pd.DataFrame, fit_columns=None):
    """One-hot encodes categoricals and returns (X, columns_used).

    If `fit_columns` is provided (inference time), the output is reindexed
    to exactly match the training-time column set - filling missing dummy
    columns with 0 and dropping any unseen categories.
    """
    df = engineer_features(clean(df))

    for col in BINARY_YESNO:
        df[col] = (df[col] == "Yes").astype(int)

    X = pd.get_dummies(df, columns=MULTI_CATEGORICAL, drop_first=False)

    drop_cols = [c for c in [ID_COL, TARGET] if c in X.columns]
    X = X.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number, bool]).astype(float)

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0.0)

    return X, list(X.columns)


def _stratified_split(df: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Manual stratified train/test split (keeps this module dependency-free
    of scikit-learn, which is only required for the actual modeling step)."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for label in y.unique():
        idx = np.where(y.values == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(set(range(len(df))) - set(test_idx)))
    return df.iloc[train_idx], df.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def load_and_split(test_size=0.2, seed=42):
    df = pd.read_csv(RAW_PATH)
    df = clean(df)
    df = df.reset_index(drop=True)
    y = (df[TARGET] == "Yes").astype(int)

    df_train, df_test, y_train, y_test = _stratified_split(df, y, test_size, seed)

    X_train, feature_cols = build_feature_frame(df_train)
    X_test, _ = build_feature_frame(df_test, fit_columns=feature_cols)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), feature_cols


def save_processed(X_train, X_test, y_train, y_test, feature_cols):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    X_train.assign(Churn=y_train.values).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(Churn=y_test.values).to_csv(PROCESSED_DIR / "test.csv", index=False)
    FEATURE_LIST_PATH.write_text(json.dumps(feature_cols, indent=2))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = load_and_split()
    save_processed(X_train, X_test, y_train, y_test, feature_cols)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(feature_cols)}")
    print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")
