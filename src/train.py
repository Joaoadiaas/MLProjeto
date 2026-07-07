"""
train.py
--------
Trains and compares three standard classifiers for churn prediction,
selects the best by ROC-AUC on a held-out test set, and persists:

    models/churn_model.pkl     - the winning fitted pipeline (joblib)
    models/feature_list.json   - exact column order expected at inference
    models/metrics.json        - metrics for every candidate model
    models/plots/*.png         - ROC curve, confusion matrix, feature importance

Run:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_processing import load_and_split, save_processed

MODELS_DIR = Path("models")
PLOTS_DIR = MODELS_DIR / "plots"


def get_candidates():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
            )),
        ]),
    }


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols = load_and_split()
    save_processed(X_train, X_test, y_train, y_test, feature_cols)

    candidates = get_candidates()
    results = {}
    fitted = {}

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipe
        print(f"{name:20s} -> {metrics}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    (MODELS_DIR / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "candidates": results}, indent=2)
    )

    # ---- plots ----
    proba = best_model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC Curve - {best_name}")
    plt.savefig(PLOTS_DIR / "roc_curve.png", bbox_inches="tight", dpi=120)
    plt.close()

    pred = (proba >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No churn", "Churn"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.close()

    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feature_cols)
        importances.nlargest(15).sort_values().plot(kind="barh", figsize=(7, 6))
        plt.title(f"Top 15 feature importances - {best_name}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=120)
        plt.close()
    elif hasattr(clf, "coef_"):
        importances = pd.Series(clf.coef_[0], index=feature_cols)
        importances.reindex(importances.abs().nlargest(15).index).sort_values().plot(
            kind="barh", figsize=(7, 6)
        )
        plt.title(f"Top 15 coefficients (|weight|) - {best_name}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=120)
        plt.close()

    print(f"\nArtifacts written to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
