# Customer Churn Prediction — End-to-End ML System

A production-style machine learning project: predicts which telecom customers
are likely to cancel their subscription, and ships the model as a real
service (API + dashboard) with tests, monitoring, and CI — not just a
notebook.

Built to demonstrate the full lifecycle a data scientist / ML engineer role
expects: data understanding, modeling, evaluation, deployment, testing, and
production monitoring.

## Why this project

Most churn-prediction portfolios stop at a notebook with an accuracy score.
This one goes further and answers the questions a hiring team actually cares
about: *can you ship a model that other systems can call, can you test it,
and would you know if it silently broke in production?*

## Architecture

```
                     ┌───────────────────┐
                     │  telco_churn.csv  │
                     └─────────┬─────────┘
                               │
                     data_processing.py  (clean + feature engineering,
                               │          shared by training AND serving)
                               │
                     ┌─────────┴─────────┐
                     │     train.py      │  trains & compares 3 models,
                     │                   │  picks best by ROC-AUC
                     └─────────┬─────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
      models/churn_model.pkl   │           models/metrics.json
              │                │
   ┌──────────┴─────────┐      │
   │   FastAPI service   │     │        monitoring.py
   │   (api/main.py)     │     │        PSI drift check between
   │   POST /predict     │     │        reference vs. current data
   └──────────┬──────────┘     │
              │                │
   ┌──────────┴──────────┐     │
   │ Streamlit dashboard  │────┘
   │ (app/dashboard.py)   │
   │ single + batch scoring, SHAP explainability
   └──────────────────────┘
```

## Dataset note

The classic reference for this task is the IBM/Kaggle **Telco Customer
Churn** dataset. This repo generates a **synthetic dataset with the identical
schema** (`src/generate_data.py`) instead of shipping the original file, with
realistic, tunable relationships between features and churn (e.g.
month-to-month contracts and low tenure meaningfully raise churn risk, same
as in the real data).

To use the real dataset instead: download it and drop it at
`data/raw/telco_churn.csv` with the same column names — every downstream
script depends only on the schema, not on the generator.

## Project structure

```
├── data/raw/telco_churn.csv     # dataset (generated or real)
├── notebooks/01_eda.ipynb       # exploratory analysis
├── src/
│   ├── generate_data.py         # synthetic data generator
│   ├── data_processing.py       # cleaning + feature engineering (shared train/serve)
│   ├── train.py                 # trains + compares models, saves best
│   └── monitoring.py            # PSI-based data drift detection
├── api/
│   ├── main.py                  # FastAPI service (/predict, /health, /model-info)
│   └── schemas.py                # request/response validation
├── app/dashboard.py              # Streamlit UI: single + batch scoring, SHAP
├── tests/                        # pytest suite
├── models/                       # trained artifacts (generated, gitignored)
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml      # test + lint on every push
```

## Getting started

```bash
git clone <this-repo>
cd churn-prediction-mlops   # or wherever you cloned it
python -m venv .venv && source .venv/bin/activate
make setup          # pip install -r requirements.txt

make data           # generates data/raw/telco_churn.csv
make train          # trains 3 models, saves the best to models/
make test           # runs the pytest suite

make api            # http://localhost:8000/docs
make dashboard      # http://localhost:8501
```

Or with Docker:

```bash
docker compose up --build
```

## Modeling approach

`src/train.py` trains and compares three standard classifiers on a
stratified 80/20 split:

| Model | Why it's included |
|---|---|
| Logistic Regression (scaled, `class_weight="balanced"`) | Interpretable baseline; coefficients are directly explainable to business stakeholders |
| Random Forest | Captures non-linear interactions (e.g. contract × tenure) without manual feature crosses |
| Gradient Boosting | Usually the strongest tabular-data performer; benchmark for the other two |

The best model is selected by **ROC-AUC** on the held-out test set (a better
metric than accuracy here, since churn is imbalanced at ~27-29%). Metrics for
every candidate are saved to `models/metrics.json` for full transparency —
not just the winner.

**Results (test set, stratified 80/20 split):**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.824 | 0.745 | 0.541 | 0.797 | 0.644 |
| Random Forest **(selected)** | 0.824 | 0.737 | 0.533 | 0.780 | 0.633 |
| Gradient Boosting | 0.822 | 0.767 | 0.625 | 0.489 | 0.549 |

Random Forest was selected by ROC-AUC (its edge over logistic regression is
marginal — both are reasonable choices, and the interpretable logistic
regression would be the safer pick if stakeholders need to see *why* a
prediction was made). Recall is prioritized over precision in the business
framing: missing a customer who's about to churn is costlier than a false
alarm, since retention offers are cheap relative to losing the account.

Re-run `python src/train.py` to regenerate this table with fresh numbers.

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0
  }'
```

```json
{ "churn_probability": 0.78, "risk_label": "high", "model_version": "gradient_boosting" }
```

## Monitoring

`src/monitoring.py` computes the **Population Stability Index (PSI)** per
feature between a reference set (training data) and current data (e.g. last
week's traffic), the standard technique for detecting silent data drift in
production ML:

```bash
python src/monitoring.py --reference data/processed/train.csv \
                          --current data/processed/test.csv
```

PSI < 0.1 → stable · 0.1–0.2 → moderate drift (watch) · > 0.2 → significant
drift (investigate/retrain). This was validated with a synthetic drift
injection test in `tests/test_monitoring.py`.

## Testing

```bash
pytest -v
```

- `test_data_processing.py` — cleaning, feature engineering, and train/test
