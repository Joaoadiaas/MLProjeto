"""
generate_data.py
-----------------
Generates a synthetic customer-churn dataset for a telecom company.

WHY SYNTHETIC DATA?
The classic reference dataset for this kind of project is the IBM/Kaggle
"Telco Customer Churn" dataset. This script does NOT download it (no
internet access was available in the environment where this was built) -
instead it generates a dataset with the *same schema* and *realistic,
controllable relationships* between features and churn, so the whole
pipeline (EDA -> feature engineering -> training -> API -> monitoring)
can be built, run and demoed end-to-end.

To use the real dataset instead, just drop a CSV with the same column
names into data/raw/telco_churn.csv - every downstream script only
depends on the schema, not on this generator.

Usage:
    python src/generate_data.py --n 7043 --seed 42
"""
import argparse
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # tenure in months, skewed towards both new and very long customers
    tenure = np.clip(rng.gamma(shape=2.0, scale=16, size=n), 0, 72).round().astype(int)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def addon(p_yes):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes]),
        )

    online_security = addon(0.29)
    online_backup = addon(0.34)
    device_protection = addon(0.34)
    tech_support = addon(0.29)
    streaming_tv = addon(0.38)
    streaming_movies = addon(0.39)

    # ---- monthly charges: base + add-ons, correlated with internet type ----
    base = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [70.0, 45.0, 20.0],
    )
    addon_cost = sum(
        np.where(a == "Yes", rng.uniform(4, 10, size=n), 0.0)
        for a in [
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
        ]
    )
    phone_cost = np.where(phone_service == "Yes", rng.uniform(15, 25, size=n), 0.0)
    noise = rng.normal(0, 3, size=n)
    monthly_charges = np.clip(base + addon_cost + phone_cost + noise, 18.25, 120.0).round(2)

    total_charges = np.clip(
        monthly_charges * tenure + rng.normal(0, 20, size=n), 0, None
    ).round(2)
    # brand-new customers legitimately have small/zero total charges
    total_charges = np.where(tenure == 0, monthly_charges, total_charges)

    customer_id = [f"{rng.integers(1000,9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)]

    # ---------------- churn probability (the "ground truth" signal) ----------------
    contract_risk = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [1.1, -0.4, -1.3],
    )
    tenure_risk = -0.045 * tenure
    charges_risk = 0.018 * (monthly_charges - 65)
    fiber_risk = np.where(internet_service == "Fiber optic", 0.35, 0.0)
    no_support_risk = np.where(tech_support == "No", 0.30, 0.0)
    no_security_risk = np.where(online_security == "No", 0.20, 0.0)
    paperless_risk = np.where(paperless_billing == "Yes", 0.15, 0.0)
    echeck_risk = np.where(payment_method == "Electronic check", 0.25, 0.0)
    senior_risk = np.where(senior_citizen == 1, 0.20, 0.0)
    partner_protect = np.where(partner == "Yes", -0.15, 0.0)
    dependents_protect = np.where(dependents == "Yes", -0.20, 0.0)

    logit = (
        -0.9
        + contract_risk
        + tenure_risk
        + charges_risk
        + fiber_risk
        + no_support_risk
        + no_security_risk
        + paperless_risk
        + echeck_risk
        + senior_risk
        + partner_protect
        + dependents_protect
        + rng.normal(0, 0.55, size=n)  # irreducible noise
    )
    churn_prob = _sigmoid(logit)
    churn = (rng.uniform(size=n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=7043, help="number of customer rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/telco_churn.csv")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
