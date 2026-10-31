"""Generate a realistic, deliberately messy Customer Churn dataset.

It contains missing values, duplicate rows, outliers and categorical columns so that
every preprocessing step of the lab (EDA, null handling, duplicates, outliers,
encoding, scaling) has something real to do.
You can replace data/churn.csv with any CSV (e.g. a Kaggle dataset) and pick the
target column in the app.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def make_churn(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Age": rng.integers(18, 80, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No service"], n, p=[0.35, 0.45, 0.20]),
        "PaymentMethod": rng.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n),
        "TenureMonths": rng.integers(1, 72, n),
        "SupportCalls": rng.poisson(2, n),
        "MonthlyCharges": np.round(rng.normal(65, 25, n).clip(15, 130), 2),
    })
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * df["TenureMonths"] * rng.uniform(0.9, 1.1, n), 2)

    # churn probability driven by realistic factors
    logit = (
        -1.0
        + 1.3 * (df["Contract"] == "Month-to-month")
        - 0.9 * (df["Contract"] == "Two year")
        + 0.6 * (df["InternetService"] == "Fiber optic")
        + 0.5 * (df["PaymentMethod"] == "Electronic check")
        + 0.015 * (df["MonthlyCharges"] - 65)
        - 0.03 * df["TenureMonths"]
        + 0.25 * df["SupportCalls"]
        + rng.normal(0, 0.6, n)
    )
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # ---- inject mess ----
    for col, frac in [("Age", 0.04), ("MonthlyCharges", 0.05), ("Contract", 0.03), ("TotalCharges", 0.03)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    out_idx = rng.choice(n, 25, replace=False)
    df.loc[out_idx[:12], "MonthlyCharges"] = rng.uniform(300, 500, 12)       # outliers
    df.loc[out_idx[12:], "SupportCalls"] = rng.integers(20, 40, 13)          # outliers
    dups = df.sample(60, random_state=1)
    df = pd.concat([df, dups], ignore_index=True)                           # duplicates
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    make_churn().to_csv("data/churn.csv", index=False)
    print("Saved data/churn.csv")
