"""
01_generate_data.py
--------------------
Generates a synthetic user base for a fintech app (digital wallet / neobank
style), with behavioral and financial features typical of what such an app
actually logs: income, spend patterns, savings rate, transaction habits,
app engagement, and investment/credit balances.

Five underlying "true" personas are used to generate the data with
realistic within-group variation and overlap -- but no persona label is
given to the clustering step. The point of the project is to see whether
unsupervised clustering can recover meaningful segments from behavior
alone, the way a real fintech growth/product team would have to.
"""

import numpy as np
import pandas as pd

np.random.seed(21)

personas = {
    "Young Digital Spender": dict(
        n=550, age=(20, 28), income=(2200, 4500), savings_rate=(0.01, 0.08),
        pct_discretionary=(0.45, 0.70), app_opens_week=(12, 30),
        avg_txn_value=(15, 60), txns_per_month=(35, 70),
        investment_balance=(0, 500), credit_utilization=(0.2, 0.55),
    ),
    "Budget-Conscious Saver": dict(
        n=480, age=(25, 42), income=(3000, 6500), savings_rate=(0.18, 0.35),
        pct_discretionary=(0.15, 0.30), app_opens_week=(4, 10),
        avg_txn_value=(25, 80), txns_per_month=(15, 35),
        investment_balance=(500, 5000), credit_utilization=(0.05, 0.25),
    ),
    "High-Income Investor": dict(
        n=350, age=(35, 58), income=(8000, 18000), savings_rate=(0.20, 0.40),
        pct_discretionary=(0.25, 0.45), app_opens_week=(3, 8),
        avg_txn_value=(60, 220), txns_per_month=(10, 25),
        investment_balance=(15000, 120000), credit_utilization=(0.02, 0.15),
    ),
    "Credit-Reliant Household": dict(
        n=420, age=(28, 55), income=(2500, 5000), savings_rate=(-0.05, 0.04),
        pct_discretionary=(0.30, 0.50), app_opens_week=(5, 14),
        avg_txn_value=(20, 55), txns_per_month=(20, 45),
        investment_balance=(0, 300), credit_utilization=(0.55, 0.95),
    ),
    "Steady Middle": dict(
        n=700, age=(26, 50), income=(4000, 8000), savings_rate=(0.08, 0.18),
        pct_discretionary=(0.28, 0.42), app_opens_week=(5, 12),
        avg_txn_value=(30, 90), txns_per_month=(18, 40),
        investment_balance=(1000, 12000), credit_utilization=(0.15, 0.40),
    ),
}

rows = []
for name, p in personas.items():
    n = p["n"]
    rows.append(pd.DataFrame({
        "true_persona": name,  # kept aside for validation only, not used in clustering
        "age": np.random.randint(p["age"][0], p["age"][1], n),
        "monthly_income": np.random.uniform(*p["income"], n).round(-1),
        "savings_rate": np.clip(np.random.uniform(*p["savings_rate"], n)
                                  + np.random.normal(0, 0.03, n), -0.1, 0.5).round(3),
        "pct_spend_discretionary": np.clip(
            np.random.uniform(*p["pct_discretionary"], n)
            + np.random.normal(0, 0.04, n), 0.05, 0.85).round(3),
        "app_opens_per_week": np.clip(
            np.random.uniform(*p["app_opens_week"], n)
            + np.random.normal(0, 1.5, n), 0, 40).round(1),
        "avg_transaction_value": np.clip(
            np.random.uniform(*p["avg_txn_value"], n)
            + np.random.normal(0, 8, n), 5, 400).round(2),
        "transactions_per_month": np.clip(
            np.random.uniform(*p["txns_per_month"], n)
            + np.random.normal(0, 4, n), 3, 100).round().astype(int),
        "investment_balance": np.clip(
            np.random.uniform(*p["investment_balance"], n)
            * np.exp(np.random.normal(0, 0.3, n)), 0, None).round(-1),
        "credit_utilization": np.clip(
            np.random.uniform(*p["credit_utilization"], n)
            + np.random.normal(0, 0.04, n), 0, 1).round(3),
        "months_active": np.random.randint(1, 60, n),
    }))

df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=21).reset_index(drop=True)

# monthly_spend derived from income and discretionary share + noise
df["monthly_spend"] = (
    df["monthly_income"] * (1 - df["savings_rate"])
    * (0.85 + 0.3 * np.random.rand(len(df)))
).round(-1).clip(lower=200)

out_path = "/home/claude/fintech_customer_segmentation/data/fintech_users.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["true_persona"].value_counts())
print(df.drop(columns="true_persona").describe().T[["min", "mean", "max"]])
