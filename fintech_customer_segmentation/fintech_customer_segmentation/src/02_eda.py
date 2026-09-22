"""
02_eda.py
---------
Exploratory analysis of the fintech user base before clustering: feature
distributions, correlations, and scale differences that motivate
standardizing before applying K-means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/fintech_customer_segmentation/data/fintech_users.csv"
PLOTS = "/home/claude/fintech_customer_segmentation/plots"

df = pd.read_csv(DATA)

feature_cols = [
    "age", "monthly_income", "savings_rate", "pct_spend_discretionary",
    "app_opens_per_week", "avg_transaction_value", "transactions_per_month",
    "investment_balance", "credit_utilization", "monthly_spend",
]

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print(df[feature_cols].describe().T)

# --- 1. Distributions -----------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, col in zip(axes.flat, feature_cols):
    sns.histplot(df[col], bins=35, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_feature_distributions.png")
plt.close()

# --- 2. Correlation heatmap -------------------------------------------------
plt.figure(figsize=(9, 7))
sns.heatmap(df[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Behavioral & Financial Features")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_correlation_heatmap.png")
plt.close()

# --- 3. A couple of intuitive bivariate views (no labels used yet) -------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(df["monthly_income"], df["investment_balance"], alpha=0.3, s=12)
axes[0].set_xlabel("Monthly Income")
axes[0].set_ylabel("Investment Balance")
axes[0].set_title("Income vs. Investment Balance")

axes[1].scatter(df["credit_utilization"], df["savings_rate"], alpha=0.3, s=12, color="#C44E52")
axes[1].set_xlabel("Credit Utilization")
axes[1].set_ylabel("Savings Rate")
axes[1].set_title("Credit Utilization vs. Savings Rate")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_bivariate_views.png")
plt.close()

print("\nSaved 3 EDA plots to", PLOTS)
