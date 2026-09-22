"""
03_clustering.py
------------------
Unsupervised segmentation of the fintech user base:
  1. Standardize features (required for distance-based clustering).
  2. Choose k via elbow method + silhouette score.
  3. Fit K-means, profile each cluster's average behavior.
  4. Visualize clusters in 2D via PCA.
  5. Cross-check discovered clusters against the "true" generating
     personas (available here because the data is synthetic; in a real
     project you wouldn't have this ground truth -- this step exists only
     to sanity-check that the method works).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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

X = df[feature_cols].copy()

# Monetary features are right-skewed (a few high-income/high-investment users
# stretch the scale) -- log-transform them first so no single feature
# dominates the Euclidean distance K-means relies on after standardizing.
skewed_features = ["monthly_income", "investment_balance", "avg_transaction_value", "monthly_spend"]
for col in skewed_features:
    X[col] = np.log1p(X[col])

X_scaled = StandardScaler().fit_transform(X)

# --- 1. Elbow method + silhouette score to choose k ------------------------
inertias, sil_scores = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_xlabel("k (number of clusters)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#C44E52")
axes[1].set_xlabel("k (number of clusters)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_k_selection.png")
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Silhouette scores: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"Best k by silhouette score: {best_k}")

# --- 2. Fit final K-means model --------------------------------------------
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# --- 3. PCA visualization ---------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="tab10", alpha=0.6, s=25)
plt.title(f"User Segments in 2D (PCA) — k={best_k}\n"
          f"(explains {pca.explained_variance_ratio_.sum():.1%} of variance)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_pca_clusters.png")
plt.close()

# --- 4. Profile each cluster -------------------------------------------------
profile = df.groupby("cluster")[feature_cols].mean().round(2)
profile["n_users"] = df["cluster"].value_counts().sort_index()
print("\nCLUSTER PROFILES:\n", profile)

# Normalized heatmap of cluster profiles for easy visual comparison
profile_z = (profile[feature_cols] - profile[feature_cols].mean()) / profile[feature_cols].std()
plt.figure(figsize=(10, max(4, best_k * 0.8)))
sns.heatmap(profile_z, annot=profile[feature_cols].values, fmt=".1f",
            cmap="RdBu_r", center=0, cbar_kws={"label": "Relative to other clusters (z-score)"})
plt.title("Cluster Profiles (values = actual mean, color = relative level)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_cluster_profile_heatmap.png")
plt.close()

# --- 5. Sanity-check against the synthetic ground-truth personas -----------
cross_tab = pd.crosstab(df["cluster"], df["true_persona"])
print("\nCROSS-TAB vs. true generating persona (sanity check only):\n", cross_tab)

plt.figure(figsize=(9, 5))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
plt.title("Discovered Clusters vs. True Generating Persona (sanity check)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_cluster_vs_true_persona.png")
plt.close()

# Save cluster profile + evaluation summary
summary = {
    "best_k": int(best_k),
    "silhouette_scores_by_k": {int(k): round(s, 4) for k, s in zip(k_range, sil_scores)},
    "final_silhouette_score": round(sil_scores[list(k_range).index(best_k)], 4),
    "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
    "cluster_sizes": df["cluster"].value_counts().sort_index().to_dict(),
}
with open("/home/claude/fintech_customer_segmentation/results.json", "w") as f:
    json.dump(summary, f, indent=2)

profile.to_csv("/home/claude/fintech_customer_segmentation/cluster_profiles.csv")

print("\nSaved elbow/silhouette, PCA scatter, profile heatmap, and sanity-check plots.")
