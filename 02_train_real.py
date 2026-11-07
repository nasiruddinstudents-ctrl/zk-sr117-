"""
02_train_real.py

Run this LOCALLY, in the same folder as hmda_sample.csv.
It trains the logistic regression on REAL HMDA data, computes the same
plaintext metrics as the synthetic prototype, and writes out only small
files (a few KB/MB) -- NOT the raw data -- which you then send back.

Files this script produces (safe/small to upload):
    plaintext_metrics.json   <- weights + metrics, a few KB
    batch_X.npy              <- 64 x 4 float array, tiny
    batch_group.npy          <- 64 floats, tiny
    batch_y.npy              <- 64 floats, tiny

pip install pandas scikit-learn numpy   (if not already installed)
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv("hmda_sample.csv")

FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]

# Protected group definition for the fair-lending attestation (Table 1 Row 4):
# 1 = a HMDA race category other than "White" alone; 0 = "White".
# Rows with unusable race codes are dropped for this attestation, consistent
# with how fair-lending analysis typically excludes "Not Available"/"Joint"
# from a binary majority/minority comparison -- note this simplification
# explicitly in the paper's limitations if it ships this way.
df = df[df["derived_race"].isin(["White", "Black or African American",
                                   "Asian", "American Indian or Alaska Native",
                                   "Native Hawaiian or Other Pacific Islander"])].copy()
df["protected_group"] = (df["derived_race"] != "White").astype(int)

X = df[FEATURES].values
y = df["action_taken_binary"].values
group = df["protected_group"].values

X_train, X_hold, y_train, y_hold, g_train, g_hold = train_test_split(
    X, y, group, test_size=0.25, random_state=11, stratify=y
)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_std = (X_train - mu) / sigma
X_hold_std = (X_hold - mu) / sigma

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_std, y_train)

probs_hold = clf.predict_proba(X_hold_std)[:, 1]
preds_hold = (probs_hold >= 0.5).astype(int)

auc = roc_auc_score(y_hold, probs_hold)
fpr, tpr, _ = roc_curve(y_hold, probs_hold)
ks = float(np.max(tpr - fpr))

approve_rate_g1 = preds_hold[g_hold == 1].mean()
approve_rate_g0 = preds_hold[g_hold == 0].mean()
dp_gap = abs(approve_rate_g1 - approve_rate_g0)

metrics = {
    "n_holdout": int(len(y_hold)),
    "auc": float(auc),
    "ks": ks,
    "approve_rate_nonwhite": float(approve_rate_g1),
    "approve_rate_white": float(approve_rate_g0),
    "demographic_parity_gap": float(dp_gap),
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
    "feature_mean": mu.tolist(),
    "feature_std": sigma.tolist(),
    "features": FEATURES,
    "data_source": "real HMDA 2022, filtered conventional/first-lien/purchase/principal-residence",
}
with open("plaintext_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

N_BATCH = 64
rng = np.random.default_rng(7)
# Try to get a mixed batch (some approvals, some denials, both groups) --
# real approval rates are high (~93%), so a pure random N=64 draw may have
# very few denials. Stratify lightly to keep the demo meaningful.
idx_pool = np.arange(len(X_hold_std))
idx = rng.choice(idx_pool, size=N_BATCH, replace=False)

batch_X = X_hold_std[idx].astype(np.float32)
batch_group = g_hold[idx].astype(np.float32)
batch_y = y_hold[idx].astype(np.float32)

np.save("batch_X.npy", batch_X)
np.save("batch_group.npy", batch_group)
np.save("batch_y.npy", batch_y)

batch_probs = clf.predict_proba(batch_X)[:, 1]
batch_preds = (batch_probs >= 0.5).astype(int)
n1, n0 = (batch_group == 1).sum(), (batch_group == 0).sum()
batch_dp_gap = (
    abs(batch_preds[batch_group == 1].mean() - batch_preds[batch_group == 0].mean())
    if n1 > 0 and n0 > 0 else float("nan")
)

print(f"Real HMDA sample: {len(df)} rows after race-category filtering")
print(f"Held-out slice (n={len(y_hold)}):")
print(f"  AUC = {auc:.4f}")
print(f"  KS  = {ks:.4f}")
print(f"  Demographic parity gap (nonwhite vs white) = {dp_gap:.4f}")
print(f"\nCommitted N={N_BATCH} evaluation batch (group1={n1}, group0={n0}):")
print(f"  Batch demographic parity gap = {batch_dp_gap:.4f}")
print("\n--- Send back these 4 small files ---")
print("plaintext_metrics.json, batch_X.npy, batch_group.npy, batch_y.npy")
