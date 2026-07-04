"""
14_ece_bootstrap_ci.py

Bootstrap 95% CI for the ECE estimator at N=32,768 (and N=8,192 for
comparison), matching the demographic-parity bootstrap methodology exactly:
2,000 resamples of size N, with replacement, from the full n=102,477
preprocessed held-out set, computing ECE on each resample.

Run this LOCALLY, same folder as hmda_sample.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

N_BOOTSTRAP = 2000
N_VALUES = [8192, 32768]
RANDOM_SEED = 11
B = 10  # bin count, matching the ECE circuit
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5

df = pd.read_csv("hmda_sample.csv")
FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]
df = df[df["derived_race"].isin(["White", "Black or African American",
                                   "Asian", "American Indian or Alaska Native",
                                   "Native Hawaiian or Other Pacific Islander"])].copy()
df["protected_group"] = (df["derived_race"] != "White").astype(int)

X = df[FEATURES].values
y = df["action_taken_binary"].values
group = df["protected_group"].values

X_train, X_hold, y_train, y_hold, g_train, g_hold = train_test_split(
    X, y, group, test_size=0.25, random_state=RANDOM_SEED, stratify=y
)
clip_low = np.percentile(X_train, CLIP_LOW_PCTILE, axis=0)
clip_high = np.percentile(X_train, CLIP_HIGH_PCTILE, axis=0)
X_train = np.clip(X_train, clip_low, clip_high)
X_hold = np.clip(X_hold, clip_low, clip_high)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_std = (X_train - mu) / sigma
X_hold_std = (X_hold - mu) / sigma

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_std, y_train)
print(f"Model A (should match: coef=[-0.0961, 0.1646, -0.6043, -0.1542], intercept=0.2220): "
      f"coef={clf.coef_[0].tolist()}, intercept={clf.intercept_[0]:.4f}")

probs_hold = clf.predict_proba(X_hold_std)[:, 1]
preds_hold = (probs_hold >= 0.5).astype(float)
correct_hold = (preds_hold == y_hold).astype(float)

bin_edges = np.linspace(0, 1, B + 1)


def compute_ece(probs, correct, n):
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, B - 1)
    total = 0.0
    for bi in range(B):
        mask = bin_idx == bi
        if mask.sum() == 0:
            continue
        c_b = correct[mask].sum()
        s_b = probs[mask].sum()
        total += abs(c_b - s_b)
    return total / n


full_ece = compute_ece(probs_hold, correct_hold, len(y_hold))
print(f"Full held-out (n={len(y_hold)}) true ECE: {full_ece:.4f}")

rng = np.random.default_rng(RANDOM_SEED)
n_hold = len(y_hold)

results = {}
for N in N_VALUES:
    eces = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_hold, size=min(N, n_hold), replace=(N > n_hold))
        e = compute_ece(probs_hold[idx], correct_hold[idx], len(idx))
        eces.append(e)
    eces = np.array(eces)
    ci_lower, ci_upper = np.percentile(eces, [2.5, 97.5])
    results[N] = {
        "mean_ece": float(eces.mean()), "std_ece": float(eces.std()),
        "ci_95_lower": float(ci_lower), "ci_95_upper": float(ci_upper),
    }
    print(f"\nN={N}: bootstrap mean ECE = {eces.mean():.4f}, std = {eces.std():.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  (compare your ece_driver_summary.json's aggregated_attested_ece against this interval)")

import json
with open("ece_bootstrap_ci_result.json", "w") as f:
    json.dump({"full_holdout_ece": float(full_ece), "by_N": results}, f, indent=2)

print("\n--- Send back: ece_bootstrap_ci_result.json ---")
