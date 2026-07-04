"""
07_bootstrap_ci.py

Run this LOCALLY, same folder as hmda_sample.csv.

Computes the bootstrap 95% CI of the demographic-parity-gap ESTIMATOR at
a given subsample size N, by repeatedly drawing random N-sized subsamples
from the full held-out set and recording the gap each time. This tells us
whether the attested value from 06_chunked_real.py is within normal
sampling variance of the true full-holdout gap, or actually anomalous.

Reports CIs at N=8192 and N=32768 (edit N_VALUES to add more / match
whatever N you actually ran the chunked attestation at).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

N_BOOTSTRAP = 2000       # number of bootstrap resamples per N
N_VALUES = [8192, 32768]  # match whatever N_TOTAL values you ran in 06_chunked_real.py
RANDOM_SEED = 11

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

# Committed preprocessing spec, matching 06_setup_once_clipped.py exactly:
# clip to [0.5th, 99.5th] percentile of the TRAINING split before standardizing.
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5
clip_low = np.percentile(X_train, CLIP_LOW_PCTILE, axis=0)
clip_high = np.percentile(X_train, CLIP_HIGH_PCTILE, axis=0)
print("Committed preprocessing spec (should match 06_setup_once_clipped.py's printed bounds):")
for j, name in enumerate(FEATURES):
    print(f"  {name}: clip to [{clip_low[j]:.4f}, {clip_high[j]:.4f}]")
X_train = np.clip(X_train, clip_low, clip_high)
X_hold = np.clip(X_hold, clip_low, clip_high)

mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_std = (X_train - mu) / sigma
X_hold_std = (X_hold - mu) / sigma

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_std, y_train)

probs_hold = clf.predict_proba(X_hold_std)[:, 1]
preds_hold = (probs_hold >= 0.5).astype(int)
full_gap = abs(preds_hold[g_hold == 1].mean() - preds_hold[g_hold == 0].mean())
print(f"Full held-out (n={len(y_hold)}) true demographic parity gap: {full_gap:.4f}")

rng = np.random.default_rng(RANDOM_SEED)
n_hold = len(y_hold)

results = {}
for N in N_VALUES:
    if N > n_hold:
        print(f"\nN={N} exceeds full holdout size ({n_hold}); sampling with replacement.")
    gaps = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_hold, size=min(N, n_hold), replace=(N > n_hold))
        sub_preds = preds_hold[idx]
        sub_group = g_hold[idx]
        n1, n0 = (sub_group == 1).sum(), (sub_group == 0).sum()
        if n1 == 0 or n0 == 0:
            continue
        gap = abs(sub_preds[sub_group == 1].mean() - sub_preds[sub_group == 0].mean())
        gaps.append(gap)
    gaps = np.array(gaps)
    ci_lower, ci_upper = np.percentile(gaps, [2.5, 97.5])
    results[N] = {
        "n_bootstrap_draws": len(gaps),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }
    print(f"\nN={N}: bootstrap mean gap = {gaps.mean():.4f}, std = {gaps.std():.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  (compare your 06_chunked_real.py aggregated_attested_gap against this interval)")

import json
with open("bootstrap_ci_result.json", "w") as f:
    json.dump({"full_holdout_gap": float(full_gap), "by_N": results}, f, indent=2)

print("\n--- Send back: bootstrap_ci_result.json ---")
