"""
02_train_real_v2.py

STEP 1 FIX. Run this LOCALLY in the same folder as hmda_sample.csv.

Changes from the first version:
  - N=64 -> N=1024
  - Uniform random batch -> stratified by protected class, where the
    STRATUM SIZES are set deterministically by an examiner-issued nonce
    (not by the bank), matching the true held-out subgroup prevalence.
    Row SELECTION within each stratum is still nonce-seeded random --
    this is what "nonce-controlled, not cherry-picked" means in practice:
    a bank cannot game which N rows land in the batch, and cannot game
    how many of each subgroup appear either.
  - Stratum sizes (n1, n0) and the nonce itself are written into
    plaintext_metrics.json as committed public metadata -- an examiner
    verifying the proof can also check that the batch composition matches
    what the nonce dictates, independent of the ZK proof itself.

Output files (all small, safe to upload):
    plaintext_metrics.json
    batch_X.npy   (1024 x 4)
    batch_group.npy (1024,)
    batch_y.npy   (1024,)
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# ---- Examiner-issued nonce (stand-in) ----
# In production this is issued by the OCC/Fed examiner for a specific
# supervisory window; here we use a stated, published string so the
# derivation is reproducible and auditable.
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
nonce_seed = int(hashlib.sha256(EXAMINER_NONCE.encode()).hexdigest(), 16) % (2**32)

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
full_dp_gap = abs(approve_rate_g1 - approve_rate_g0)

# ---- Stratified, nonce-controlled batch, N=1024 ----
N_BATCH = 65536
true_prop_g1 = g_hold.mean()  # true nonwhite prevalence in held-out set
n1 = int(round(N_BATCH * true_prop_g1))
n0 = N_BATCH - n1

rng = np.random.default_rng(nonce_seed)
idx_g1 = np.where(g_hold == 1)[0]
idx_g0 = np.where(g_hold == 0)[0]
sel_g1 = rng.choice(idx_g1, size=min(n1, len(idx_g1)), replace=False)
sel_g0 = rng.choice(idx_g0, size=min(n0, len(idx_g0)), replace=False)
idx = np.concatenate([sel_g1, sel_g0])
rng.shuffle(idx)

batch_X = X_hold_std[idx].astype(np.float32)
batch_group = g_hold[idx].astype(np.float32)
batch_y = y_hold[idx].astype(np.float32)

np.save("batch_X.npy", batch_X)
np.save("batch_group.npy", batch_group)
np.save("batch_y.npy", batch_y)

batch_probs = clf.predict_proba(batch_X)[:, 1]
batch_preds = (batch_probs >= 0.5).astype(int)
batch_dp_gap = abs(
    batch_preds[batch_group == 1].mean() - batch_preds[batch_group == 0].mean()
)

metrics = {
    "n_holdout": int(len(y_hold)),
    "auc": float(auc),
    "ks": ks,
    "approve_rate_nonwhite": float(approve_rate_g1),
    "approve_rate_white": float(approve_rate_g0),
    "full_holdout_demographic_parity_gap": float(full_dp_gap),
    "batch_demographic_parity_gap": float(batch_dp_gap),
    "batch_size": N_BATCH,
    "batch_n_nonwhite": int(len(sel_g1)),
    "batch_n_white": int(len(sel_g0)),
    "true_nonwhite_prevalence_in_holdout": float(true_prop_g1),
    "examiner_nonce": EXAMINER_NONCE,
    "examiner_nonce_seed_derived": nonce_seed,
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
    "feature_mean": mu.tolist(),
    "feature_std": sigma.tolist(),
    "features": FEATURES,
    "data_source": "real HMDA 2022, filtered conventional/first-lien/purchase/principal-residence",
}
with open("plaintext_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Full held-out slice (n={len(y_hold)}): AUC={auc:.4f} KS={ks:.4f} full_gap={full_dp_gap:.4f}")
print(f"Nonce-controlled batch: N={N_BATCH}, n_nonwhite={len(sel_g1)}, n_white={len(sel_g0)}")
print(f"  (true nonwhite prevalence in holdout: {true_prop_g1:.4f})")
print(f"  Batch demographic parity gap = {batch_dp_gap:.4f}  (compare to full gap {full_dp_gap:.4f})")
print("\n--- Send back: plaintext_metrics.json, batch_X.npy, batch_group.npy, batch_y.npy ---")
