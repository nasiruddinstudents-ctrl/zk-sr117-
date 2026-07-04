"""
15_ece_stratified_check.py

THE decisive diagnostic before writing up Section 7.5. Computes plaintext
ECE on the EXACT 32,768 stratified rows the attestation actually committed
to (reconstructed deterministically from chunks_idx_ece.npz), not the full
n=102,477 holdout. This isolates circuit-precision error from the separate
effect of stratified vs. uniform sampling -- the two were conflated in the
original bootstrap comparison.

Requires the files 13_ece_setup_once.py already wrote:
    X_hold_std_ece.npy, y_hold_ece.npy, chunks_idx_ece.npz
If you've deleted these, rerun 13_ece_setup_once.py first (it's
deterministic given the same nonce and data, so it will reconstruct the
identical stratified sample).

Run this LOCALLY, same folder.
"""
import json
import numpy as np

B = 10
ATTESTED_ECE = 0.33360250666737556  # from ece_driver_summary.json

X_hold_std = np.load("X_hold_std_ece.npy")
y_hold = np.load("y_hold_ece.npy")
chunks = np.load("chunks_idx_ece.npz")

with open("ece_meta.json") as f:
    meta = json.load(f)
K = meta["K"]

# Reconstruct the EXACT 32,768-row stratified sample the attestation used
all_idx = np.concatenate([chunks[f"chunk_{k}"] for k in range(K)])
print(f"Reconstructed stratified sample: {len(all_idx)} rows (should be 32,768)")

X_sample = X_hold_std[all_idx]
y_sample = y_hold[all_idx]

# Model A's confirmed coefficients (printed and cross-checked against the
# version-binding experiment earlier in this project -- using these directly
# rather than re-reading a metrics file avoids any ambiguity about which run
# produced it).
W = np.array([-0.09611620008945465, 0.16462214291095734, -0.6042760610580444, -0.15416495501995087])
b_intercept = 0.22204791009426117

logits = X_sample @ W + b_intercept
p = 1.0 / (1.0 + np.exp(-logits))
yhat = (p >= 0.5).astype(int)
correct = (yhat == y_sample).astype(int)

bins = np.clip(np.floor(p * B).astype(int), 0, B - 1)
n_b = np.bincount(bins, minlength=B)
c_b = np.bincount(bins, weights=correct, minlength=B)
s_b = np.bincount(bins, weights=p, minlength=B)

ece_plaintext_sample = np.sum(np.abs(c_b - s_b)) / len(X_sample)

print(f"\nPlaintext ECE on the EXACT 32,768-row stratified sample: {ece_plaintext_sample:.5f}")
print(f"Attested ECE (from the zkSNARK proofs):                 {ATTESTED_ECE:.5f}")
print(f"Delta (circuit precision only, sampling-design-free):    {ece_plaintext_sample - ATTESTED_ECE:.5f}")
print(f"\nFull-holdout plaintext ECE (n=102,477), for reference:   0.33590")
print(f"Delta of stratified-sample plaintext vs. full-holdout:   {ece_plaintext_sample - 0.33590:.5f}")

print("\n--- Interpretation ---")
delta_circuit = abs(ece_plaintext_sample - ATTESTED_ECE)
delta_sampling = abs(ece_plaintext_sample - 0.33590)
if delta_circuit < 0.0005:
    print("Case A (most likely): circuit is numerically exact on this exact sample.")
    print("The ~0.0023 gap vs. full-holdout is entirely a stratified-vs-uniform")
    print("sampling design effect, not a circuit precision issue.")
elif abs(ece_plaintext_sample - 0.33590) < 0.0005:
    print("Case B: plaintext-on-sample matches full-holdout closely, but the")
    print("ATTESTED value differs -- this points to real fixed-point drift in")
    print("the circuit itself (likely the sigmoid or bin-index computation),")
    print("not a sampling-design effect. Worth raising input_scale or reducing B.")
else:
    print("Case C: a mix of both effects. Report the decomposition explicitly:")
    print(f"  circuit-precision component: {delta_circuit:.5f}")
    print(f"  sampling-design component:   {delta_sampling:.5f}")

with open("ece_stratified_check_result.json", "w") as f:
    json.dump({
        "ece_plaintext_on_stratified_sample": float(ece_plaintext_sample),
        "attested_ece": ATTESTED_ECE,
        "full_holdout_ece": 0.33590,
        "delta_circuit_precision": float(ece_plaintext_sample - ATTESTED_ECE),
        "delta_sampling_design": float(ece_plaintext_sample - 0.33590),
    }, f, indent=2)

print("\n--- Send back: ece_stratified_check_result.json ---")
