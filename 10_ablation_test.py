"""
10_ablation_test.py

Check 3: remove the suspected outlier (row 113, loan_to_value_ratio ~= 2932)
from chunk 4 and retry. If it now runs in ~3.8s like every other chunk,
we've confirmed the cause. This is a DIAGNOSTIC step only -- it does not
fix the real chunk 4 (which still needs its full n=1,024 rows under the
committed preprocessing spec, see 11_apply_preprocessing_spec.py) -- it
just confirms the hypothesis before we commit to that fix.

Run this LOCALLY, same folder, before running the reseed script.
"""
import json
import time
import numpy as np

OUTLIER_ROW_LOCAL_INDEX = 113  # the row index WITHIN chunk 4, as printed by 09_diagnose_chunk4.py

with open("input_chunk_4.json") as f:
    data = json.load(f)

flat_X, flat_g = data["input_data"]
n = len(flat_g)
X = np.array(flat_X).reshape(n, 4)
g = np.array(flat_g)

print(f"Original chunk 4: n={n}")
print(f"Row {OUTLIER_ROW_LOCAL_INDEX} (the suspect): {X[OUTLIER_ROW_LOCAL_INDEX].tolist()}, group={g[OUTLIER_ROW_LOCAL_INDEX]}")

# Remove the suspect row, duplicate a random OTHER row to keep n=1024
# (the circuit is compiled for a fixed n; for this diagnostic ablation we
# just need *a* valid n=1024 input without the outlier -- duplicating a
# benign row is fine for testing the hypothesis, this is not the real fix).
mask = np.ones(n, dtype=bool)
mask[OUTLIER_ROW_LOCAL_INDEX] = False
X_ablated = X[mask]
g_ablated = g[mask]
# pad back to n=1024 by duplicating row 0
X_ablated = np.concatenate([X_ablated, X_ablated[0:1]], axis=0)
g_ablated = np.concatenate([g_ablated, g_ablated[0:1]], axis=0)

json.dump({"input_data": [X_ablated.flatten().tolist(), g_ablated.flatten().tolist()]},
           open("input_chunk_4_ablated.json", "w"))

# Run witness generation directly to test just that step, which is where
# the hang was observed.
import ezkl

with open("setup_meta.json") as f:
    meta = json.load(f)
srs_path = meta["srs_path"]

print("\nTesting witness generation on ABLATED chunk 4 (outlier row removed)...")
t0 = time.time()
try:
    ezkl.gen_witness("input_chunk_4_ablated.json", "network_chunk.ezkl", "witness_chunk_4_ablated.json")
    elapsed = time.time() - t0
    print(f"SUCCESS: witness generated in {elapsed:.2f}s (compare to hang on original chunk 4)")
    print("\n>>> CONFIRMED: the outlier row (LTV~=2932) is the cause. <<<")
    print("Next step: apply a committed preprocessing spec (percentile clip) and")
    print("regenerate all 32 chunks properly -- do NOT ship this ablated/duplicated")
    print("version, it was only to confirm the diagnosis.")
except Exception as e:
    elapsed = time.time() - t0
    print(f"Still failed/hung after {elapsed:.2f}s even without the outlier: {e}")
    print("This means the outlier row alone doesn't explain it -- worth checking")
    print("for a second, less extreme problem row, or a different cause entirely.")
