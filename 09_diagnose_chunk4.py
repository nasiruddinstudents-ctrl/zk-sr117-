"""
09_diagnose_chunk4.py

Run this LOCALLY, in the same folder, BEFORE running the reseed retry
script (which deletes chunk 4's original input file -- we need it intact
for this diagnosis).

Checks 1 and 2 from the diagnostic plan: compares chunk 4's standardized
feature distributions against a healthy chunk (chunk 5), flags outliers,
NaN/Inf, and prints the most extreme rows so we can find the specific
offending value(s).

Note: features here are STANDARDIZED (z-scored against the training set),
not raw HMDA values -- so a "min/max" that's an extreme z-score (e.g. |z|
> 8-10) is the signal to look for, not a literal dollar amount. An extreme
z-score of, say, 400 would mean the raw value was ~400 standard deviations
from the training mean -- exactly the kind of outlier or sentinel-value
artifact the advisor's hypothesis predicts.
"""
import json
import numpy as np

FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]
FAILING_CHUNK = 4
COMPARISON_CHUNK = 5  # a chunk that succeeded cleanly


def load_chunk_input(k):
    path = f"input_chunk_{k}.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  !! {path} not found -- was it already deleted (e.g. by the reseed script)?")
        return None, None
    flat_X, flat_g = data["input_data"]
    n = len(flat_g)
    X = np.array(flat_X).reshape(n, len(FEATURES))
    g = np.array(flat_g)
    return X, g


print("=" * 70)
print("CHECK 1: Feature distribution comparison")
print("=" * 70)

for c in [FAILING_CHUNK, COMPARISON_CHUNK]:
    X, g = load_chunk_input(c)
    if X is None:
        continue
    label = "FAILING" if c == FAILING_CHUNK else "healthy (comparison)"
    print(f"\nChunk {c} ({label}), n={len(g)}:")
    for j, name in enumerate(FEATURES):
        col = X[:, j]
        n_nan = np.isnan(col).sum()
        n_inf = np.isinf(col).sum()
        finite = col[np.isfinite(col)]
        print(f"  {name:22s}: min={finite.min():10.3f}  max={finite.max():10.3f}  "
              f"mean={finite.mean():8.3f}  std={finite.std():8.3f}  "
              f"nan={n_nan}  inf={n_inf}  "
              f"|z|>8={int((np.abs(finite) > 8).sum())}  |z|>20={int((np.abs(finite) > 20).sum())}")

print("\n" + "=" * 70)
print("CHECK 2: Most extreme rows in the FAILING chunk")
print("=" * 70)

X4, g4 = load_chunk_input(FAILING_CHUNK)
if X4 is not None:
    for j, name in enumerate(FEATURES):
        col = X4[:, j]
        order = np.argsort(col)
        print(f"\n{name} -- bottom 3 rows (most negative z-score):")
        for idx in order[:3]:
            print(f"  row {idx}: {name}={col[idx]:.3f}, full_row={X4[idx].round(3).tolist()}, group={g4[idx]}")
        print(f"{name} -- top 3 rows (most positive z-score):")
        for idx in order[-3:]:
            print(f"  row {idx}: {name}={col[idx]:.3f}, full_row={X4[idx].round(3).tolist()}, group={g4[idx]}")

    print("\n" + "=" * 70)
    print("Any NaN/Inf anywhere in chunk 4?")
    print("=" * 70)
    nan_rows = np.where(np.isnan(X4).any(axis=1))[0]
    inf_rows = np.where(np.isinf(X4).any(axis=1))[0]
    print(f"Rows with NaN: {nan_rows.tolist()}")
    print(f"Rows with Inf: {inf_rows.tolist()}")
    if len(nan_rows) or len(inf_rows):
        print(">>> FOUND: NaN/Inf present -- this is almost certainly the cause.")

print("\n--- Review the output above. Look for any value wildly outside the ")
print("    comparison chunk's range, or any NaN/Inf. That row is the suspect. ---")
