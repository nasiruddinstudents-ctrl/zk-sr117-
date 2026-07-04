"""
11_flatsum_real_sweep.py

Real-data confirmation of Table 2's flat-sum breakdown (currently measured
only on synthetic batches of matched shape). Builds the SAME flat
single-ReduceSum circuit used in the original N=1024 result, on REAL
preprocessed HMDA data (post-clip, per Table 1 Row 9), at N in
{2048, 4096, 8192}, and reports the numerical fidelity error at each --
directly comparable to the synthetic characterization in Table 2.

This checks calibration/settings-generation fidelity only (fast, no
prove/verify) -- that's the number Table 2 actually reports (relative
error), and it avoids risking a repeat of the earlier stuck-process issue
from attempting a full prove at a known-broken N.

Run this LOCALLY, same folder as hmda_sample.csv.
"""
import hashlib
import json
import numpy as np
import pandas as pd
import onnx
from onnx import helper, TensorProto, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ezkl

EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
N_VALUES = [2048, 4096, 8192]
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5


def build_flatsum_circuit(N, D, W, b, path):
    """The ORIGINAL flat single-ReduceSum design (not chunked, not tree) --
    same circuit shape as the very first N=1024 result in Section 7.1."""
    X_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, D])
    group_in = helper.make_tensor_value_info("group", TensorProto.FLOAT, [N])
    gap_out = helper.make_tensor_value_info("gap", TensorProto.FLOAT, [1])
    W_init = numpy_helper.from_array(W.reshape(D, 1), name="W")
    b_init = numpy_helper.from_array(b, name="b")
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    shape_N = numpy_helper.from_array(np.array([N], dtype=np.int64), name="shape_N")
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["logits2d"]),
        helper.make_node("Reshape", ["logits2d", "shape_N"], ["logits2d_r"]),
        helper.make_node("Add", ["logits2d_r", "b"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["probs"]),
        helper.make_node("Greater", ["probs", "half"], ["preds_bool"]),
        helper.make_node("Cast", ["preds_bool"], ["preds"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "group"], ["not_group"]),
        helper.make_node("Mul", ["preds", "group"], ["preds_g1"]),
        helper.make_node("Mul", ["preds", "not_group"], ["preds_g0"]),
        helper.make_node("ReduceSum", ["preds_g1"], ["sum_g1"], keepdims=1),
        helper.make_node("ReduceSum", ["group"], ["n1"], keepdims=1),
        helper.make_node("ReduceSum", ["preds_g0"], ["sum_g0"], keepdims=1),
        helper.make_node("ReduceSum", ["not_group"], ["n0"], keepdims=1),
        helper.make_node("Div", ["sum_g1", "n1"], ["approve1"]),
        helper.make_node("Div", ["sum_g0", "n0"], ["approve0"]),
        helper.make_node("Sub", ["approve1", "approve0"], ["diff"]),
        helper.make_node("Abs", ["diff"], ["gap"]),
    ]
    graph = helper.make_graph(nodes, "flatsum", [X_in, group_in], [gap_out],
                               initializer=[W_init, b_init, half, one, shape_N])
    model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


# ---- Train on real HMDA, same preprocessing spec as the main result ----
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
W = clf.coef_[0].astype(np.float32)
b = np.array([clf.intercept_[0]], dtype=np.float32)

nonce_seed = int(hashlib.sha256(EXAMINER_NONCE.encode()).hexdigest(), 16) % (2**32)
rng = np.random.default_rng(nonce_seed)
true_prop_g1 = g_hold.mean()

results = []
for N in N_VALUES:
    print(f"\n=== N={N} (real HMDA, post-clip) ===")
    n1 = min(int(round(N * true_prop_g1)), (g_hold == 1).sum())
    n0 = N - n1
    idx_g1 = rng.choice(np.where(g_hold == 1)[0], size=n1, replace=False)
    idx_g0 = rng.choice(np.where(g_hold == 0)[0], size=n0, replace=False)
    idx = np.concatenate([idx_g1, idx_g0])
    rng.shuffle(idx)

    Xn = X_hold_std[idx].astype(np.float32)
    gn = g_hold[idx].astype(np.float32)

    onnx_path = f"flatsum_N{N}.onnx"
    build_flatsum_circuit(N, 4, W, b, onnx_path)

    input_path = f"input_flatsum_N{N}.json"
    json.dump({"input_data": [Xn.flatten().tolist(), gn.flatten().tolist()]}, open(input_path, "w"))

    settings_path = f"settings_flatsum_N{N}.json"
    py_args = ezkl.PyRunArgs()
    py_args.input_scale = 16
    py_args.param_scale = 16
    ezkl.gen_settings(onnx_path, settings_path, py_run_args=py_args)
    # target=accuracy is the corrected calibration mode (see Section 7.2's
    # discussion of the target=resources bug found earlier in this project).
    ezkl.calibrate_settings(input_path, onnx_path, settings_path,
                             target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1])

    with open(settings_path) as f:
        s = json.load(f)
    logrows = s["run_args"]["logrows"]
    final_scale = s["run_args"]["input_scale"]

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    true_gap = float(sess.run(["gap"], {"X": Xn, "group": gn})[0][0])

    print(f"  logrows={logrows}, final_scale={final_scale}, true_gap(plaintext)={true_gap:.4f}")
    results.append({"N": N, "logrows": logrows, "final_scale": final_scale, "true_gap_plaintext": true_gap})

with open("flatsum_real_sweep_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(results, indent=2))
print("\nNote: this script reports circuit calibration behavior (logrows, scale) at each N.")
print("Compare against Table 2's synthetic-data thresholds. If you also want the actual")
print("EZKL Numerical Fidelity Report printed during calibrate_settings (shown above each")
print("N's output), that IS the real-data equivalent of Table 2's error percentages --")
print("copy those tables from the terminal output for the paper.")
print("\n--- Send back: flatsum_real_sweep_result.json AND the full terminal output")
print("    (including the Numerical Fidelity Report tables EZKL prints for each N) ---")
