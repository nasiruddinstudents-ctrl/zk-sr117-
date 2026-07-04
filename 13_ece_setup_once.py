"""
13_ece_setup_once.py

ECE as a second attested control (Table 1 Row 2), per the implementation
blueprint. Circuit computes, per bin b: n_b (count), c_b (count of
correctly predicted rows), s_b (sum of predicted probabilities) -- the
exact standard ECE decomposition, not an approximation:

    ECE = sum_b |c_b - s_b| / N

All three are additive across chunks, same pattern as demographic parity.

IMPORTANT: per the blueprint's "do not retrain" instruction, this reuses
the EXACT SAME training procedure as the demographic-parity result
(random_state=11, test_size=0.25, same clip spec) so the fitted weights
are the same Model A -- deterministic training on identical data with an
identical seed reproduces the same weights, satisfying "same model,
second control" without needing to persist weights to disk separately.

Run this FIRST, locally, same folder as hmda_sample.csv.
"""
import asyncio
import hashlib
import json
import os
import numpy as np
import pandas as pd
import onnx
from onnx import helper, TensorProto, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ezkl

N_TOTAL = 32768
K = 32
B = 10  # equal-width bins, standard ECE practice
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5


def build_ece_circuit(n, D, W, b_intercept, B, path):
    """Per-chunk circuit outputting 3*B values: [n_0,c_0,s_0, n_1,c_1,s_1, ...]"""
    X_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, D])
    y_in = helper.make_tensor_value_info("y", TensorProto.FLOAT, [n])
    out = helper.make_tensor_value_info("bin_stats", TensorProto.FLOAT, [3 * B])

    W_init = numpy_helper.from_array(W.reshape(D, 1), name="W")
    b_init = numpy_helper.from_array(b_intercept, name="b")
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    shape_n = numpy_helper.from_array(np.array([n], dtype=np.int64), name="shape_n")

    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["logits2d"]),
        helper.make_node("Reshape", ["logits2d", "shape_n"], ["logits2d_r"]),
        helper.make_node("Add", ["logits2d_r", "b"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["probs"]),
        helper.make_node("Greater", ["probs", "half"], ["preds_bool"]),
        helper.make_node("Cast", ["preds_bool"], ["preds"], to=TensorProto.FLOAT),
        # correct_i = 1 - |preds_i - y_i|  (both in {0,1}, so this is exactly
        # 1 when preds==y and 0 when preds!=y -- avoids needing an Equal/XOR op)
        helper.make_node("Sub", ["preds", "y"], ["diff_py"]),
        helper.make_node("Abs", ["diff_py"], ["absdiff_py"]),
        helper.make_node("Sub", ["one", "absdiff_py"], ["correct"]),
    ]
    initializers = [W_init, b_init, half, one, shape_n]

    bin_edges = np.linspace(0, 1, B + 1)
    concat_inputs = []
    for bi in range(B):
        lo, hi = float(bin_edges[bi]), float(bin_edges[bi + 1])
        lo_name, hi_name = f"lo_{bi}", f"hi_{bi}"
        initializers.append(numpy_helper.from_array(np.array([lo], dtype=np.float32), name=lo_name))
        initializers.append(numpy_helper.from_array(np.array([hi], dtype=np.float32), name=hi_name))

        nodes.append(helper.make_node("GreaterOrEqual", ["probs", lo_name], [f"ge_{bi}"]))
        nodes.append(helper.make_node("Less", ["probs", hi_name], [f"lt_{bi}"]))
        nodes.append(helper.make_node("And", [f"ge_{bi}", f"lt_{bi}"], [f"inbin_bool_{bi}"]))
        nodes.append(helper.make_node("Cast", [f"inbin_bool_{bi}"], [f"inbin_{bi}"], to=TensorProto.FLOAT))

        # n_b
        nodes.append(helper.make_node("ReduceSum", [f"inbin_{bi}"], [f"n_{bi}"], keepdims=1))
        # c_b = sum(inbin * correct)
        nodes.append(helper.make_node("Mul", [f"inbin_{bi}", "correct"], [f"inbin_correct_{bi}"]))
        nodes.append(helper.make_node("ReduceSum", [f"inbin_correct_{bi}"], [f"c_{bi}"], keepdims=1))
        # s_b = sum(inbin * probs)  -- the actual predicted-probability sum, per the blueprint
        nodes.append(helper.make_node("Mul", [f"inbin_{bi}", "probs"], [f"inbin_prob_{bi}"]))
        nodes.append(helper.make_node("ReduceSum", [f"inbin_prob_{bi}"], [f"s_{bi}"], keepdims=1))

        concat_inputs.extend([f"n_{bi}", f"c_{bi}", f"s_{bi}"])

    nodes.append(helper.make_node("Concat", concat_inputs, ["bin_stats"], axis=0))

    graph = helper.make_graph(nodes, "sr117_ece", [X_in, y_in], [out], initializer=initializers)
    model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


async def main():
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
    b_intercept = np.array([clf.intercept_[0]], dtype=np.float32)
    print(f"Trained (Model A, same procedure as demographic-parity result): "
          f"coef={W.tolist()}, intercept={float(b_intercept[0]):.4f}")

    probs_hold = clf.predict_proba(X_hold_std)[:, 1]
    preds_hold = (probs_hold >= 0.5).astype(float)
    correct_hold = (preds_hold == y_hold).astype(float)
    bin_edges = np.linspace(0, 1, B + 1)
    bin_idx = np.clip(np.digitize(probs_hold, bin_edges) - 1, 0, B - 1)
    ece_terms = []
    for bi in range(B):
        mask = bin_idx == bi
        if mask.sum() == 0:
            continue
        n_b = mask.sum()
        c_b = correct_hold[mask].sum()
        s_b = probs_hold[mask].sum()
        ece_terms.append(abs(c_b - s_b))
    full_ece = float(sum(ece_terms) / len(y_hold))
    print(f"Full held-out (n={len(y_hold)}) true ECE ({B} bins): {full_ece:.4f}")

    nonce_seed = int(hashlib.sha256((EXAMINER_NONCE + "-ECE").encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(nonce_seed)
    true_prop_g1 = g_hold.mean()
    n1_total = min(int(round(N_TOTAL * true_prop_g1)), (g_hold == 1).sum())
    n0_total = N_TOTAL - n1_total
    idx_g1 = rng.choice(np.where(g_hold == 1)[0], size=n1_total, replace=False)
    idx_g0 = rng.choice(np.where(g_hold == 0)[0], size=n0_total, replace=False)
    all_idx = np.concatenate([idx_g1, idx_g0])
    rng.shuffle(all_idx)
    n_per_chunk = N_TOTAL // K
    chunks_idx = np.array_split(all_idx[:n_per_chunk * K], K)
    print(f"Stratified nonce-controlled selection: N={N_TOTAL}, K={K}, n_per_chunk={n_per_chunk}")

    np.save("X_hold_std_ece.npy", X_hold_std.astype(np.float32))
    np.save("y_hold_ece.npy", y_hold.astype(np.float32))
    np.savez("chunks_idx_ece.npz", **{f"chunk_{i}": c for i, c in enumerate(chunks_idx)})

    onnx_path = "sr117_ece.onnx"
    build_ece_circuit(n_per_chunk, 4, W, b_intercept, B, onnx_path)
    print(f"Built ECE circuit: n={n_per_chunk}, B={B}, output_dim={3*B}")

    sample_idx = chunks_idx[0]
    sample_X = X_hold_std[sample_idx].astype(np.float32)
    sample_y = y_hold[sample_idx].astype(np.float32)
    json.dump({"input_data": [sample_X.flatten().tolist(), sample_y.flatten().tolist()]},
               open("input_ece.json", "w"))

    py_args = ezkl.PyRunArgs()
    py_args.input_scale = 16
    py_args.param_scale = 16
    ezkl.gen_settings(onnx_path, "settings_ece.json", py_run_args=py_args)
    ezkl.calibrate_settings("input_ece.json", onnx_path, "settings_ece.json",
                             target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1])
    ezkl.compile_circuit(onnx_path, "network_ece.ezkl", "settings_ece.json")
    await ezkl.get_srs("settings_ece.json")
    logrows = json.load(open("settings_ece.json"))["run_args"]["logrows"]
    srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")
    ezkl.setup("network_ece.ezkl", "vk_ece.key", "pk_ece.key", srs_path=srs_path)

    with open("ece_meta.json", "w") as f:
        json.dump({
            "N_total": N_TOTAL, "K": K, "B": B, "n_per_chunk": n_per_chunk,
            "full_holdout_true_ece": full_ece, "logrows": logrows, "srs_path": srs_path,
        }, f, indent=2)

    print("\nSetup complete. Now run: python3 13_ece_driver.py")

if __name__ == "__main__":
    asyncio.run(main())
