"""
16_ece_version_binding.py

A4: repeats the Section 7.3 Model-A/Model-B cross-verification test, now
for the ECE circuit. Trains Model B (90/10 reshuffled split, random_state=99,
same as the original version-binding test), builds its own ECE circuit,
and confirms (a) Model B's ECE proof verifies against its own key, and
(b) Model A's ECE proof (from 13_ece_setup_once.py / 13_ece_driver.py's
chunk 0, reused here) does NOT verify against Model B's key.

Run this LOCALLY, same folder, AFTER 13_ece_setup_once.py and
13_ece_driver.py have already produced proof_ece_0.json (Model A's chunk-0
ECE proof) -- this script reuses that file rather than reproving.
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

CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5
B = 10
N_CHUNK = 1024
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001-ECE"


def build_ece_circuit(n, D, W, b_intercept, B, path):
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
        nodes.append(helper.make_node("ReduceSum", [f"inbin_{bi}"], [f"n_{bi}"], keepdims=1))
        nodes.append(helper.make_node("Mul", [f"inbin_{bi}", "correct"], [f"inbin_correct_{bi}"]))
        nodes.append(helper.make_node("ReduceSum", [f"inbin_correct_{bi}"], [f"c_{bi}"], keepdims=1))
        nodes.append(helper.make_node("Mul", [f"inbin_{bi}", "probs"], [f"inbin_prob_{bi}"]))
        nodes.append(helper.make_node("ReduceSum", [f"inbin_prob_{bi}"], [f"s_{bi}"], keepdims=1))
        concat_inputs.extend([f"n_{bi}", f"c_{bi}", f"s_{bi}"])
    nodes.append(helper.make_node("Concat", concat_inputs, ["bin_stats"], axis=0))
    graph = helper.make_graph(nodes, "sr117_ece_b", [X_in, y_in], [out], initializer=initializers)
    model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


async def main():
    if not os.path.exists("proof_ece_0.json"):
        print("ERROR: proof_ece_0.json not found. Run 13_ece_setup_once.py and")
        print("13_ece_driver.py first (Model A's ECE chunk proofs must already exist).")
        return

    df = pd.read_csv("hmda_sample.csv")
    FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]
    df = df[df["derived_race"].isin(["White", "Black or African American",
                                       "Asian", "American Indian or Alaska Native",
                                       "Native Hawaiian or Other Pacific Islander"])].copy()
    df["protected_group"] = (df["derived_race"] != "White").astype(int)
    X = df[FEATURES].values
    y = df["action_taken_binary"].values
    group = df["protected_group"].values

    # Model B: same 90/10 reshuffled split as the original version-binding test
    X_train, X_hold, y_train, y_hold, g_train, g_hold = train_test_split(
        X, y, group, test_size=0.10, random_state=99, stratify=y
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
    print(f"Model B trained: coef={W.tolist()}, intercept={float(b_intercept[0]):.4f}")

    nonce_seed = int(hashlib.sha256(EXAMINER_NONCE.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(nonce_seed)
    n1 = min(int(round(N_CHUNK * g_hold.mean())), (g_hold == 1).sum())
    n0 = N_CHUNK - n1
    idx_g1 = rng.choice(np.where(g_hold == 1)[0], size=n1, replace=False)
    idx_g0 = rng.choice(np.where(g_hold == 0)[0], size=n0, replace=False)
    idx = np.concatenate([idx_g1, idx_g0])
    rng.shuffle(idx)
    Xn = X_hold_std[idx].astype(np.float32)
    yn = y_hold[idx].astype(np.float32)

    onnx_path = "ece_circuit_B.onnx"
    build_ece_circuit(N_CHUNK, 4, W, b_intercept, B, onnx_path)

    input_path = "ece_input_B.json"
    json.dump({"input_data": [Xn.flatten().tolist(), yn.flatten().tolist()]}, open(input_path, "w"))

    settings_path = "ece_settings_B.json"
    py_args = ezkl.PyRunArgs()
    py_args.input_scale = 16
    py_args.param_scale = 16
    ezkl.gen_settings(onnx_path, settings_path, py_run_args=py_args)
    ezkl.calibrate_settings(input_path, onnx_path, settings_path,
                             target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1])
    network_path = "ece_network_B.ezkl"
    ezkl.compile_circuit(onnx_path, network_path, settings_path)

    logrows = json.load(open(settings_path))["run_args"]["logrows"]
    srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")
    if not os.path.exists(srs_path):
        await ezkl.get_srs(settings_path)

    vk_path, pk_path = "ece_vk_B.key", "ece_pk_B.key"
    ezkl.setup(network_path, vk_path, pk_path, srs_path=srs_path)

    witness_path = "ece_witness_B.json"
    ezkl.gen_witness(input_path, network_path, witness_path)
    proof_path_B = "ece_proof_B.json"
    ezkl.prove(witness_path, network_path, pk_path, proof_path_B, srs_path=srs_path)

    self_verify_B = ezkl.verify(proof_path_B, settings_path, vk_path, srs_path=srs_path)
    print(f"Model B ECE proof self-verify (against its own key): {self_verify_B}")

    print("\nCross-verification: does Model A's ECE proof (proof_ece_0.json) verify against Model B's key?")
    try:
        cross = ezkl.verify("proof_ece_0.json", settings_path, vk_path, srs_path=srs_path)
        print(f"Result: {cross} (EXPECTED: False, or an exception)")
    except Exception as e:
        cross = f"EXCEPTION: {e}"
        print(f"Model A's ECE proof was rejected with an exception (a valid 'rejected' outcome): {e}")

    result = {
        "model_B_self_verify": bool(self_verify_B),
        "cross_verify_modelA_ece_proof_against_modelB_key": str(cross),
        "version_binding_confirmed_for_ece": (
            bool(self_verify_B) and str(cross) not in ("True",)
        ),
    }
    with open("ece_version_binding_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print("\n--- Send back: ece_version_binding_result.json ---")

if __name__ == "__main__":
    asyncio.run(main())
