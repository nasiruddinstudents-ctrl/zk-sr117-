"""
12_version_binding.py

Substantiates Section 4's claim that attestations bind to a specific model
version and that a stale proof should not verify after the model changes.

Trains TWO model versions on real HMDA:
  Model A: the original 75/25 split (random_state=11) used throughout this
           paper's results.
  Model B: a reshuffled 90/10 split (random_state=99) -- different training
           data, so different fitted weights, simulating a bank retraining
           its model between supervisory windows.

For each model version, builds its own chunk-counts circuit (the weights
are baked into the circuit at compile time, so a different weight vector
produces a different circuit -> different proving/verifying key pair),
compiles+sets up independently, and produces a proof.

Then checks THREE things:
  (a) proof_A verifies against vk_A          -> should be True
  (b) proof_B verifies against vk_B          -> should be True
  (c) proof_A verifies against vk_B          -> should be False/error
Result (c) failing is exactly the "stale attestation does not verify
against a new model version" property Section 4 claims.

Run this LOCALLY, same folder as hmda_sample.csv. Uses a single small
chunk (n=1024) rather than the full 32-chunk sweep, since one chunk is
enough to demonstrate the binding property and this only needs to run
once per model version (~4s each).
"""
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

EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
N_CHUNK = 1024
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5


def build_chunk_circuit(n, D, W, b, path):
    X_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, D])
    group_in = helper.make_tensor_value_info("group", TensorProto.FLOAT, [n])
    out = helper.make_tensor_value_info("counts", TensorProto.FLOAT, [4])
    W_init = numpy_helper.from_array(W.reshape(D, 1), name="W")
    b_init = numpy_helper.from_array(b, name="b")
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
        helper.make_node("Sub", ["one", "group"], ["not_group"]),
        helper.make_node("Mul", ["preds", "group"], ["preds_g1"]),
        helper.make_node("Mul", ["preds", "not_group"], ["preds_g0"]),
        helper.make_node("ReduceSum", ["preds_g1"], ["sum_g1"], keepdims=1),
        helper.make_node("ReduceSum", ["group"], ["n1"], keepdims=1),
        helper.make_node("ReduceSum", ["preds_g0"], ["sum_g0"], keepdims=1),
        helper.make_node("ReduceSum", ["not_group"], ["n0"], keepdims=1),
        helper.make_node("Concat", ["sum_g1", "n1", "sum_g0", "n0"], ["counts"], axis=0),
    ]
    graph = helper.make_graph(nodes, "chunk", [X_in, group_in], [out],
                               initializer=[W_init, b_init, half, one, shape_n])
    model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def train_and_attest(version_label, test_size, random_state, df, FEATURES):
    X = df[FEATURES].values
    y = df["action_taken_binary"].values
    group = df["protected_group"].values

    X_train, X_hold, y_train, y_hold, g_train, g_hold = train_test_split(
        X, y, group, test_size=test_size, random_state=random_state, stratify=y
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
    print(f"[{version_label}] trained: n_train={len(y_train)}, n_hold={len(y_hold)}, "
          f"coef={W.tolist()}, intercept={float(b[0]):.4f}")

    nonce_seed = int(hashlib.sha256((EXAMINER_NONCE + version_label).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(nonce_seed)
    true_prop_g1 = g_hold.mean()
    n1 = min(int(round(N_CHUNK * true_prop_g1)), (g_hold == 1).sum())
    n0 = N_CHUNK - n1
    idx_g1 = rng.choice(np.where(g_hold == 1)[0], size=n1, replace=False)
    idx_g0 = rng.choice(np.where(g_hold == 0)[0], size=n0, replace=False)
    idx = np.concatenate([idx_g1, idx_g0])
    rng.shuffle(idx)
    Xn = X_hold_std[idx].astype(np.float32)
    gn = g_hold[idx].astype(np.float32)

    onnx_path = f"vb_circuit_{version_label}.onnx"
    build_chunk_circuit(N_CHUNK, 4, W, b, onnx_path)

    input_path = f"vb_input_{version_label}.json"
    json.dump({"input_data": [Xn.flatten().tolist(), gn.flatten().tolist()]}, open(input_path, "w"))

    settings_path = f"vb_settings_{version_label}.json"
    py_args = ezkl.PyRunArgs()
    py_args.input_scale = 16
    py_args.param_scale = 16
    ezkl.gen_settings(onnx_path, settings_path, py_run_args=py_args)
    ezkl.calibrate_settings(input_path, onnx_path, settings_path,
                             target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1])
    network_path = f"vb_network_{version_label}.ezkl"
    ezkl.compile_circuit(onnx_path, network_path, settings_path)

    with open(settings_path) as f:
        logrows = json.load(f)["run_args"]["logrows"]
    srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")
    if not os.path.exists(srs_path):
        import asyncio
        asyncio.run(ezkl.get_srs(settings_path))

    vk_path = f"vb_vk_{version_label}.key"
    pk_path = f"vb_pk_{version_label}.key"
    ezkl.setup(network_path, vk_path, pk_path, srs_path=srs_path)

    witness_path = f"vb_witness_{version_label}.json"
    ezkl.gen_witness(input_path, network_path, witness_path)
    proof_path = f"vb_proof_{version_label}.json"
    ezkl.prove(witness_path, network_path, pk_path, proof_path, srs_path=srs_path)

    self_verify = ezkl.verify(proof_path, settings_path, vk_path, srs_path=srs_path)
    print(f"[{version_label}] self-verify (proof_{version_label} against vk_{version_label}): {self_verify}")

    return {
        "version": version_label, "proof_path": proof_path, "settings_path": settings_path,
        "vk_path": vk_path, "srs_path": srs_path, "self_verify": bool(self_verify),
        "coef": W.tolist(), "intercept": float(b[0]),
    }


df = pd.read_csv("hmda_sample.csv")
FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]
df = df[df["derived_race"].isin(["White", "Black or African American",
                                   "Asian", "American Indian or Alaska Native",
                                   "Native Hawaiian or Other Pacific Islander"])].copy()
df["protected_group"] = (df["derived_race"] != "White").astype(int)

print("=== Training Model A (original 75/25 split, random_state=11) ===")
A = train_and_attest("A", test_size=0.25, random_state=11, df=df, FEATURES=FEATURES)

print("\n=== Training Model B (reshuffled 90/10 split, random_state=99) ===")
B = train_and_attest("B", test_size=0.10, random_state=99, df=df, FEATURES=FEATURES)

print("\n=== Cross-verification: does proof_A verify against vk_B? ===")
try:
    cross_result = ezkl.verify(A["proof_path"], B["settings_path"], B["vk_path"], srs_path=B["srs_path"])
    print(f"proof_A against vk_B: {cross_result}  (EXPECTED: False, or an exception)")
except Exception as e:
    cross_result = f"EXCEPTION: {e}"
    print(f"proof_A against vk_B raised an exception (also a valid 'rejected' outcome): {e}")

result = {
    "model_A": {"coef": A["coef"], "intercept": A["intercept"], "self_verify": A["self_verify"]},
    "model_B": {"coef": B["coef"], "intercept": B["intercept"], "self_verify": B["self_verify"]},
    "cross_verify_A_proof_against_B_key": str(cross_result),
    "version_binding_confirmed": (A["self_verify"] is True and B["self_verify"] is True
                                    and str(cross_result) not in ("True",)),
}
with open("version_binding_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(result, indent=2))
print("\n--- Send back: version_binding_result.json ---")
