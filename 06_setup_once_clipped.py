"""
06_setup_once_clipped.py

Same as 06_setup_once.py, but adds the committed preprocessing spec that
fixes the chunk-4 root cause (confirmed via ablation: row 113's
loan_to_value_ratio z-scored to ~2932, ~300x larger than any other value
in the dataset -- almost certainly a HMDA sentinel/placeholder code that
survived standardization).

THE SPEC (publish this in the paper's Table 1 as a new row -- "data
quality / preprocessing"):
  Each of the four features (income, loan_amount, debt_to_income_ratio,
  loan_to_value_ratio) is clipped to the [0.5th, 99.5th] percentile range
  of the TRAINING split, computed once and fixed as public constants,
  BEFORE standardization and BEFORE nonce-controlled sampling. This is a
  deterministic, publicly-specified transformation -- a supervisor
  approves the percentile thresholds as part of the validation-data
  schema, exactly as SR 11-7's "conceptual soundness" / data-quality
  language already requires. It does not weaken the nonce-controlled,
  cherry-picking-resistant sampling design: clipping is applied uniformly
  to the entire held-out pool before any row is selected, so a bank still
  cannot choose which specific rows land in which chunk.

This regenerates ALL setup artifacts from scratch (the clip changes mu/
sigma slightly, so old chunk files, keys, and results are stale and are
NOT reused -- run this instead of 06_setup_once.py, then delete old
result_chunk_*.json / proof_chunk_*.json / witness_chunk_*.json before
running 06_driver.py fresh).
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
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
CLIP_LOW_PCTILE = 0.5
CLIP_HIGH_PCTILE = 99.5


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
    graph = helper.make_graph(nodes, "sr117_chunk_counts", [X_in, group_in], [out],
                               initializer=[W_init, b_init, half, one, shape_n])
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

    # ---- COMMITTED PREPROCESSING SPEC: percentile clip, fit on TRAIN only ----
    clip_low = np.percentile(X_train, CLIP_LOW_PCTILE, axis=0)
    clip_high = np.percentile(X_train, CLIP_HIGH_PCTILE, axis=0)
    print("Committed preprocessing spec (percentile clip bounds, fit on training split):")
    for j, name in enumerate(FEATURES):
        print(f"  {name}: clip to [{clip_low[j]:.4f}, {clip_high[j]:.4f}] "
              f"({CLIP_LOW_PCTILE}th-{CLIP_HIGH_PCTILE}th percentile of training data)")
    X_train = np.clip(X_train, clip_low, clip_high)
    X_hold = np.clip(X_hold, clip_low, clip_high)

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_hold_std = (X_hold - mu) / sigma

    # Sanity check: confirm the clip actually bounds the standardized range now.
    max_abs_z = np.abs(X_hold_std).max()
    print(f"\nMax |z-score| in held-out set after clipping: {max_abs_z:.3f} "
          f"(was 2932.0 before the fix)")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_std, y_train)

    probs_hold = clf.predict_proba(X_hold_std)[:, 1]
    preds_hold = (probs_hold >= 0.5).astype(int)
    full_dp_gap = abs(preds_hold[g_hold == 1].mean() - preds_hold[g_hold == 0].mean())
    print(f"\nFull held-out (n={len(y_hold)}) true demographic parity gap (post-clip): {full_dp_gap:.4f}")

    W = clf.coef_[0].astype(np.float32)
    b = np.array([clf.intercept_[0]], dtype=np.float32)

    nonce_seed = int(hashlib.sha256(EXAMINER_NONCE.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(nonce_seed)
    true_prop_g1 = g_hold.mean()
    n1_total = min(int(round(N_TOTAL * true_prop_g1)), (g_hold == 1).sum())
    n0_total = N_TOTAL - n1_total
    idx_g1 = np.where(g_hold == 1)[0]
    idx_g0 = np.where(g_hold == 0)[0]
    sel_g1 = rng.choice(idx_g1, size=n1_total, replace=False)
    sel_g0 = rng.choice(idx_g0, size=n0_total, replace=False)
    all_idx = np.concatenate([sel_g1, sel_g0])
    rng.shuffle(all_idx)
    n_per_chunk = N_TOTAL // K
    chunks_idx = np.array_split(all_idx[:n_per_chunk * K], K)
    print(f"Stratified nonce-controlled selection: N={N_TOTAL}, K={K}, n_per_chunk={n_per_chunk}")

    np.save("X_hold_std.npy", X_hold_std.astype(np.float32))
    np.save("g_hold.npy", g_hold.astype(np.float32))
    np.savez("chunks_idx.npz", **{f"chunk_{i}": c for i, c in enumerate(chunks_idx)})

    onnx_path = "sr117_chunk.onnx"
    build_chunk_circuit(n_per_chunk, 4, W, b, onnx_path)
    print(f"Built chunk circuit: n={n_per_chunk}")

    sample_idx = chunks_idx[0]
    sample_X = X_hold_std[sample_idx].astype(np.float32)
    sample_g = g_hold[sample_idx].astype(np.float32)
    json.dump({"input_data": [sample_X.flatten().tolist(), sample_g.flatten().tolist()]},
               open("input_chunk.json", "w"))

    py_args = ezkl.PyRunArgs()
    py_args.input_scale = 16
    py_args.param_scale = 16
    ezkl.gen_settings(onnx_path, "settings_chunk.json", py_run_args=py_args)
    ezkl.calibrate_settings("input_chunk.json", onnx_path, "settings_chunk.json",
                             target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1])
    ezkl.compile_circuit(onnx_path, "network_chunk.ezkl", "settings_chunk.json")
    await ezkl.get_srs("settings_chunk.json")
    logrows = json.load(open("settings_chunk.json"))["run_args"]["logrows"]
    srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")
    ezkl.setup("network_chunk.ezkl", "vk_chunk.key", "pk_chunk.key", srs_path=srs_path)

    with open("srs_path.txt", "w") as f:
        f.write(srs_path)
    with open("setup_meta.json", "w") as f:
        json.dump({
            "N_total": N_TOTAL, "K": K, "n_per_chunk": n_per_chunk,
            "full_holdout_true_gap": float(full_dp_gap), "logrows": logrows,
            "srs_path": srs_path,
            "preprocessing_spec": {
                "clip_low_percentile": CLIP_LOW_PCTILE, "clip_high_percentile": CLIP_HIGH_PCTILE,
                "clip_low": clip_low.tolist(), "clip_high": clip_high.tolist(),
                "features": FEATURES,
            },
        }, f, indent=2)

    print("\nSetup complete (with committed preprocessing spec applied).")
    print("Now delete stale chunk results and run the driver fresh:")
    print("  rm -f result_chunk_*.json proof_chunk_*.json witness_chunk_*.json input_chunk_*.json")
    print("  python3 06_driver.py")

if __name__ == "__main__":
    asyncio.run(main())
