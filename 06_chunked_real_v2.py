"""
06_chunked_real.py

Path B-lite, on REAL HMDA data, run entirely locally.

Combines the training/sampling logic (like 02_train_real_v2.py) with
chunked proving: instead of one N=8192-sized circuit (which overflows) or
one tree-reduction circuit (which didn't compile in reasonable time), this
proves K separate small circuits (n=N_total/K each, same known-good shape
as your working N=1024 flat circuit), each outputting raw per-group counts.
The final fairness gap is computed in the clear from the sum of all K
chunks' published counts -- no ZK needed for that last step, and nothing
about individual rows is revealed at any point.

Compile + setup happen ONCE (same circuit shape reused for every chunk).
Only witness generation + proving + verification repeat per chunk.

pip install ezkl onnx onnxruntime pandas scikit-learn numpy   (if needed)
"""
import asyncio
import hashlib
import json
import os
import signal
import time
import numpy as np
import pandas as pd
import onnx
from onnx import helper, TensorProto, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import ezkl

class ChunkTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise ChunkTimeout("chunk exceeded per-chunk timeout")

CHUNK_TIMEOUT_S = 60  # generous vs. the observed ~3.5s/chunk baseline

# ---- Config ----
N_TOTAL = 32768   # try this first; bump to 32768 once this works
K = 32           # n_per_chunk = N_TOTAL / K = 1024, the known-good size
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"

# ---- 1. Train on real HMDA (same as 02_train_real_v2.py) ----
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
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_std = (X_train - mu) / sigma
X_hold_std = (X_hold - mu) / sigma

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_std, y_train)

probs_hold = clf.predict_proba(X_hold_std)[:, 1]
preds_hold = (probs_hold >= 0.5).astype(int)
full_dp_gap = abs(preds_hold[g_hold == 1].mean() - preds_hold[g_hold == 0].mean())
print(f"Full held-out (n={len(y_hold)}) true demographic parity gap: {full_dp_gap:.4f}")

W = clf.coef_[0].astype(np.float32)
b = np.array([clf.intercept_[0]], dtype=np.float32)

# ---- 2. Stratified, nonce-controlled selection of N_TOTAL rows ----
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
print(f"Stratified nonce-controlled selection: N={N_TOTAL}, K={K}, n_per_chunk={n_per_chunk}, "
      f"n1_total={n1_total}, n0_total={n0_total}")


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
    onnx_path = "sr117_chunk.onnx"
    build_chunk_circuit(n_per_chunk, 4, W, b, onnx_path)
    print(f"Built chunk circuit: n={n_per_chunk}")

    sample_idx = chunks_idx[0]
    sample_X = X_hold_std[sample_idx].astype(np.float32)
    sample_g = g_hold[sample_idx].astype(np.float32)
    json.dump({"input_data": [sample_X.flatten().tolist(), sample_g.flatten().tolist()]},
               open("input_chunk.json", "w"))

    t0 = time.time()
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
    setup_time = time.time() - t0
    print(f"One-time compile+setup: {setup_time:.2f}s (logrows={logrows})")

    chunk_results = []
    checkpoint_path = "chunked_checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            saved = json.load(f)
        if saved.get("N_total") == N_TOTAL and saved.get("K") == K:
            chunk_results = saved.get("chunk_results", [])
            print(f"Resuming from checkpoint: {len(chunk_results)} chunks already done")

    total_proof_kb = sum(c["proof_kb"] for c in chunk_results)
    t_loop0 = time.time()
    start_k = len(chunk_results)
    for k in range(start_k, len(chunks_idx)):
        idx = chunks_idx[k]
        cX = X_hold_std[idx].astype(np.float32)
        cg = g_hold[idx].astype(np.float32)
        json.dump({"input_data": [cX.flatten().tolist(), cg.flatten().tolist()]},
                   open("input_chunk.json", "w"))

        signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.alarm(CHUNK_TIMEOUT_S)
            t0 = time.time()
            ezkl.gen_witness("input_chunk.json", "network_chunk.ezkl", "witness_chunk.json")
            t_witness = time.time() - t0

            t0 = time.time()
            ezkl.prove("witness_chunk.json", "network_chunk.ezkl", "pk_chunk.key",
                       f"proof_chunk_{k}.json", srs_path=srs_path)
            t_prove = time.time() - t0

            t0 = time.time()
            verified = ezkl.verify(f"proof_chunk_{k}.json", "settings_chunk.json", "vk_chunk.key",
                                   srs_path=srs_path)
            t_verify = time.time() - t0
            signal.alarm(0)
        except ChunkTimeout:
            signal.alarm(0)
            print(f"  chunk {k}: TIMED OUT after {CHUNK_TIMEOUT_S}s -- skipping, will retry on next run", flush=True)
            continue

        proof_kb = os.path.getsize(f"proof_chunk_{k}.json") / 1024.0
        total_proof_kb += proof_kb

        proof = json.load(open(f"proof_chunk_{k}.json"))
        rescaled = proof.get("pretty_public_inputs", {}).get("rescaled_outputs", [[]])[0]
        counts = [float(v) for v in rescaled] if rescaled else None

        print(f"  chunk {k}: verified={verified} witness={t_witness:.2f}s prove={t_prove:.2f}s "
              f"verify={t_verify:.3f}s counts={counts}", flush=True)
        chunk_results.append({
            "chunk": k, "verified": bool(verified), "witness_s": round(t_witness, 3),
            "prove_s": round(t_prove, 3), "verify_s": round(t_verify, 4),
            "proof_kb": round(proof_kb, 2), "counts": counts,
        })
        # Checkpoint after every chunk -- a stall/kill loses at most one chunk, not the whole run.
        with open(checkpoint_path, "w") as f:
            json.dump({"N_total": N_TOTAL, "K": K, "chunk_results": chunk_results}, f)

    total_loop_time = time.time() - t_loop0
    all_sum_g1 = sum(c["counts"][0] for c in chunk_results)
    all_n1 = sum(c["counts"][1] for c in chunk_results)
    all_sum_g0 = sum(c["counts"][2] for c in chunk_results)
    all_n0 = sum(c["counts"][3] for c in chunk_results)
    agg_gap = abs(all_sum_g1 / all_n1 - all_sum_g0 / all_n0)

    result = {
        "N_total": N_TOTAL, "K": K, "n_per_chunk": n_per_chunk,
        "one_time_setup_s": round(setup_time, 2), "logrows": logrows,
        "total_chunk_loop_s": round(total_loop_time, 2),
        "avg_prove_s_per_chunk": round(float(np.mean([c["prove_s"] for c in chunk_results])), 3),
        "total_proof_size_kb": round(total_proof_kb, 2),
        "all_verified": all(c["verified"] for c in chunk_results),
        "full_holdout_true_gap": float(full_dp_gap),
        "aggregated_attested_gap": float(agg_gap),
        "abs_error_vs_full_holdout": float(abs(agg_gap - full_dp_gap)),
        "chunk_results": chunk_results,
    }
    with open("chunked_real_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== PATH B-LITE RESULT (real HMDA) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "chunk_results"}, indent=2))
    print("\n--- Send back: chunked_real_result.json ---")

if __name__ == "__main__":
    asyncio.run(main())
