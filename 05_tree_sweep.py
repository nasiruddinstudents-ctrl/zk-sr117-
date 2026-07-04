"""
05_tree_sweep.py

Run this LOCALLY. Sweeps the tree-reduction circuit (03_build_onnx_circuit_tree.py's
logic, inlined here) across N = 1024, 4096, 16384, 32768 on SYNTHETIC data
matching your real batch's subgroup prevalence (18.8% nonwhite), to find
where -- if anywhere -- the tree-reduction fix actually breaks down, and
how long gen_settings/calibrate_settings/prove take at each size.

This can take a while at the larger N values. Run it in the background so
it survives if your terminal session gets interrupted:

    nohup python3 05_tree_sweep.py > sweep_log.txt 2>&1 &

Then check progress with:

    tail -f sweep_log.txt

When it's done (or you want to stop and see partial results), send back
sweep_results.json -- it's tiny.
"""
import json
import math
import time
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def build_tree_circuit(N, D, W, b, path):
    N_padded = next_pow2(N)
    pad_amount = N_padded - N

    def tree_sum(nodes, initializers, input_name, N_padded, prefix):
        cur = input_name
        cur_len = N_padded
        level = 0
        while cur_len > 1:
            half = cur_len // 2
            left = f"{prefix}_l{level}_left"
            right = f"{prefix}_l{level}_right"
            summed = f"{prefix}_l{level}_sum"
            split_sizes_name = f"{prefix}_l{level}_splitsizes"
            initializers.append(numpy_helper.from_array(
                np.array([half, half], dtype=np.int64), name=split_sizes_name))
            nodes.append(helper.make_node(
                "Split", [cur, split_sizes_name], [left, right], axis=0,
                name=f"{prefix}_split{level}"))
            nodes.append(helper.make_node("Add", [left, right], [summed], name=f"{prefix}_add{level}"))
            cur = summed
            cur_len = half
            level += 1
        return cur

    X_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, D])
    group_in = helper.make_tensor_value_info("group", TensorProto.FLOAT, [N])
    gap_out = helper.make_tensor_value_info("gap", TensorProto.FLOAT, [1])

    W_init = numpy_helper.from_array(W.reshape(D, 1), name="W")
    b_init = numpy_helper.from_array(b, name="b")
    half_const = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half_const")
    one_const = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one_const")
    shape_N = numpy_helper.from_array(np.array([N], dtype=np.int64), name="shape_N")
    pad_zeros = numpy_helper.from_array(np.zeros(pad_amount, dtype=np.float32), name="pad_zeros")

    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["logits2d"]),
        helper.make_node("Reshape", ["logits2d", "shape_N"], ["logits2d_r"]),
        helper.make_node("Add", ["logits2d_r", "b"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["probs"]),
        helper.make_node("Greater", ["probs", "half_const"], ["preds_bool"]),
        helper.make_node("Cast", ["preds_bool"], ["preds"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one_const", "group"], ["not_group"]),
        helper.make_node("Mul", ["preds", "group"], ["preds_g1"]),
        helper.make_node("Mul", ["preds", "not_group"], ["preds_g0"]),
    ]
    initializers = [W_init, b_init, half_const, one_const, shape_N]

    if pad_amount > 0:
        nodes.append(helper.make_node("Concat", ["preds_g1", "pad_zeros"], ["preds_g1_pad"], axis=0))
        nodes.append(helper.make_node("Concat", ["preds_g0", "pad_zeros"], ["preds_g0_pad"], axis=0))
        nodes.append(helper.make_node("Concat", ["group", "pad_zeros"], ["group_pad"], axis=0))
        nodes.append(helper.make_node("Concat", ["not_group", "pad_zeros"], ["not_group_pad"], axis=0))
        initializers.append(pad_zeros)
        sum_g1_in, sum_g0_in, n1_in, n0_in = "preds_g1_pad", "preds_g0_pad", "group_pad", "not_group_pad"
    else:
        sum_g1_in, sum_g0_in, n1_in, n0_in = "preds_g1", "preds_g0", "group", "not_group"

    sum_g1 = tree_sum(nodes, initializers, sum_g1_in, N_padded, "sg1")
    sum_g0 = tree_sum(nodes, initializers, sum_g0_in, N_padded, "sg0")
    n1 = tree_sum(nodes, initializers, n1_in, N_padded, "n1")
    n0 = tree_sum(nodes, initializers, n0_in, N_padded, "n0")

    nodes.append(helper.make_node("Div", [sum_g1, n1], ["approve1"]))
    nodes.append(helper.make_node("Div", [sum_g0, n0], ["approve0"]))
    nodes.append(helper.make_node("Sub", ["approve1", "approve0"], ["diff"]))
    nodes.append(helper.make_node("Abs", ["diff"], ["gap"]))

    graph = helper.make_graph(nodes, "tree", [X_in, group_in], [gap_out], initializer=initializers)
    model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return len(nodes)


with open("plaintext_metrics.json") as f:
    m = json.load(f)
W = np.array(m["coef"], dtype=np.float32)
b = np.array([m["intercept"]], dtype=np.float32)
true_prop_g1 = m.get("true_nonwhite_prevalence_in_holdout", 0.1882)

results = []
for N in [1024, 4096, 16384, 32768]:
    print(f"\n=== N={N} ===", flush=True)
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(N, 4)).astype(np.float32)
    g = rng.binomial(1, true_prop_g1, size=N).astype(np.float32)
    np.save("batch_X.npy", X)
    np.save("batch_group.npy", g)

    onnx_path = "sr117_circuit_tree.onnx"
    n_nodes = build_tree_circuit(N, 4, W, b, onnx_path)
    print(f"circuit built: {n_nodes} nodes", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    true_out = float(sess.run(["gap"], {"X": X, "group": g})[0][0])

    input_data = {"input_data": [X.flatten().tolist(), g.flatten().tolist()]}
    with open("input_sweep.json", "w") as f:
        json.dump(input_data, f)

    import ezkl
    t0 = time.time()
    try:
        py_args = ezkl.PyRunArgs()
        py_args.input_scale = 16
        py_args.param_scale = 16
        ezkl.gen_settings(onnx_path, "settings_sweep.json", py_run_args=py_args)
        ezkl.calibrate_settings(
            "input_sweep.json", onnx_path, "settings_sweep.json",
            target="accuracy", lookup_safety_margin=2, scale_rebase_multiplier=[1],
        )
        settings_time = time.time() - t0
        with open("settings_sweep.json") as f:
            s = json.load(f)
        logrows = s["run_args"]["logrows"]
        print(f"settings+calibrate: {settings_time:.1f}s, logrows={logrows}", flush=True)

        result = {
            "N": N, "n_nodes": n_nodes, "settings_calibrate_s": round(settings_time, 2),
            "logrows": logrows, "true_gap_onnx": true_out, "status": "settings_ok",
        }
    except Exception as e:
        result = {"N": N, "n_nodes": n_nodes, "status": "settings_FAILED", "error": repr(e)}
        results.append(result)
        with open("sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"FAILED at settings stage: {e}", flush=True)
        continue

    results.append(result)
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("\n=== SWEEP DONE ===")
print(json.dumps(results, indent=2))
