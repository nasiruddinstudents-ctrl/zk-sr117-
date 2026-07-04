"""
03_build_onnx_circuit_tree.py

Path A fix: replace the flat ReduceSum (which accumulates all N quantized
values in one circuit op, overflowing EZKL's calibrated lookup range as N
grows) with an explicit balanced binary-tree reduction: log2(N) levels of
pairwise elementwise Add on split halves. Each individual Add's output
magnitude only ever grows by a constant factor per level, so the *circuit's*
per-op accumulator range scales with O(log N) instead of O(N), even though
the mathematical result (the full sum) is identical.

N is padded up to the next power of two with zeros (padding a "preds*group"
style vector and a "group" mask vector both with zero contributes nothing
to either the numerator or denominator sums, so this is exact, not an
approximation).
"""
import json
import math
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def tree_sum(nodes, initializers, input_name, N_padded, prefix):
    """Append a log2(N_padded)-level binary-tree summation to `nodes`.
    Returns the name of the final scalar (shape (1,)) output."""
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
            name=f"{prefix}_split{level}",
        ))
        nodes.append(helper.make_node("Add", [left, right], [summed], name=f"{prefix}_add{level}"))
        cur = summed
        cur_len = half
        level += 1
    return cur  # shape (1,)


with open("plaintext_metrics.json") as f:
    m = json.load(f)

batch_X = np.load("batch_X.npy")
N, D = batch_X.shape
N_padded = next_pow2(N)
pad_amount = N_padded - N
levels = int(math.log2(N_padded))
print(f"Building tree-reduction circuit for N={N} (padded to {N_padded}, {levels} tree levels), D={D}")

W = np.array(m["coef"], dtype=np.float32)
b = np.array([m["intercept"]], dtype=np.float32)

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

graph = helper.make_graph(
    nodes, "sr117_dp_gap_tree_circuit", [X_in, group_in], [gap_out],
    initializer=initializers,
)
model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "sr117_circuit.onnx")
print("Wrote sr117_circuit.onnx (tree-reduction version)")

import onnxruntime as ort
sess = ort.InferenceSession("sr117_circuit.onnx")
group = np.load("batch_group.npy").astype(np.float32)
out = sess.run(["gap"], {"X": batch_X.astype(np.float32), "group": group})[0]
print(f"ONNX circuit output (demographic parity gap on batch): {float(out[0]):.4f}")
