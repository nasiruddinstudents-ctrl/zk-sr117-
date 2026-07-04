"""
03_build_onnx_circuit_v2.py

Same circuit as before (linear model -> sigmoid -> threshold -> group-wise
demographic-parity gap), generalized to whatever N the batch files actually
contain (now 1024, from 02_train_real_v2.py's stratified nonce sampling),
instead of a hardcoded N=64.
"""
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

with open("plaintext_metrics.json") as f:
    m = json.load(f)

batch_X = np.load("batch_X.npy")
N, D = batch_X.shape
print(f"Building circuit for N={N}, D={D}")

W = np.array(m["coef"], dtype=np.float32)
b = np.array([m["intercept"]], dtype=np.float32)

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
    helper.make_node("ReduceSum", ["preds_g0"], ["sum_g0"], keepdims=1),
    helper.make_node("ReduceSum", ["group"], ["n1"], keepdims=1),
    helper.make_node("ReduceSum", ["not_group"], ["n0"], keepdims=1),
    helper.make_node("Div", ["sum_g1", "n1"], ["approve1"]),
    helper.make_node("Div", ["sum_g0", "n0"], ["approve0"]),
    helper.make_node("Sub", ["approve1", "approve0"], ["diff"]),
    helper.make_node("Abs", ["diff"], ["gap"]),
]

graph = helper.make_graph(
    nodes, "sr117_dp_gap_circuit_v2", [X_in, group_in], [gap_out],
    initializer=[W_init, b_init, half, one, shape_N],
)
model = helper.make_model(graph, producer_name="zk-sr117", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "sr117_circuit.onnx")
print("Wrote sr117_circuit.onnx")

import onnxruntime as ort
sess = ort.InferenceSession("sr117_circuit.onnx")
group = np.load("batch_group.npy").astype(np.float32)
out = sess.run(["gap"], {"X": batch_X.astype(np.float32), "group": group})[0]
print(f"ONNX circuit output (demographic parity gap on batch): {float(out[0]):.4f}")
