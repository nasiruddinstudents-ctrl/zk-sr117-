"""
mlp_common.py -- shared data preparation, model training, sampling and circuit
builders for the MLP extension of ZK-SR117.

Everything upstream of the model is IDENTICAL to the logistic-regression
pipeline in the paper (06_setup_once_clipped.py / 13_ece_setup_once.py):
same HMDA filters, same race coding, same 75/25 split (random_state=11),
same 0.5-99.5 percentile clip fit on the training split, same
standardization, and the same nonce-derived stratified row selection
(demographic parity: nonce seed; ECE: nonce + "-ECE", exactly as in
13_ece_setup_once.py). The attested rows are therefore the same rows the
paper's LR results were computed on, so LR and MLP results are directly
comparable.
"""
import hashlib
import os
import numpy as np
import pandas as pd
import onnx
from onnx import helper, TensorProto, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

FEATURES = ["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"]
RACES = ["White", "Black or African American", "Asian",
         "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander"]
EXAMINER_NONCE = "SR117-2026Q3-EXAMINER-NONCE-001"
CLIP_LOW_PCTILE, CLIP_HIGH_PCTILE = 0.5, 99.5
N_TOTAL = int(os.environ.get("ZK_N_TOTAL", 32768))  # override only for smoke tests
B = 10

# Model A = paper's split; Model B = reshuffled split used for version binding
SPLITS = {"A": dict(test_size=0.25, random_state=11), "B": dict(test_size=0.10, random_state=99)}


def load_and_prepare(csv_path, model_version="A"):
    df = pd.read_csv(csv_path)
    df = df[df["derived_race"].isin(RACES)].copy()
    df["protected_group"] = (df["derived_race"] != "White").astype(int)
    X = df[FEATURES].values
    y = df["action_taken_binary"].values
    g = df["protected_group"].values
    sp = SPLITS[model_version]
    X_tr, X_ho, y_tr, y_ho, g_tr, g_ho = train_test_split(
        X, y, g, test_size=sp["test_size"], random_state=sp["random_state"], stratify=y)
    lo = np.percentile(X_tr, CLIP_LOW_PCTILE, axis=0)
    hi = np.percentile(X_tr, CLIP_HIGH_PCTILE, axis=0)
    X_tr, X_ho = np.clip(X_tr, lo, hi), np.clip(X_ho, lo, hi)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return dict(X_tr=((X_tr - mu) / sd).astype(np.float32), y_tr=y_tr, g_tr=g_tr,
                X_ho=((X_ho - mu) / sd).astype(np.float32), y_ho=y_ho, g_ho=g_ho,
                n_rows=int(len(df)), clip_low=lo.tolist(), clip_high=hi.tolist())


def train_model(d, model_type, hidden, seed=11):
    """Returns (predict_proba_fn, layers) where layers = [(W, b), ...] in float32.

    LR: same estimator as the paper (class_weight='balanced').
    MLP: sklearn MLPClassifier with ReLU. MLPClassifier has no class_weight,
    so the training set is class-balanced by deterministic oversampling of the
    minority class (seed fixed), which plays the same role as the LR's
    class_weight='balanced'.
    """
    X, y = d["X_tr"], d["y_tr"]
    if model_type == "lr":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
        layers = [(clf.coef_[0].reshape(-1, 1).astype(np.float32),
                   np.array([clf.intercept_[0]], np.float32))]
        return (lambda Z: clf.predict_proba(Z)[:, 1]), layers, int(clf.coef_.size + 1)

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    minority, majority = (neg, pos) if len(neg) < len(pos) else (pos, neg)
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    idx = rng.permutation(np.concatenate([majority, minority, extra]))
    clf = MLPClassifier(hidden_layer_sizes=tuple(hidden), activation="relu", max_iter=200,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                        random_state=seed).fit(X[idx], y[idx])
    layers = [(W.astype(np.float32), b.astype(np.float32)) for W, b in zip(clf.coefs_, clf.intercepts_)]
    nparams = int(sum(W.size + b.size for W, b in layers))
    return (lambda Z: clf.predict_proba(Z)[:, 1]), layers, nparams


def select_rows(d, control):
    """Nonce-controlled stratified selection, identical to the paper's scripts."""
    nonce = EXAMINER_NONCE if control == "dp" else EXAMINER_NONCE + "-ECE"
    seed = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    g = d["g_ho"]
    n1 = min(int(round(N_TOTAL * g.mean())), int((g == 1).sum()))
    n0 = N_TOTAL - n1
    s1 = rng.choice(np.where(g == 1)[0], size=n1, replace=False)
    s0 = rng.choice(np.where(g == 0)[0], size=n0, replace=False)
    all_idx = np.concatenate([s1, s0])
    rng.shuffle(all_idx)
    return all_idx, nonce


def plaintext_stats(p, y, g, control):
    """Float (non-circuit) statistics for a set of rows, same definitions as the circuit."""
    pred = (p > 0.5).astype(float)
    if control == "dp":
        return [float((pred * g).sum()), float(g.sum()),
                float((pred * (1 - g)).sum()), float((1 - g).sum())]
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, B + 1)
    out = []
    for bi in range(B):
        m = (p >= edges[bi]) & (p < edges[bi + 1])
        out += [float(m.sum()), float(correct[m].sum()), float(p[m].sum())]
    return out


def aggregate(stats_list, control):
    tot = np.sum(np.array(stats_list, dtype=float), axis=0)
    if control == "dp":
        s1, n1, s0, n0 = tot
        return float(abs(s1 / n1 - s0 / n0)), tot.tolist()
    n_b, c_b, s_b = tot[0::3], tot[1::3], tot[2::3]
    return float(np.abs(c_b - s_b).sum() / n_b.sum()), tot.tolist()


def build_circuit(layers, control, n, path):
    """Model layers (ReLU between hidden layers) -> sigmoid -> control head."""
    D = layers[0][0].shape[0]
    X_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, D])
    second = "group" if control == "dp" else "y"
    s_in = helper.make_tensor_value_info(second, TensorProto.FLOAT, [n])
    out_dim = 4 if control == "dp" else 3 * B
    out = helper.make_tensor_value_info("stats", TensorProto.FLOAT, [out_dim])
    inits, nodes = [], []
    N = helper.make_node

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))

    h = "X"
    for li, (W, b) in enumerate(layers):
        const(f"W{li}", W); const(f"b{li}", b)
        nodes += [N("MatMul", [h, f"W{li}"], [f"mm{li}"]), N("Add", [f"mm{li}", f"b{li}"], [f"z{li}"])]
        if li < len(layers) - 1:
            nodes.append(N("Relu", [f"z{li}"], [f"h{li}"]))
            h = f"h{li}"
    last = f"z{len(layers) - 1}"
    inits.append(numpy_helper.from_array(np.array([n], np.int64), name="shape_n"))
    const("half", [0.5]); const("one", [1.0])
    nodes += [N("Reshape", [last, "shape_n"], ["logits"]), N("Sigmoid", ["logits"], ["probs"]),
              N("Greater", ["probs", "half"], ["pb"]), N("Cast", ["pb"], ["preds"], to=TensorProto.FLOAT)]

    if control == "dp":
        nodes += [N("Sub", ["one", "group"], ["ng"]),
                  N("Mul", ["preds", "group"], ["p1"]), N("Mul", ["preds", "ng"], ["p0"]),
                  N("ReduceSum", ["p1"], ["s1"], keepdims=1), N("ReduceSum", ["group"], ["n1"], keepdims=1),
                  N("ReduceSum", ["p0"], ["s0"], keepdims=1), N("ReduceSum", ["ng"], ["n0"], keepdims=1),
                  N("Concat", ["s1", "n1", "s0", "n0"], ["stats"], axis=0)]
    else:
        nodes += [N("Sub", ["preds", "y"], ["dpy"]), N("Abs", ["dpy"], ["adpy"]),
                  N("Sub", ["one", "adpy"], ["correct"])]
        edges = np.linspace(0, 1, B + 1)
        cat = []
        for bi in range(B):
            const(f"lo{bi}", [edges[bi]]); const(f"hi{bi}", [edges[bi + 1]])
            nodes += [N("GreaterOrEqual", ["probs", f"lo{bi}"], [f"ge{bi}"]),
                      N("Less", ["probs", f"hi{bi}"], [f"lt{bi}"]),
                      N("And", [f"ge{bi}", f"lt{bi}"], [f"ib{bi}"]),
                      N("Cast", [f"ib{bi}"], [f"in{bi}"], to=TensorProto.FLOAT),
                      N("ReduceSum", [f"in{bi}"], [f"n{bi}"], keepdims=1),
                      N("Mul", [f"in{bi}", "correct"], [f"ic{bi}"]),
                      N("ReduceSum", [f"ic{bi}"], [f"c{bi}"], keepdims=1),
                      N("Mul", [f"in{bi}", "probs"], [f"ip{bi}"]),
                      N("ReduceSum", [f"ip{bi}"], [f"s{bi}"], keepdims=1)]
            cat += [f"n{bi}", f"c{bi}", f"s{bi}"]
        nodes.append(N("Concat", cat, ["stats"], axis=0))

    graph = helper.make_graph(nodes, f"zk_sr117_{control}", [X_in, s_in], [out], initializer=inits)
    model = helper.make_model(graph, producer_name="zk-sr117-mlp", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
