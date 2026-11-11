"""
mlp_setup.py -- one-time setup for one (control, model, chunk size, scale) configuration.

  python3 mlp_setup.py --control dp  --model-type mlp --hidden 16,8 --n 1024 --scale 8 --workdir runs/dp_mlp_n1024_s8
  python3 mlp_setup.py --control ece --model-type mlp --hidden 16,8 --n 1024 --scale 8 --workdir runs/ece_mlp_n1024_s8

Trains the model (same data pipeline as the paper), selects the nonce-controlled
N_TOTAL rows, builds the per-chunk circuit, calibrates EZKL at the requested
fixed-point scale, compiles, fetches the production Powers-of-Tau SRS
(ezkl.get_srs; use --test-srs only for local smoke tests), and generates keys.
Writes everything into --workdir.
"""
import argparse, asyncio, json, os, time
import numpy as np
import ezkl
import mlp_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["dp", "ece"], required=True)
    ap.add_argument("--model-type", choices=["mlp", "lr"], default="mlp")
    ap.add_argument("--hidden", default="16,8")
    ap.add_argument("--n", type=int, default=1024, help="rows per chunk")
    ap.add_argument("--scale", type=int, default=8, help="EZKL input/param fixed-point scale")
    ap.add_argument("--version", choices=["A", "B"], default="A", help="A = paper split; B = binding-test model")
    ap.add_argument("--csv", default="hmda_sample.csv")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--test-srs", action="store_true", help="local test-only SRS (never for reported results)")
    ap.add_argument("--paper-calibration", action="store_true",
                    help="calibrate exactly as the paper's LR scripts (scale 16 start, EZKL free to pick scales)")
    a = ap.parse_args()

    hidden = [int(h) for h in a.hidden.split(",")] if a.model_type == "mlp" else []
    csv = os.path.abspath(a.csv)
    os.makedirs(a.workdir, exist_ok=True)
    os.chdir(a.workdir)
    T = {}

    t = time.time()
    d = C.load_and_prepare(csv, a.version)
    predict, layers, nparams = C.train_model(d, a.model_type, hidden)
    T["train_s"] = time.time() - t
    np.savez("model_weights.npz", **{f"W{i}": W for i, (W, _) in enumerate(layers)},
             **{f"b{i}": b for i, (_, b) in enumerate(layers)})

    p_ho = predict(d["X_ho"])
    second_ho = d["g_ho"] if a.control == "dp" else d["y_ho"]
    full_stat, _ = C.aggregate([C.plaintext_stats(p_ho, d["y_ho"], d["g_ho"], a.control)], a.control)
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(d["y_ho"], p_ho))
    except Exception:
        pass

    idx, nonce = C.select_rows(d, a.control)
    if C.N_TOTAL % a.n:
        raise SystemExit(f"N_TOTAL={C.N_TOTAL} must be divisible by n={a.n}")
    K = C.N_TOTAL // a.n
    X_sel = d["X_ho"][idx].astype(np.float32)
    s_sel = second_ho[idx].astype(np.float32)
    p_sel, y_sel, g_sel = p_ho[idx], d["y_ho"][idx], d["g_ho"][idx]
    np.save("X_sel.npy", X_sel)
    np.save("second_sel.npy", s_sel)
    chunk_plain = [C.plaintext_stats(p_sel[k*a.n:(k+1)*a.n], y_sel[k*a.n:(k+1)*a.n],
                                     g_sel[k*a.n:(k+1)*a.n], a.control) for k in range(K)]
    committed_stat, _ = C.aggregate(chunk_plain, a.control)
    json.dump(chunk_plain, open("chunk_plaintext_stats.json", "w"))

    C.build_circuit(layers, a.control, a.n, "circuit.onnx")
    json.dump({"input_data": [X_sel[:a.n].flatten().tolist(), s_sel[:a.n].tolist()]},
              open("calib_input.json", "w"))

    t = time.time()
    ra = ezkl.PyRunArgs()
    ra.input_scale = a.scale
    ra.param_scale = a.scale
    ezkl.gen_settings("circuit.onnx", "settings.json", py_run_args=ra)
    if a.paper_calibration:
        ezkl.calibrate_settings("calib_input.json", "circuit.onnx", "settings.json", target="accuracy",
                                lookup_safety_margin=2, scale_rebase_multiplier=[1])
    else:
        ezkl.calibrate_settings("calib_input.json", "circuit.onnx", "settings.json", target="accuracy",
                                scales=[a.scale], lookup_safety_margin=2, scale_rebase_multiplier=[1])
    ezkl.compile_circuit("circuit.onnx", "network.ezkl", "settings.json")
    T["settings_calibrate_compile_s"] = time.time() - t
    st = json.load(open("settings.json"))
    logrows = st["run_args"]["logrows"]

    t = time.time()
    if a.test_srs:
        srs_path = os.path.abspath(f"kzg{logrows}_TEST_ONLY.srs")
        if not os.path.exists(srs_path):
            ezkl.gen_srs(srs_path, logrows)
    else:
        async def _fetch():
            return await ezkl.get_srs("settings.json")
        asyncio.run(_fetch())
        srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")
    T["srs_s"] = time.time() - t

    t = time.time()
    ezkl.setup("network.ezkl", "vk.key", "pk.key", srs_path=srs_path)
    T["setup_s"] = time.time() - t

    meta = dict(control=a.control, model_type=a.model_type, hidden=hidden, n_params=nparams,
                version=a.version, n_per_chunk=a.n, K=K, N_total=C.N_TOTAL, scale=a.scale,
                logrows=logrows, final_input_scale=st["run_args"]["input_scale"],
                paper_calibration=bool(a.paper_calibration), num_rows=st.get("num_rows"), srs_path=srs_path,
                test_srs=bool(a.test_srs), nonce=nonce, analysis_rows=d["n_rows"],
                n_holdout=int(len(d["y_ho"])), holdout_auc=auc,
                full_holdout_stat=full_stat, committed_rows_plaintext_stat=committed_stat,
                pk_mb=round(os.path.getsize("pk.key") / 2**20, 1),
                vk_kb=round(os.path.getsize("vk.key") / 1024, 1),
                clip_low=d["clip_low"], clip_high=d["clip_high"],
                timings_s={k: round(v, 2) for k, v in T.items()})
    json.dump(meta, open("meta.json", "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("control", "model_type", "hidden", "n_params", "n_per_chunk",
                                           "K", "scale", "logrows", "full_holdout_stat",
                                           "committed_rows_plaintext_stat", "pk_mb", "timings_s")}, indent=1))


if __name__ == "__main__":
    main()
