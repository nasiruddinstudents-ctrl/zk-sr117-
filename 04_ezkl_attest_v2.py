"""
04_ezkl_attest_v2.py

Run this LOCALLY (needs real internet access to fetch the ceremony SRS --
this is the piece the sandbox couldn't do). Same commit-prove-verify cycle
as before, but:
  - uses ezkl.get_srs() (the real, audited Powers-of-Tau ceremony file)
    instead of gen_srs() (local, testing-only)
  - runs against N=1024 instead of N=64

pip install ezkl   (if not already installed)
"""
import asyncio
import json
import time
import os
import numpy as np
import ezkl

MODEL = "sr117_circuit.onnx"

async def main():
    timings = {}

    X = np.load("batch_X.npy").astype(np.float32)
    group = np.load("batch_group.npy").astype(np.float32)
    input_data = {"input_data": [X.flatten().tolist(), group.flatten().tolist()]}
    with open("input.json", "w") as f:
        json.dump(input_data, f)

    t0 = time.time()
    ezkl.gen_settings(MODEL, "settings.json")
    timings["gen_settings_s"] = time.time() - t0

    t0 = time.time()
    ezkl.calibrate_settings(
        "input.json", MODEL, "settings.json",
        target="resources", lookup_safety_margin=2, scale_rebase_multiplier=[1],
    )
    timings["calibrate_settings_s"] = time.time() - t0

    t0 = time.time()
    ok = ezkl.compile_circuit(MODEL, "network.ezkl", "settings.json")
    timings["compile_circuit_s"] = time.time() - t0
    assert ok, "compile_circuit failed"

    # ---- REAL ceremony SRS (this is the Step 1 fix for the SRS limitation) ----
    t0 = time.time()
    await ezkl.get_srs("settings.json")
    timings["get_srs_s"] = time.time() - t0

    with open("settings.json") as f:
        logrows = json.load(f)["run_args"]["logrows"]
    srs_path = os.path.expanduser(f"~/.ezkl/srs/kzg{logrows}.srs")

    t0 = time.time()
    ok = ezkl.setup("network.ezkl", "vk.key", "pk.key", srs_path=srs_path)
    timings["setup_s"] = time.time() - t0
    assert ok, "setup failed"

    t0 = time.time()
    ezkl.gen_witness("input.json", "network.ezkl", "witness.json")
    timings["gen_witness_s"] = time.time() - t0

    t0 = time.time()
    ezkl.prove("witness.json", "network.ezkl", "pk.key", "proof.json", srs_path=srs_path)
    timings["prove_s"] = time.time() - t0

    t0 = time.time()
    verified = ezkl.verify("proof.json", "settings.json", "vk.key", srs_path=srs_path)
    timings["verify_s"] = time.time() - t0

    proof_size_kb = os.path.getsize("proof.json") / 1024.0

    with open("proof.json") as f:
        proof = json.load(f)

    result = {
        "verified": bool(verified),
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "proof_size_kb": round(proof_size_kb, 2),
        "srs": "REAL ceremony SRS via ezkl.get_srs()",
        "logrows": logrows,
    }
    with open("attestation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nRescaled output (attested demographic parity gap):")
    print(proof.get("pretty_public_inputs", {}).get("rescaled_outputs"))
    print("\n--- Send back: attestation_result.json (and paste this printed output) ---")

if __name__ == "__main__":
    asyncio.run(main())
