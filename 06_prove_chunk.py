"""
06_prove_chunk.py <chunk_index>

Proves ONE chunk. Meant to be invoked as a subprocess by 06_driver.py, not
run directly (though you can: python3 06_prove_chunk.py 4). Running each
chunk in its own OS process is what makes a real timeout possible -- the
driver can kill this process from outside if it hangs, which a Python-level
signal.alarm cannot do against a blocking native/Rust call.

Writes result_chunk_<k>.json on success.
"""
import json
import os
import sys
import time
import numpy as np
import ezkl

k = int(sys.argv[1])

with open("setup_meta.json") as f:
    meta = json.load(f)
srs_path = meta["srs_path"]

X_hold_std = np.load("X_hold_std.npy")
g_hold = np.load("g_hold.npy")
chunks = np.load("chunks_idx.npz")
idx = chunks[f"chunk_{k}"]

cX = X_hold_std[idx].astype(np.float32)
cg = g_hold[idx].astype(np.float32)
json.dump({"input_data": [cX.flatten().tolist(), cg.flatten().tolist()]},
           open(f"input_chunk_{k}.json", "w"))

t0 = time.time()
ezkl.gen_witness(f"input_chunk_{k}.json", "network_chunk.ezkl", f"witness_chunk_{k}.json")
t_witness = time.time() - t0

t0 = time.time()
ezkl.prove(f"witness_chunk_{k}.json", "network_chunk.ezkl", "pk_chunk.key",
           f"proof_chunk_{k}.json", srs_path=srs_path)
t_prove = time.time() - t0

t0 = time.time()
verified = ezkl.verify(f"proof_chunk_{k}.json", "settings_chunk.json", "vk_chunk.key", srs_path=srs_path)
t_verify = time.time() - t0

proof_kb = os.path.getsize(f"proof_chunk_{k}.json") / 1024.0
proof = json.load(open(f"proof_chunk_{k}.json"))
rescaled = proof.get("pretty_public_inputs", {}).get("rescaled_outputs", [[]])[0]
counts = [float(v) for v in rescaled] if rescaled else None

result = {
    "chunk": k, "verified": bool(verified), "witness_s": round(t_witness, 3),
    "prove_s": round(t_prove, 3), "verify_s": round(t_verify, 4),
    "proof_kb": round(proof_kb, 2), "counts": counts,
}
with open(f"result_chunk_{k}.json", "w") as f:
    json.dump(result, f)

print(json.dumps(result))
