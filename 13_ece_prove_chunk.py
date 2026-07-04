"""
13_ece_prove_chunk.py <chunk_index>

Proves ONE ECE chunk. Invoked as a subprocess by 13_ece_driver.py, same
resilience pattern as 06_prove_chunk.py (a hung witness/prove call inside
one process can be killed from outside by an OS-level timeout; a
Python-level signal cannot interrupt it).

Writes result_ece_chunk_<k>.json on success: [n_0,c_0,s_0, n_1,c_1,s_1, ...].
"""
import json
import os
import sys
import time
import numpy as np
import ezkl

k = int(sys.argv[1])

with open("ece_meta.json") as f:
    meta = json.load(f)
srs_path = meta["srs_path"]

X_hold_std = np.load("X_hold_std_ece.npy")
y_hold = np.load("y_hold_ece.npy")
chunks = np.load("chunks_idx_ece.npz")
idx = chunks[f"chunk_{k}"]

cX = X_hold_std[idx].astype(np.float32)
cy = y_hold[idx].astype(np.float32)
json.dump({"input_data": [cX.flatten().tolist(), cy.flatten().tolist()]},
           open(f"input_ece_{k}.json", "w"))

t0 = time.time()
ezkl.gen_witness(f"input_ece_{k}.json", "network_ece.ezkl", f"witness_ece_{k}.json")
t_witness = time.time() - t0

t0 = time.time()
ezkl.prove(f"witness_ece_{k}.json", "network_ece.ezkl", "pk_ece.key",
           f"proof_ece_{k}.json", srs_path=srs_path)
t_prove = time.time() - t0

t0 = time.time()
verified = ezkl.verify(f"proof_ece_{k}.json", "settings_ece.json", "vk_ece.key", srs_path=srs_path)
t_verify = time.time() - t0

proof_kb = os.path.getsize(f"proof_ece_{k}.json") / 1024.0
proof = json.load(open(f"proof_ece_{k}.json"))
rescaled = proof.get("pretty_public_inputs", {}).get("rescaled_outputs", [[]])[0]
bin_stats = [float(v) for v in rescaled] if rescaled else None

result = {
    "chunk": k, "verified": bool(verified), "witness_s": round(t_witness, 3),
    "prove_s": round(t_prove, 3), "verify_s": round(t_verify, 4),
    "proof_kb": round(proof_kb, 2), "bin_stats": bin_stats,
}
with open(f"result_ece_chunk_{k}.json", "w") as f:
    json.dump(result, f)

print(json.dumps(result))
