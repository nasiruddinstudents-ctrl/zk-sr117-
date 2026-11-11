"""
mlp_prove_chunk.py <workdir> <chunk_index>

Proves and verifies ONE chunk; run as a subprocess by mlp_driver.py so a hung
native call can be killed by an OS-level timeout. Writes results/chunk_<k>.json
with the attested statistics, the plaintext statistics for the same rows, the
difference between them, timings, proof size and peak memory.
"""
import json, os, resource, sys, time
import numpy as np
import ezkl

wd, k = sys.argv[1], int(sys.argv[2])
os.chdir(wd)
meta = json.load(open("meta.json"))
n = meta["n_per_chunk"]
X = np.load("X_sel.npy", mmap_mode="r")[k*n:(k+1)*n]
s = np.load("second_sel.npy", mmap_mode="r")[k*n:(k+1)*n]
os.makedirs("results", exist_ok=True); os.makedirs("proofs", exist_ok=True)
inp, wit, prf = f"input_{k}.json", f"witness_{k}.json", f"proofs/proof_{k}.json"
json.dump({"input_data": [np.asarray(X, np.float32).flatten().tolist(),
                          np.asarray(s, np.float32).tolist()]}, open(inp, "w"))

t = time.time(); ezkl.gen_witness(inp, "network.ezkl", wit); tw = time.time() - t
t = time.time(); ezkl.prove(wit, "network.ezkl", "pk.key", prf, srs_path=meta["srs_path"]); tp = time.time() - t
t = time.time(); ok = ezkl.verify(prf, "settings.json", "vk.key", srs_path=meta["srs_path"]); tv = time.time() - t

proof = json.load(open(prf))
attested = [float(v) for v in proof["pretty_public_inputs"]["rescaled_outputs"][0]]
plain = json.load(open("chunk_plaintext_stats.json"))[k]
diff = [a - b for a, b in zip(attested, plain)]
if meta["control"] == "dp":
    count_mismatch = float(sum(abs(round(x)) for x in diff))          # rows whose decision differs
    prob_sum_abs_err = 0.0
else:
    count_mismatch = float(sum(abs(round(diff[i])) for i in range(len(diff)) if i % 3 != 2))
    prob_sum_abs_err = float(sum(abs(diff[i]) for i in range(2, len(diff), 3)))

res = dict(chunk=k, verified=bool(ok), attested=attested, plaintext=plain,
           count_mismatch=count_mismatch, prob_sum_abs_err=prob_sum_abs_err,
           witness_s=round(tw, 3), prove_s=round(tp, 3), verify_s=round(tv, 4),
           proof_kb=round(os.path.getsize(prf) / 1024, 2),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
json.dump(res, open(f"results/chunk_{k}.json", "w"))
for f in (inp, wit):
    os.remove(f)
print(json.dumps({x: res[x] for x in ("chunk", "verified", "count_mismatch", "prob_sum_abs_err",
                                      "prove_s", "verify_s", "proof_kb", "peak_rss_mb")}))
