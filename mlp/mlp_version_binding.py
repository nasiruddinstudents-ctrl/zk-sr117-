"""
mlp_version_binding.py <workdir_model_A> <workdir_model_B>

Model-version binding test (same idea as 12_/16_version_binding.py):
each model's chunk-0 proof must verify under its OWN verifying key and be
REJECTED under the other model's key. Both workdirs must already have
results/chunk_0.json (run mlp_driver.py <wd> --chunks 1 first).
"""
import json, os, sys
import ezkl

A, B = sys.argv[1], sys.argv[2]

def check(proof_wd, key_wd):
    meta = json.load(open(os.path.join(key_wd, "meta.json")))
    try:
        ok = ezkl.verify(os.path.join(proof_wd, "proofs", "proof_0.json"),
                         os.path.join(key_wd, "settings.json"), os.path.join(key_wd, "vk.key"),
                         srs_path=meta["srs_path"])
        return {"accepted": bool(ok), "error": None}
    except Exception as e:  # EZKL raises on a constraint-system mismatch
        return {"accepted": False, "error": str(e)[:300]}

res = {"A_proof_with_A_key": check(A, A), "B_proof_with_B_key": check(B, B),
       "A_proof_with_B_key": check(A, B), "B_proof_with_A_key": check(B, A)}
res["binding_confirmed"] = (res["A_proof_with_A_key"]["accepted"] and res["B_proof_with_B_key"]["accepted"]
                            and not res["A_proof_with_B_key"]["accepted"]
                            and not res["B_proof_with_A_key"]["accepted"])
out = os.path.join(A, "version_binding.json")
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=1))
