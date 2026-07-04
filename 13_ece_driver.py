"""
13_ece_driver.py

Run AFTER 13_ece_setup_once.py. Loops over K chunks, each as its own
subprocess with a real OS-enforced timeout (same pattern as 06_driver.py,
which fixed the silent-hang problem the demographic-parity pipeline hit).
Resumable: cached result_ece_chunk_<k>.json files are skipped.
"""
import json
import subprocess
import sys
import time
import os

CHUNK_TIMEOUT_S = 120  # ECE circuit is larger (3B outputs vs 4) than the
                       # demographic-parity circuit; generous vs. the
                       # blueprint's projected 8-15s per chunk
MAX_RETRIES = 2

with open("ece_meta.json") as f:
    meta = json.load(f)
K = meta["K"]
B = meta["B"]

results = []
failed_chunks = []

for k in range(K):
    result_path = f"result_ece_chunk_{k}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            results.append(json.load(f))
        print(f"chunk {k}: already done (cached), skipping")
        continue

    ok = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"chunk {k}: attempt {attempt}/{MAX_RETRIES}...", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "13_ece_prove_chunk.py", str(k)],
                timeout=CHUNK_TIMEOUT_S, capture_output=True, text=True,
            )
            elapsed = time.time() - t0
            if proc.returncode == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                results.append(r)
                print(f"  chunk {k}: OK in {elapsed:.1f}s, verified={r['verified']}")
                ok = True
                break
            else:
                print(f"  chunk {k}: subprocess failed (returncode={proc.returncode}) after {elapsed:.1f}s")
                print(f"  stderr tail: {proc.stderr[-500:] if proc.stderr else '(none)'}")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  chunk {k}: TIMED OUT after {elapsed:.1f}s (limit {CHUNK_TIMEOUT_S}s) -- process killed")
    if not ok:
        print(f"chunk {k}: FAILED after {MAX_RETRIES} attempts, skipping")
        failed_chunks.append(k)

print(f"\n{len(results)}/{K} chunks succeeded. Failed: {failed_chunks}")
if len(results) < K:
    print("Rerun this script to retry only the missing chunks (cached ones are skipped).")

if results:
    n_total = [0.0] * B
    c_total = [0.0] * B
    s_total = [0.0] * B
    for r in results:
        bs = r["bin_stats"]
        for bi in range(B):
            n_total[bi] += bs[3 * bi]
            c_total[bi] += bs[3 * bi + 1]
            s_total[bi] += bs[3 * bi + 2]

    N_covered = sum(n_total)
    ece = sum(abs(c_total[bi] - s_total[bi]) for bi in range(B)) / N_covered
    total_proof_kb = sum(r["proof_kb"] for r in results)

    summary = {
        "N_total": meta["N_total"], "K": meta["K"], "B": B,
        "chunks_succeeded": len(results), "chunks_failed": failed_chunks,
        "all_verified": all(r["verified"] for r in results),
        "total_proof_size_kb": round(total_proof_kb, 2),
        "avg_prove_s_per_chunk": round(sum(r["prove_s"] for r in results) / len(results), 3),
        "min_prove_s": round(min(r["prove_s"] for r in results), 3),
        "max_prove_s": round(max(r["prove_s"] for r in results), 3),
        "total_verify_s_sequential": round(sum(r["verify_s"] for r in results), 4),
        "full_holdout_true_ece": meta["full_holdout_true_ece"],
        "aggregated_attested_ece": float(ece),
        "abs_error_vs_full_holdout": float(abs(ece - meta["full_holdout_true_ece"])),
        "N_covered": int(N_covered),
    }
    with open("ece_driver_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== ECE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n--- Send back: ece_driver_summary.json ---")
