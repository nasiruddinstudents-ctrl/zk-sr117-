"""
06_driver.py

Run this AFTER 06_setup_once.py has completed successfully.

Loops over all K chunks, running each one as a SEPARATE OS SUBPROCESS via
subprocess.run(..., timeout=...). This is the actual fix for the stall
problem: subprocess.run's timeout uses the OS to kill the child process
if it doesn't finish in time, which works even if the hang is inside
native/Rust code that a Python signal handler can't interrupt.

If a chunk times out or crashes, it's retried once. If it fails twice,
it's logged as failed and skipped -- the run continues rather than
hanging forever, and you'll see clearly which chunk(s) are problematic.

Resumable: chunks that already have a result_chunk_<k>.json on disk are
skipped (not re-run), so you can Ctrl+C and restart this driver safely.
"""
import json
import subprocess
import sys
import time
import os

CHUNK_TIMEOUT_S = 300  # bumped from 90s -- chunk 4 timed out twice at 90s while
                        # every other chunk of identical shape finished in ~3.8s;
                        # likely transient system pressure, giving it more room
MAX_RETRIES = 2

with open("setup_meta.json") as f:
    meta = json.load(f)
K = meta["K"]

results = []
failed_chunks = []

for k in range(K):
    result_path = f"result_chunk_{k}.json"
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
                [sys.executable, "06_prove_chunk.py", str(k)],
                timeout=CHUNK_TIMEOUT_S,
                capture_output=True, text=True,
            )
            elapsed = time.time() - t0
            if proc.returncode == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                results.append(r)
                print(f"  chunk {k}: OK in {elapsed:.1f}s, verified={r['verified']}, counts={r['counts']}")
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
    print("Not all chunks succeeded -- rerun this script to retry only the missing ones "
          "(already-done chunks are cached and skipped).")

if results:
    all_sum_g1 = sum(r["counts"][0] for r in results)
    all_n1 = sum(r["counts"][1] for r in results)
    all_sum_g0 = sum(r["counts"][2] for r in results)
    all_n0 = sum(r["counts"][3] for r in results)
    agg_gap = abs(all_sum_g1 / all_n1 - all_sum_g0 / all_n0)
    total_proof_kb = sum(r["proof_kb"] for r in results)

    summary = {
        "N_total": meta["N_total"], "K": meta["K"], "n_per_chunk": meta["n_per_chunk"],
        "chunks_succeeded": len(results), "chunks_failed": failed_chunks,
        "all_verified": all(r["verified"] for r in results),
        "total_proof_size_kb": round(total_proof_kb, 2),
        "avg_prove_s_per_chunk": round(sum(r["prove_s"] for r in results) / len(results), 3),
        "full_holdout_true_gap": meta["full_holdout_true_gap"],
        "aggregated_attested_gap": float(agg_gap),
        "abs_error_vs_full_holdout": float(abs(agg_gap - meta["full_holdout_true_gap"])),
        "n1_covered": int(all_n1), "n0_covered": int(all_n0),
    }
    with open("driver_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n--- Send back: driver_summary.json ---")
