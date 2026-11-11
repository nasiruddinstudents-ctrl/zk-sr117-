"""
mlp_driver.py <workdir> [--timeout SECONDS] [--chunks K]

Proves every chunk of a configuration as a separate subprocess with a timeout,
resumably (finished chunks are skipped), then aggregates the attested
statistic in the clear and writes summary.json:

  attested_stat                  -- from the K proofs' published outputs
  committed_rows_plaintext_stat  -- same rows, computed in floating point
  circuit_precision_delta        -- |attested - committed plaintext|  (circuit accuracy)
  full_holdout_stat              -- whole holdout, floating point
  sampling_delta                 -- |committed plaintext - full holdout| (sampling design)
"""
import argparse, json, os, subprocess, sys, time
import mlp_common as C

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--timeout", type=int, default=3600)
ap.add_argument("--chunks", type=int, default=None, help="prove only the first K chunks (timing tests)")
ap.add_argument("--retries", type=int, default=2)
a = ap.parse_args()

wd = a.workdir
meta = json.load(open(os.path.join(wd, "meta.json")))
K = a.chunks or meta["K"]
here = os.path.dirname(os.path.abspath(__file__))
results, failed = [], []
for k in range(K):
    rp = os.path.join(wd, "results", f"chunk_{k}.json")
    if os.path.exists(rp):
        results.append(json.load(open(rp))); continue
    ok = False
    for attempt in range(1, a.retries + 1):
        t = time.time()
        try:
            pr = subprocess.run([sys.executable, os.path.join(here, "mlp_prove_chunk.py"), wd, str(k)],
                                timeout=a.timeout, capture_output=True, text=True)
            if pr.returncode == 0 and os.path.exists(rp):
                r = json.load(open(rp)); results.append(r); ok = True
                print(f"chunk {k}/{K}: verified={r['verified']} prove={r['prove_s']}s "
                      f"mismatch={r['count_mismatch']} mem={r['peak_rss_mb']}MB", flush=True)
                break
            print(f"chunk {k}: failed (rc={pr.returncode}) after {time.time()-t:.0f}s\n{pr.stderr[-800:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"chunk {k}: TIMEOUT after {a.timeout}s (attempt {attempt})", flush=True)
    if not ok:
        failed.append(k)

summary = dict(config={x: meta[x] for x in ("control", "model_type", "hidden", "n_params", "version",
                                            "n_per_chunk", "K", "N_total", "scale", "logrows",
                                            "num_rows", "test_srs", "nonce", "pk_mb", "holdout_auc")},
               chunks_proved=len(results), chunks_requested=K, chunks_failed=failed)
if results:
    attested, tot = C.aggregate([r["attested"] for r in results], meta["control"])
    plain, _ = C.aggregate([r["plaintext"] for r in results], meta["control"])
    summary.update(
        all_verified=all(r["verified"] for r in results),
        rows_covered=len(results) * meta["n_per_chunk"],
        attested_stat=attested, committed_rows_plaintext_stat=plain,
        circuit_precision_delta=abs(attested - plain),
        full_holdout_stat=meta["full_holdout_stat"],
        sampling_delta=abs(plain - meta["full_holdout_stat"]),
        total_decision_mismatches=sum(r["count_mismatch"] for r in results),
        total_prob_sum_abs_err=sum(r["prob_sum_abs_err"] for r in results),
        prove_s_mean=sum(r["prove_s"] for r in results) / len(results),
        prove_s_min=min(r["prove_s"] for r in results), prove_s_max=max(r["prove_s"] for r in results),
        verify_s_total=sum(r["verify_s"] for r in results),
        proof_kb_total=sum(r["proof_kb"] for r in results),
        peak_rss_mb_max=max(r["peak_rss_mb"] for r in results),
        setup_timings_s=meta["timings_s"],
        aggregated_public_outputs=tot)
    if len(results) < meta["K"]:
        summary["note"] = "PARTIAL: statistic covers only the chunks proved so far."
out = os.path.join(wd, "summary.json" if K == meta["K"] else f"summary_first{K}.json")
json.dump(summary, open(out, "w"), indent=2)
print(json.dumps({x: summary.get(x) for x in ("chunks_proved", "all_verified", "attested_stat",
      "committed_rows_plaintext_stat", "circuit_precision_delta", "full_holdout_stat",
      "total_decision_mismatches", "prove_s_mean", "peak_rss_mb_max")}, indent=1))
