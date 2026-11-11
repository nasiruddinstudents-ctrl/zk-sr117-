#!/usr/bin/env bash
# run_all.sh -- MLP extension of ZK-SR117 on a rented machine (e.g., Vast.ai).
#
#   bash run_all.sh install                 # install EZKL 23.0.5 + deps, run a 2-minute smoke test
#   bash run_all.sh timing                  # one-chunk timing/precision grid -> timing_table.txt
#   bash run_all.sh full N SCALE_DP SCALE_ECE   # full 32,768-row runs + binding, e.g.  full 1024 8 12
#   bash run_all.sh pack                    # bundle results into results_mlp.tgz for download
#
# Run long stages under nohup so they survive a dropped SSH connection:
#   nohup bash run_all.sh timing > timing.log 2>&1 &      then   tail -f timing.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=python3
HIDDEN="${HIDDEN:-16,8}"

need_csv() { [ -f hmda_sample.csv ] || { echo "ERROR: hmda_sample.csv not found in $HERE"; exit 1; }; }

case "${1:-}" in
install)
  $PY -m pip install -q "ezkl==23.0.5" onnx onnxruntime pandas scikit-learn numpy \
    || $PY -m pip install -q --break-system-packages "ezkl==23.0.5" onnx onnxruntime pandas scikit-learn numpy
  $PY -c "import ezkl, onnx, sklearn; print('ezkl', ezkl.__version__)"
  if curl -sfI https://kzg.ezkl.xyz/kzg15.srs >/dev/null; then echo "Production SRS host reachable: OK";
  else echo "WARNING: cannot reach kzg.ezkl.xyz -- the production Powers-of-Tau SRS cannot be downloaded on this machine"; fi
  echo "CPU cores: $(nproc)   Memory: $(free -g | awk '/Mem:/{print $2}') GB   Disk free: $(df -h . | awk 'NR==2{print $4}')"
  need_csv
  echo "--- smoke test (tiny, test-only SRS; checks the pipeline runs on this machine) ---"
  rm -rf runs/_smoke
  ZK_N_TOTAL=128 $PY mlp_setup.py --control dp --hidden 8,4 --n 64 --scale 8 --workdir runs/_smoke --test-srs > runs_smoke.log 2>&1
  ZK_N_TOTAL=128 $PY mlp_driver.py runs/_smoke --chunks 1 | tail -12
  rm -rf runs/_smoke runs_smoke.log
  echo "Smoke test passed."
  ;;

timing)
  need_csv
  # control  model  n     scale  extra
  GRID=$(cat <<EOF
dp  mlp 256  8
dp  mlp 1024 8
dp  mlp 1024 12
ece mlp 1024 8
ece mlp 1024 12
dp  lr  1024 16 --paper-calibration
EOF
)
  echo "$GRID" | while read -r ctl mt n sc extra; do
    wd="runs/timing_${ctl}_${mt}_n${n}_s${sc}"
    [ -f "$wd/summary_first2.json" ] && { echo "skip $wd (done)"; continue; }
    echo "=== $ctl $mt n=$n scale=$sc ==="
    $PY mlp_setup.py --control "$ctl" --model-type "$mt" --hidden "$HIDDEN" --n "$n" --scale "$sc" $extra --workdir "$wd" > "$wd.setup.log" 2>&1 \
      || { echo "setup FAILED (see $wd.setup.log)"; tail -5 "$wd.setup.log"; continue; }
    $PY mlp_driver.py "$wd" --chunks 2 --timeout 5400 || true
    rm -f "$wd/pk.key"      # keys are large; regenerate in the full run
  done
  $PY - <<'EOF'
import glob, json
rows = []
for f in sorted(glob.glob("runs/timing_*/summary_first2.json")):
    s = json.load(open(f)); c = s["config"]; m = json.load(open(f.replace("summary_first2.json", "meta.json")))
    rows.append((c["control"], c["model_type"], c["n_params"], c["n_per_chunk"], c["scale"], c["logrows"], c["pk_mb"],
                 m["timings_s"].get("setup_s"), s.get("prove_s_mean"), s.get("peak_rss_mb_max"),
                 s.get("total_decision_mismatches"), s.get("circuit_precision_delta"), s.get("all_verified")))
hdr = "control model params n scale logrows pk_MB setup_s prove_s/chunk peak_mem_MB mismatches precision_delta verified"
out = [hdr] + [" ".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) for r in rows]
open("timing_table.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
EOF
  ;;

full)
  need_csv
  N="${2:?usage: full N SCALE_DP SCALE_ECE}"; SDP="${3:?}"; SECE="${4:?}"
  for spec in "dp $SDP" "ece $SECE"; do
    set -- $spec; ctl=$1; sc=$2
    A="runs/full_${ctl}_mlp_n${N}_s${sc}_A"; Bwd="runs/full_${ctl}_mlp_n${N}_s${sc}_B"
    if [ ! -f "$A/meta.json" ]; then
      echo "=== setup $ctl (Model A) n=$N scale=$sc ==="
      $PY mlp_setup.py --control "$ctl" --hidden "$HIDDEN" --n "$N" --scale "$sc" --workdir "$A" | tail -25
    fi
    echo "=== prove all chunks: $ctl ==="
    $PY mlp_driver.py "$A" --timeout 7200
    if [ ! -f "$Bwd/meta.json" ]; then
      echo "=== setup $ctl (Model B, binding test) ==="
      $PY mlp_setup.py --control "$ctl" --hidden "$HIDDEN" --n "$N" --scale "$sc" --version B --workdir "$Bwd" | tail -5
    fi
    $PY mlp_driver.py "$Bwd" --chunks 1 --timeout 7200
    $PY mlp_version_binding.py "$A" "$Bwd" | tail -3
    rm -f "$Bwd/pk.key"
  done
  echo "Full runs finished. Now: bash run_all.sh pack"
  ;;

pack)
  tar czf results_mlp.tgz --exclude='pk.key' --exclude='*.srs' --exclude='*.npy' --exclude='network.ezkl' \
      --exclude='calib_input.json' timing_table.txt runs/ 2>/dev/null || \
  tar czf results_mlp.tgz --exclude='pk.key' --exclude='*.srs' --exclude='*.npy' --exclude='network.ezkl' \
      --exclude='calib_input.json' runs/
  ls -lh results_mlp.tgz
  echo "Download results_mlp.tgz, then DESTROY the instance in the Vast.ai console."
  ;;

*)
  sed -n '2,12p' "$0"; exit 1 ;;
esac
