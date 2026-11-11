# ZK-SR117 MLP extension

Extends the paper's chunked attestation from the 5-parameter logistic regression to a
multilayer perceptron (default 4 -> 16 -> 8 -> 1, ReLU, 225 parameters) for both controls
(demographic parity and ECE), on the same rows, nonce, preprocessing and splits as the paper.

Files: `mlp_common.py` (data, model, sampling, circuits), `mlp_setup.py` (one configuration),
`mlp_prove_chunk.py` (one chunk), `mlp_driver.py` (all chunks, resumable, summary),
`mlp_version_binding.py` (Model A vs. Model B keys), `run_all.sh` (stages).

Put `hmda_sample.csv` (from `filter_hmda.py`) in this folder, then:

    bash run_all.sh install
    nohup bash run_all.sh timing > timing.log 2>&1 &        # then: tail -f timing.log
    nohup bash run_all.sh full 1024 8 12 > full.log 2>&1 &  # choose N and scales from timing_table.txt
    bash run_all.sh pack                                    # -> results_mlp.tgz

Each `summary.json` reports the attested statistic, the plaintext statistic on the exact
committed rows, their difference (circuit precision), the full-holdout statistic, decision
mismatches caused by fixed-point rounding, prove/verify times, proof sizes and peak memory.
`HIDDEN=32,16 bash run_all.sh ...` changes the MLP size.
