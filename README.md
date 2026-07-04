# ZK-SR117: Chunked Zero-Knowledge Attestation for SR 11-7 Supervisory Controls

Companion code for the paper *"ZK-SR117: A Chunked Zero-Knowledge Attestation
Design for Aggregated Fair-Lending Metrics, with a Control Mapping toward Full
SR 11-7 Coverage."*

This repository implements and evaluates a chunked zero-knowledge attestation
mechanism that lets a bank prove statistical properties of a credit model —
fair-lending demographic parity and expected calibration error — to a
regulator, without disclosing model weights or customer data. Both attested
controls are demonstrated end-to-end on real 2022 HMDA mortgage-application
data.

## What's in this repository

| Path | Description |
|---|---|
| `01_generate_data.py` | HMDA schema reference / synthetic data stand-in for initial testing |
| `02_train_real_v2.py` | Trains the logistic regression credit model (Model A) on real HMDA data |
| `03_build_onnx_circuit.py` | Flat-summation circuit (naive design — overflows past ~N=1,024–2,048) |
| `03_build_onnx_circuit_tree.py` | Tree-reduction circuit (numerically exact, but hit a compile-time wall in EZKL) |
| `06_setup_once_clipped.py` | One-time setup for the chunked demographic-parity attestation, including the Table 1 Row 9 preprocessing spec |
| `06_prove_chunk.py` | Per-chunk worker (run as a subprocess by the driver, for OS-level timeout safety) |
| `06_driver.py` | Orchestrates the K=32 chunk sweep for demographic parity, resumable, with per-chunk timeout |
| `07_bootstrap_ci_clipped.py` | Bootstrap 95% CI for the demographic-parity-gap estimator |
| `11_flatsum_real_sweep.py` | Real-HMDA confirmation of the flat-sum circuit's breakdown pattern (N=2048/4096/8192) |
| `12_version_binding.py` | Model-version-binding cross-verification test (demographic parity) |
| `13_ece_setup_once.py` | One-time setup for the chunked ECE (calibration) attestation |
| `13_ece_prove_chunk.py` | Per-chunk ECE worker |
| `13_ece_driver.py` | Orchestrates the K=32 chunk sweep for ECE |
| `14_ece_bootstrap_ci.py` | Bootstrap 95% CI for the ECE estimator |
| `15_ece_stratified_check.py` | Diagnostic: plaintext ECE on the exact stratified committed sample vs. the full holdout |
| `16_ece_version_binding.py` | Model-version-binding cross-verification test (ECE) |

## Data

This repository does **not** redistribute HMDA microdata, consistent with
standard practice for regulatory disclosure datasets. To reproduce:

1. Download the 2022 HMDA Loan Application Register from the public
   [FFIEC/CFPB HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/).
2. Filter to: conventional, first-lien, home-purchase, principal-residence
   originations and denials only (`action_taken` in `{1, 3}`).
3. Save the filtered extract as `hmda_sample.csv` in this repository's root,
   with at minimum these columns: `income`, `loan_amount`,
   `debt_to_income_ratio`, `loan_to_value_ratio`, `derived_race`,
   `action_taken_binary`.

The exact filtering criteria and the preprocessing-specification percentile
clip bounds (Table 1, Row 9 of the paper) are hard-coded as constants at the
top of `02_train_real_v2.py` and `06_setup_once_clipped.py`
(`CLIP_LOW_PCTILE = 0.5`, `CLIP_HIGH_PCTILE = 99.5`).

## Dependencies

```bash
pip install ezkl onnx onnxruntime pandas scikit-learn numpy --break-system-packages
```

Tested with:
- `ezkl` 23.0.5
- Python 3.12
- `onnx` / `onnxruntime` (current PyPI releases as of mid-2026)

## Hardware notes

Developed and tested on:
- Apple Silicon Mac (per-chunk prove times reported in the paper: ~3.7–4.0s
  for demographic parity, ~14.4–15.1s for ECE, at n=1,024 per chunk)
- A cloud sandbox environment (used for circuit-correctness verification
  against synthetic data; could not reach EZKL's production trusted-setup
  ceremony host due to network restrictions — see note below)

**Trusted setup:** production runs use `ezkl.get_srs()`, which downloads the
real Powers-of-Tau ceremony file. This requires outbound network access to
EZKL's ceremony host. If your environment blocks this, `ezkl.gen_srs()`
generates a local test-only SRS as a substitute — **do not use this for
anything beyond local testing**, as noted explicitly in the code comments.

## Reproducing the paper's results

**Demographic-parity attestation (Table 3):**
```bash
python3 06_setup_once_clipped.py
python3 06_driver.py
python3 07_bootstrap_ci_clipped.py
```

**ECE attestation (Table 5):**
```bash
python3 13_ece_setup_once.py
python3 13_ece_driver.py
python3 14_ece_bootstrap_ci.py
python3 15_ece_stratified_check.py
```

**Supporting checks (Sections 7.2, 7.4, and version-binding):**
```bash
python3 11_flatsum_real_sweep.py       # real-data flat-sum confirmation
python3 12_version_binding.py          # demographic-parity model-version binding
python3 16_ece_version_binding.py      # ECE model-version binding
```

Each driver script (`06_driver.py`, `13_ece_driver.py`) is resumable: if
interrupted, rerunning skips chunks that already completed and only retries
what's missing.

## Known limitations (see paper Section 8 for full discussion)

- Tree-reduction circuit compile time was characterized as an EZKL-specific
  finding, not tested against other proving backends (Halo2 directly, Jolt,
  etc.) — see `03_build_onnx_circuit_tree.py`.
- The nonce protocol used throughout (`SR117-2026Q3-EXAMINER-NONCE-001`) is a
  fixed, published string standing in for the full beacon-derived
  construction specified in Section 5.7 of the paper.
- Bootstrap confidence intervals use uniform resampling; the paper's Section
  7.5.3 identifies that a stratified-resampling recomputation would be a more
  precisely matched comparison to the attestation's actual nonce-controlled
  sampling design (Section 9, future work item 6).

## Citation

If you use this code, please cite:

```
[Author]. "ZK-SR117: A Chunked Zero-Knowledge Attestation Design for
Aggregated Fair-Lending Metrics, with a Control Mapping toward Full SR 11-7
Coverage." [Venue], 2026.
```

## License

MIT
