# FaceVault

**Protection of face biometric templates by threshold secret sharing across distributed storage.**

FaceVault is a research reference implementation that protects face-recognition templates so that no single storage location ever holds enough information to reconstruct a user's biometric. A protected template is split into *n* shares using a *k*-of-*n* threshold secret-sharing scheme and distributed across independent storage nodes. The template can only be recovered when at least *k* shares are combined; a compromise of fewer than *k* shares reveals nothing useful about the original face.

This repository reproduces every figure and table from the manuscript *"Protection of Face Biometric Templates by Threshold Secret Sharing across Distributed Storage: A Comparative Study of Two Schemes."*

## Overview

The study compares two protection configurations applied to the full protected record:

**Configuration A — NPN + CRT (ramp scheme).** A share layout based on non-positional number systems and the Chinese Remainder Theorem, providing graceful (ramp) reconstruction behaviour.

**Configuration B — Multivariate linear + verification (perfect scheme).** A perfect threshold scheme with an added verification step, so that fewer than the threshold number of shares leaks no information.

The default parameters are a 3-of-5 threshold split (`K_THRESHOLD = 3`, `N_SHARES = 5`).

## What the experiments measure

The evaluation suite reports reconstruction correctness, recognition accuracy (lossless check), bit-level information leakage, an attack suite, a baseline comparison, storage-overhead breakdown, timing, reliability, and unlinkability. All raw numbers are written to `results_full.json`.

## Repository contents

The full source lives in `facevault.zip`. After extraction:

- `schemes.py` — implementations of Configuration A (NPN + CRT, ramp) and Configuration B (multivariate linear + verification, perfect).
- `experiment_full.py` — runs the complete evaluation and writes `results_full.json`.
- `make_figures.py` — regenerates all figures into `figs/`.
- `run_lfw_benchmark.py` — runs the full pipeline on the LFW verification benchmark (requires internet access and a face encoder).
- `results_full.json` — raw numerical results.
- `figs/` — generated figures (PNG and PDF).
- `requirements.txt`, `LICENSE`, `.gitignore`.

## Requirements

Python 3 with:

- numpy 2.4.4
- scipy 1.17.1
- matplotlib 3.10.8

Optional, only for the real LFW benchmark (`run_lfw_benchmark.py`):

- scikit-learn > 1.4
- deepface > 0.0.90

## Installation

```bash
git clone https://github.com/yergeshnursultan-dev/FACEID.git
cd FACEID
unzip facevault.zip
cd facevault
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Run the full evaluation (writes results_full.json)
python experiment_full.py

# 2. Regenerate all figures into figs/
python make_figures.py

# 3. (Optional) Run the real LFW verification benchmark
python run_lfw_benchmark.py
```

## Download and review

The complete, reviewable source code is packaged in `facevault.zip` in this repository. To inspect or test it, download the archive from the repository's Code page (use the green Code button, then Download ZIP, or download `facevault.zip` directly), extract it, and follow the installation steps above.

## License

Released under the MIT License. See the `LICENSE` file inside the archive for the full text.
