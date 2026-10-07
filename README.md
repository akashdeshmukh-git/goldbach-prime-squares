# Granville's prime-square refinement of the Hardy–Littlewood Goldbach prediction

Code and data for the paper *"On Granville's prime-square refinement of the Hardy–Littlewood prediction for Goldbach representations"* by Akash Deshmukh.

Preprint: [doi:10.5281/zenodo.23208565](https://doi.org/10.5281/zenodo.23208565)

## Contents
- `code/` – all scripts
- `data/` – block sums for every table and figure (all even n ≤ 10^8 and twelve windows of 2^21 even n between 10^7 and 10^10), results, zeta zeros, and the pre-registration file

## Requirements
Python 3.10+, NumPy, SciPy, matplotlib, mpmath.

## Reproducing
Run from the repository root with an `out/` folder (copy `data/*` into `out/` to skip the long computations):

1. `python3 code/zeros.py` – first 3000 zeta zeros
2. `bash code/run_all.sh` – counts and block sums for the full range and the 12 windows (about 2 hours on 2 cores)
3. `python3 code/lw.py full 100000000`, `python3 code/lw.py window A`, `python3 code/lwq.py A` – log-weighted linear and race terms
4. `python3 code/averaged.py 1000000000 out/zeta_zeros.txt` – averaged check under RH
5. `python3 code/random_model.py A` – random-prime baselines
6. `python3 code/analyze.py` – block bootstrap, statistics, pre-registered tests → `out/results.json`
7. `python3 code/figs.py; python3 code/make_tables.py` – figures (`out/figures/`) and tables (`out/tables.json`)

`data/preregistration.txt` was written before the eight pre-registered windows were computed; its SHA-256 is in `data/preregistration.sha256`.
