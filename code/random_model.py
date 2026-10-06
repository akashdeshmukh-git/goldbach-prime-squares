#!/usr/bin/env python3
"""
Random-primes baselines for the spread of z = (R - E)/sqrt(E) inside a window (Section 5.6).

Cramer model:          X_m ~ Bernoulli(min(1, 2/log m)) independently for odd m >= 3.
Cramer-Granville(z):   X_m = 0 if m has a prime factor <= z, else Bernoulli(min(1, c_z/log m)),
                       c_z = prod_{p<=z} (1-1/p)^(-1)   (same mean density).
R(n) = sum_{a+b=n} X_a X_b (ordered); E(n) = sum p_a p_b its exact expectation.
For each realisation: within-window spread = sd over even n in [A, A+W) of (R-E)/sqrt(E),
after removing the window mean (as for the real data). Also the analytic value
sqrt(2 sum p_a p_b (1-p_a)(1-p_b) / sum p_a p_b) for the quadratic part.
"""
import json
import math
import sys

import numpy as np

import goldbach_segmented as gs


def run(model, z, A, W, reps, seed):
    N = A + W
    m = np.arange(N + 1, dtype=np.float64)
    p = np.zeros(N + 1)
    odd = (np.arange(N + 1) % 2 == 1) & (np.arange(N + 1) >= 3)
    if model == "cramer":
        p[odd] = np.minimum(1.0, 2.0 / np.log(m[odd]))
    else:
        small = gs.simple_sieve(z)
        cz = float(np.prod([q / (q - 1.0) for q in small]))
        ok = np.ones(N + 1, dtype=bool)
        for q in small:
            ok[::q] = False
        ok[:2] = False
        p[ok] = np.minimum(1.0, cz / np.log(m[ok]))
    L = 1 << (2 * N + 1).bit_length()
    E = np.fft.irfft(np.fft.rfft(p, L) ** 2, L)[: N + 1]
    q2 = p * (1 - p)
    V = np.fft.irfft(np.fft.rfft(q2, L) ** 2, L)[: N + 1]     # sum p(1-p) p(1-p)
    idx = np.arange(A + (A % 2), A + W, 2)
    analytic = math.sqrt(2 * V[idx].sum() / E[idx].sum())
    rng = np.random.default_rng(seed)
    sds, means = [], []
    for _ in range(reps):
        X = (rng.random(N + 1) < p).astype(np.float64)
        R = np.rint(np.fft.irfft(np.fft.rfft(X, L) ** 2, L)[: N + 1])
        zz = (R[idx] - E[idx]) / np.sqrt(E[idx])
        sds.append(float(zz.std()))
        means.append(float(zz.mean()))
    return dict(model=model, z=z, A=A, W=W, analytic_quadratic=analytic,
                within_sd=sds, window_means=means)


def main():
    A = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    W = 1 << 22
    res = [run("cramer", 0, A, W, 4, 1),
           run("cg", 30, A, W, 4, 2),
           run("cg", 100, A, W, 4, 3)]
    for r in res:
        print(r["model"], r["z"], "analytic", round(r["analytic_quadratic"], 3),
              "within-window sd", [round(s, 3) for s in r["within_sd"]],
              "window means", [round(s, 2) for s in r["window_means"]], flush=True)
    json.dump(res, open(f"out/random_model_{A}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
