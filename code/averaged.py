#!/usr/bin/env python3
"""
Averaged check (Section 3, Table 4).

For each N:
  Theta2(N) = sum_{p+q<=N} log p log q            (primes only)
  G2(N)     = sum_{a+b<=N} Lambda(a) Lambda(b)    (von Mangoldt weights)
  A(N) = (N^2/2 - Theta2)/N^{3/2},  B(N) = (N^2/2 - G2)/N^{3/2},  P(N) = A - B  (exact prime-power term)
  Pan(N) = 2 sum_{k>=2} sum_{p^k<=N} log p (N - p^k) / N^{3/2}   (the term in Proposition 3.1)
Explicit-formula prediction for B(N) (RH; first K zeros):
  B(N) ~ Z_K(N) + 2 log(2 pi)/sqrt(N),
  Z_K(N) = 2 sum_{|gamma|<=gamma_K} N^{rho-1/2}/(rho(rho+1)) = 4 Re sum_{0<gamma<=gamma_K} N^{i gamma}/(rho(rho+1)).
Only arrays of length pi(N) are held in memory.
Usage: python3 averaged.py NMAX zeros_file
"""
import json
import math
import sys

import numpy as np

import goldbach_segmented as gs


def main():
    NMAX = int(sys.argv[1])
    zfile = sys.argv[2] if len(sys.argv) > 2 else None
    bm = gs.PrimeBitmap(NMAX)
    P = np.concatenate([np.nonzero(bm.get(a, min(a + (1 << 24), NMAX + 1)))[0] + a
                        for a in range(0, NMAX + 1, 1 << 24)]).astype(np.int64)
    lp = np.log(P.astype(np.float64))
    Cth = np.cumsum(lp)                                   # theta at P[i]
    pp, pw = [], []                                       # proper prime powers with log p
    for p in P[P * P <= NMAX]:
        pk = int(p) * int(p)
        while pk <= NMAX:
            pp.append(pk); pw.append(math.log(int(p))); pk *= int(p)
    PP = np.concatenate([P, np.array(pp, dtype=np.int64)])
    W = np.concatenate([lp, np.array(pw)])
    o = np.argsort(PP, kind="stable"); PP, W = PP[o], W[o]
    Cps = np.cumsum(W)                                    # psi at PP[i]
    PPp, PPw = np.array(pp, dtype=np.int64), np.array(pw)

    def theta(y):
        i = np.searchsorted(P, y, side="right") - 1
        return np.where(i >= 0, Cth[np.maximum(i, 0)], 0.0)

    def psi(y):
        i = np.searchsorted(PP, y, side="right") - 1
        return np.where(i >= 0, Cps[np.maximum(i, 0)], 0.0)

    zeros = None
    if zfile:
        zeros = np.array([float(l.split()[1]) for l in open(zfile)])

    Ns = []
    M = 10 ** 5
    while M <= NMAX:
        for m in (M, 3 * M):
            if m <= NMAX:
                Ns.append(m)
        M *= 10
    rows = []
    for N in Ns:
        m = P <= N
        Th2 = float(np.sum(lp[m] * theta(N - P[m])))
        m2 = PP <= N
        G2 = float(np.sum(W[m2] * psi(N - PP[m2])))
        m3 = PPp <= N
        Pan = 2.0 * float(np.sum(PPw[m3] * (N - PPp[m3])))
        n15 = N ** 1.5
        row = dict(N=N, A=(N * N / 2 - Th2) / n15, B=(N * N / 2 - G2) / n15, P=(G2 - Th2) / n15,
                   Pan=Pan / n15, logterm=2 * math.log(2 * math.pi) / math.sqrt(N))
        if zeros is not None:
            rho = 0.5 + 1j * zeros
            Z = 4.0 * np.real(np.sum(np.exp(1j * zeros * math.log(N)) / (rho * (rho + 1))))
            row["Z"] = float(Z)
            row["K"] = int(len(zeros))
            row["B_minus_pred"] = row["B"] - row["Z"] - row["logterm"]
        rows.append(row)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    if zeros is not None:
        g = zeros[-1]
        tail_bound = (2 / math.pi) * (math.log(g / (2 * math.pi)) + 1) / g
        print(f"K = {len(zeros)} zeros, gamma_K = {g:.3f}, worst-case tail bound on |Z - Z_K| = {tail_bound:.5f}")
        s = float(np.sum(1.0 / (0.25 + zeros ** 2)))
        print(f"2*sum_{{0<gamma<=gamma_K}} 1/|rho|^2 = {2 * s:.6f}  (full sum 2+gamma-log(4pi) = "
              f"{2 + 0.5772156649015329 - math.log(4 * math.pi):.6f})")
        json.dump(dict(rows=rows, K=len(zeros), gammaK=g, tail_bound=tail_bound,
                       partial_inv_rho2=2 * s), open("out/averaged.json", "w"), indent=1)
    else:
        json.dump(dict(rows=rows), open("out/averaged_nozeros.json", "w"), indent=1)


if __name__ == "__main__":
    main()
