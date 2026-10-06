#!/usr/bin/env python3
"""
Log-weighted prime-race terms modulo q (post hoc; exploratory), for the residue gaps of Table 3.

Linearising R_w(n) = sum_{a+b=n} Lambda(a)Lambda(b)/(ln a ln b) around the local model
Lambda*(m) = 2q/(q-1) [(m, 2q) = 1] gives, relative to the main term,

    L_{q,w}(n) = 2(q-1)/(q-2) * (X(n) - X_0(n) - X_r(n)) / I(n) - 2      if q does not divide n, r = n mod q,
                 2 (X(n) - X_0(n)) / I(n) - 2                              if q | n,

where X_c(n) = sum over odd a = c (mod q), 3 <= a <= n-3, of w(a)/ln(n-a), and X = sum_c X_c.
(For q = 3 this is L_{3,w} of lw.py.) The class-dependent part -2(q-1)/(q-2) X_r/I is the race term:
an excess of prime powers congruent to n mod q, which cannot pair, lowers R_w(n).

Window mode only. Output out/lwq_window_A.npz with block sums g{q}{p,m}_cnt and g{q}{p,m}_lq
(squares / non-squares mod q), on the blocks of goldbach_rev.py.
"""
import sys
import time

import numpy as np
import scipy.fft as sfft

import goldbach_segmented as gs
from lw import CH, D, W_DEFAULT, bsum, proper_pp, wvals

QS = (3, 5, 7, 11, 13)


def run_window(A, W=W_DEFAULT):
    A -= A % 2
    t0 = time.time()
    Dl = min(D, A)
    bm = gs.PrimeBitmap(A + W)
    ppl = proper_pp(A + W, bm.base)
    a0 = A - Dl
    w, _ = wvals(bm, a0, A + W, ppl)
    u = w[1::2].copy()
    del w
    a_near = a0 + 1 + 2 * np.arange(len(u))
    jmax = (Dl + W) // 2 + 1
    v = np.zeros(jmax)
    v[1:] = 1.0 / np.log(2 * np.arange(1, jmax) + 1.0)
    L = sfft.next_fast_len(len(u) + len(v) - 1, real=True)
    fv = sfft.rfft(v, L, workers=2)
    del v
    n = np.arange(A, A + W, 2)
    k = (n - a0 - 2) // 2
    m = 11
    c0, h = A + W / 2, W / 2
    nodes = c0 + h * np.cos(np.pi * (np.arange(m) + 0.5) / m)
    # far sums per residue class for every q (one pass over the primes)
    F = {q: np.zeros((q, m)) for q in QS}
    Ftot = np.zeros(m)
    for lo in range(0, a0, CH):
        hi = min(lo + CH, a0)
        ww, _ = wvals(bm, lo, hi, ppl)
        idx = np.nonzero(ww)[0]
        ai = idx + lo
        odd = ai % 2 == 1
        ai, wa = ai[odd], ww[idx][odd]
        af = ai.astype(np.float64)
        res = {q: (ai % q).astype(np.intp) for q in QS}
        for j in range(m):
            t = wa / np.log(nodes[j] - af)
            Ftot[j] += t.sum()
            for q in QS:
                F[q][:, j] += np.bincount(res[q], weights=t, minlength=q)

    def interp(vals):
        return np.polynomial.chebyshev.Chebyshev.fit((nodes - c0) / h, vals, m - 1)((n - c0) / h)

    def near(mask):
        fu = sfft.rfft(np.where(mask, u, 0.0), L, workers=2)
        fu *= fv
        return sfft.irfft(fu, L, workers=2)[k]

    Xtot = near(np.ones(len(u), dtype=bool)) + interp(Ftot)
    Ifun = gs.spline_over(A, A + W, gs.I_exact, gs.base_I, 9)
    I = Ifun(n.astype(np.float64))
    out = {}
    for q in QS:
        Xc = np.zeros((q, len(n)))
        for c in range(q):
            Xc[c] = near(a_near % q == c) + interp(F[q][c])
        r = n % q
        Xr = Xc[r, np.arange(len(n))]
        Lq = np.where(r != 0, 2 * (q - 1) / (q - 2) * (Xtot - Xc[0] - Xr) / I, 2 * (Xtot - Xc[0]) / I) - 2
        chi = gs.legendre(n, q)
        for s_, tag in ((1, "p"), (-1, "m")):
            msk = (chi == s_).astype(np.float64)
            out[f"g{q}{tag}_cnt"] = bsum(n, msk, A, 1024, 2049)
            out[f"g{q}{tag}_lq"] = bsum(n, msk * Lq, A, 1024, 2049)
        s = np.sqrt(n)
        gp = np.mean((s * Lq)[chi == 1]) - np.mean((s * Lq)[chi == -1])
        print(f"  q={q}: predicted (q-2) sqrt(n) gap of R_w/HLi = {(q - 2) * gp:+.4f}", flush=True)
    keep = bsum(n, np.ones(len(n)), A, 1024, 2049) > 0
    np.savez_compressed(f"out/lwq_window_{A}.npz", **{k_: v_[keep] for k_, v_ in out.items()})
    print(f"window {A:,} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run_window(int(sys.argv[1]))
