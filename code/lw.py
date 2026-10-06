#!/usr/bin/env python3
"""
Exact log-weighted linear terms (post hoc; NOT part of the pre-registered tests).

Expanding Lambda = Lambda* + f around a local model Lambda* in
R_w(n) = sum_{a+b=n} Lambda(a) Lambda(b) / (ln a ln b), and keeping the term linear in f,
relative to the main term, gives:

  parity model Lambda* = 2 [m odd]:
      L_w(n)  = 2 X(n)/I(n) - 2,           X(n)   = sum_{a odd, 3<=a<=n-3} w(a)/ln(n-a)
  mod-6 model Lambda* = 3 [(m,6)=1]:
      L3_w(n) = 4 X_2(n)/I(n) - 2   (n = 1 mod 3)
                4 X_1(n)/I(n) - 2   (n = 2 mod 3)
                2 (X_1+X_2)(n)/I(n) - 2   (n = 0 mod 3),
      X_c(n) = the part of X(n) with a = c mod 3,
with w(p^k) = 1/k.  L_w and L3_w replace L(n) = 2(psi(n-1)-(n-1))/n and L3(n): they weight the
prime excess near 0 and near n by the actual factors 1/ln a and 1/ln(n-a) of the main term,
and they omit the powers of 2 (which cannot pair with anything for even n).

X_c(n) is computed exactly: in window mode, a >= A - D by FFT convolutions over odd numbers
(D = 2^26), a < A - D (so n - a > D) by exact sums at 11 Chebyshev nodes in n plus polynomial
interpolation across the window; in full mode, by FFT convolutions for all even n <= N.
Block sums use exactly the blocks of goldbach_rev.py.
Output: out/lw_window_A.npz or out/lw_full_N.npz with block sums cnt, lw, lpsi (uniform L, a
cross-check of psi), l3w, and c{0,1,2}_cnt, c{0,1,2}_l3w.
"""
import math
import sys
import time

import numpy as np
import scipy.fft as sfft

import goldbach_segmented as gs

D = 1 << 26
W_DEFAULT = 1 << 22
CH = 1 << 24


def wvals(bm, lo, hi, ppl):
    """w(m) and Lambda(m) for m in [lo, hi)."""
    w = bm.get(lo, hi).astype(np.float64)
    lam = np.zeros(hi - lo)
    idx = np.nonzero(w)[0]
    lam[idx] = np.log(idx + lo)
    for pk, p, k in ppl:
        if lo <= pk < hi:
            w[pk - lo] = 1.0 / k
            lam[pk - lo] = math.log(p)
    return w, lam


def proper_pp(limit, base):
    out = []
    for p in base:
        p = int(p)
        if p * p > limit:
            break
        pk, k = p * p, 2
        while pk <= limit:
            out.append((pk, p, k))
            pk *= p
            k += 1
    return out


def conv(u, v):
    L = sfft.next_fast_len(len(u) + len(v) - 1, real=True)
    fu = sfft.rfft(u, L, workers=2)
    fu *= sfft.rfft(v, L, workers=2)
    return sfft.irfft(fu, L, workers=2)


def bsum(n, x, lo, be, nb):
    return np.bincount((n - lo) // (2 * be), weights=x, minlength=nb)[:nb]


def finish(n, X, I, psi, lo, be, nb):
    """Per-n terms and their block sums."""
    Xt = X[0] + X[1] + X[2]
    Lw = 2 * Xt / I - 2
    c3 = n % 3
    L3w = np.where(c3 == 1, 4 * X[2] / I, np.where(c3 == 2, 4 * X[1] / I, 2 * (X[1] + X[2]) / I)) - 2
    Lpsi = 2 * (psi - (n - 1)) / n
    one = np.ones(len(n))
    out = dict(cnt=bsum(n, one, lo, be, nb), lw=bsum(n, Lw, lo, be, nb), lpsi=bsum(n, Lpsi, lo, be, nb),
               l3w=bsum(n, L3w, lo, be, nb))
    for c in range(3):
        m = (c3 == c).astype(np.float64)
        out[f"c{c}_cnt"] = bsum(n, m, lo, be, nb)
        out[f"c{c}_l3w"] = bsum(n, m * L3w, lo, be, nb)
    keep = out["cnt"] > 0
    s = np.sqrt(n)
    msg = f"mean sqrt(n) L_w = {np.mean(s * Lw):+.4f}, L = {np.mean(s * Lpsi):+.4f}, L3_w = {np.mean(s * L3w):+.4f}"
    return {k: v[keep] for k, v in out.items()}, msg


def run_window(A, W=W_DEFAULT):
    A -= A % 2
    t0 = time.time()
    Dl = min(D, A)
    bm = gs.PrimeBitmap(A + W)
    ppl = proper_pp(A + W, bm.base)
    a0 = A - Dl
    w, lam = wvals(bm, a0, A + W, ppl)
    u = w[1::2]                                  # a = a0 + 1 + 2i
    a_near = a0 + 1 + 2 * np.arange(len(u))
    jmax = (Dl + W) // 2 + 1
    b = 2 * np.arange(jmax) + 1.0
    v = np.zeros(jmax)
    v[1:] = 1.0 / np.log(b[1:])
    n = np.arange(A, A + W, 2)
    k = (n - a0 - 2) // 2
    Xn = [conv(np.where(a_near % 3 == c, u, 0.0), v)[k] for c in range(3)]
    lam_c = np.cumsum(lam)
    m = 11
    c0, h = A + W / 2, W / 2
    nodes = c0 + h * np.cos(np.pi * (np.arange(m) + 0.5) / m)
    F = np.zeros((3, m))
    psi_far = 0.0
    for lo in range(0, a0, CH):
        hi = min(lo + CH, a0)
        ww, ll = wvals(bm, lo, hi, ppl)
        psi_far += ll.sum()
        idx = np.nonzero(ww)[0]
        ai = idx + lo
        odd = ai % 2 == 1
        ai, wa = ai[odd], ww[idx][odd]
        af = ai.astype(np.float64)
        cls = ai % 3
        for j in range(m):
            t = wa / np.log(nodes[j] - af)
            F[:, j] += np.bincount(cls, weights=t, minlength=3)
    X = []
    for c in range(3):
        cheb = np.polynomial.chebyshev.Chebyshev.fit((nodes - c0) / h, F[c], m - 1)
        X.append(Xn[c] + cheb((n - c0) / h))
    Ifun = gs.spline_over(A, A + W, gs.I_exact, gs.base_I, 9)
    I = Ifun(n.astype(np.float64))
    psi = psi_far + lam_c[n - 1 - a0]
    out, msg = finish(n, X, I, psi, A, 1024, 2049)
    np.savez_compressed(f"out/lw_window_{A}.npz", **out)
    print(f"window {A:,}: {msg}, {time.time() - t0:.0f}s", flush=True)


def run_full(N):
    t0 = time.time()
    bm = gs.PrimeBitmap(N + 1)
    ppl = proper_pp(N + 1, bm.base)
    w, lam = wvals(bm, 0, N + 1, ppl)
    u = w[1::2].copy()                           # a = 1 + 2i
    del w
    psi_odd = np.cumsum(lam)[1::2].copy()        # psi(m), m = 1 + 2i
    del lam
    cls = ((1 + 2 * np.arange(len(u))) % 3).astype(np.int8)
    v = np.zeros(len(u))
    bb = 2 * np.arange(len(u)) + 1.0
    v[1:] = 1.0 / np.log(bb[1:])
    del bb
    Z = []
    for c in range(3):                           # n = 2 + 2k
        uc = np.where(cls == c, u, 0.0)
        Z.append(conv(uc, v)[: len(u) + 1].copy())
        del uc
    del u, v, cls
    Ifun = gs.spline_over(1000, N + 2, gs.I_exact, gs.base_I, 400)
    d = {}
    j, i = 10, 0
    while 2 ** j < N:
        lo, hi = 2 ** j, min(2 ** (j + 1), N + 1)
        n_even = (hi - lo) // 2
        be = 1024
        while be > 16 and n_even // be < 64:
            be //= 2
        nb = -(-n_even // be) + 1
        n = np.arange(lo, hi, 2)
        n = n[n >= 1000]
        kk = (n - 2) // 2
        out, msg = finish(n, [z[kk] for z in Z], Ifun(n.astype(np.float64)), psi_odd[kk], lo, be, nb)
        for key, val in out.items():
            d[f"r{i:02d}_{key}"] = val
        print(f"[2^{j}, 2^{j + 1}): {msg}", flush=True)
        j += 1
        i += 1
    np.savez_compressed(f"out/lw_full_{N}.npz", **d)
    print(f"full {N:,} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "full":
        run_full(int(sys.argv[2]))
    else:
        run_window(int(sys.argv[2]))
