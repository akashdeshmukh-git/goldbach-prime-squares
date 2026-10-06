#!/usr/bin/env python3
"""
Revision computations for
  "On Granville's prime-square refinement of the Hardy-Littlewood prediction
   for Goldbach representations".

Builds on goldbach_segmented.py (segmented sieve, block-FFT R(n), T_k(n), S(n), eps(n),
I(n), I_T(n)). For every even n >= 1000 in a range it forms the quantities below and
accumulates their sums over BLOCKS of consecutive even n, so that every reported mean,
gap or moment can be given a block-bootstrap standard error afterwards (mode `analyze`).

Notation (C2 = 0.6601618... is the twin-prime constant)
  S(n)    = 2 C2 prod_{p | n, p > 2} (p-1)/(p-2)
  I(n)    = int_2^{n-2} dt / (log t log(n-t));        HLi = S I
  T_k(n)  = #{p prime : n - p^k prime};               T = T_2
  E(n)    = sum over a + b = n, a and b proper prime powers, of w(a) w(b), w(p^k) = 1/k
  R_w(n)  = R(n) + sum_{k>=2} (2/k) T_k(n) + E(n)     (exact; all k up to log2 n)
  eps(n)  = prod_{l odd prime, l <= 1000, l does not divide n} (1 - (n|l)/(l-2))
  I_T(n)  = int_2^{sqrt(n-3)} dt / (log t log(n - t^2));   Tpred = S eps I_T
  P_o     = HLi - Tpred               (corrected prediction for R)
  P_g     = HLi (1 - 4 eps / sqrt n)  (Granville 2007, eq. (1.1))
  L(n)    = 2 (psi(n-1) - (n-1)) / n
  L3(n)   = 4 E_2/n, 4 E_1/n, 2 (E_1 + E_2)/n  for n = 1, 2, 0 mod 3,
            E_c = psi(n-1; 3, c) - (n-1)/2
Modes
  full N         all even n in [1000, N], dyadic ranges          -> out/full_N.npz
  window A [W]   window [A, A+W), default W = 2^22               -> out/window_A.npz
"""
import argparse
import math
import os
import time

import numpy as np

import goldbach_segmented as gs

W_DEFAULT = 1 << 22
QS = (3, 5, 7, 11, 13)
HBINS = np.linspace(-6.0, 6.0, 241)
OUT = "out"


# ----------------------------------------------------------------- prime powers, psi
def proper_prime_powers(limit, base):
    """Sorted list of (p^k, p, k) with k >= 2 and p^k <= limit."""
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
    out.sort()
    return out


def theta_classes(bm, lo, hi, chunk=1 << 24):
    """(theta, theta_1, theta_2) over primes in [lo, hi): sum of log p, all and by p mod 3."""
    tot = np.zeros(3)
    a = lo
    while a < hi:
        b = min(a + chunk, hi)
        idx = np.nonzero(bm.get(a, b))[0] + a
        if len(idx):
            lg, c = np.log(idx), idx % 3
            tot += (lg.sum(), lg[c == 1].sum(), lg[c == 2].sum())
        a = b
    return tot


def psi_before(bm, A, ppl):
    """(psi(A-1), psi(A-1;3,1), psi(A-1;3,2))."""
    v = theta_classes(bm, 0, A)
    for pk, p, k in ppl:
        if pk >= A:
            break
        lp, c = math.log(p), pk % 3
        v[0] += lp
        if c:
            v[c] += lp
    return v


def both_proper(A, W, ppl):
    """E(n) for n in [A, A+W): w-weighted ordered pairs of proper prime powers."""
    E = np.zeros(W)
    vals = np.array([x[0] for x in ppl], dtype=np.int64)
    wts = np.array([1.0 / x[2] for x in ppl])
    for i, a in enumerate(vals):
        if a >= A + W:
            break
        j0 = np.searchsorted(vals, A - a)
        j1 = np.searchsorted(vals, A + W - a)
        if j1 > j0:
            np.add.at(E, vals[j0:j1] + a - A, wts[i] * wts[j0:j1])
    return E


# ----------------------------------------------------------------- per-n quantities
def window_quantities(bm, A, W, Ifun, ITfun, psi0, ppl, nmin=1000):
    """Per-n arrays for even n >= nmin in [A, A+W); also returns psi at A+W-1 (carry)."""
    t0 = time.time()
    R = gs.window_R(bm, A, W).astype(np.float64)
    T = gs.window_Tk(bm, A, W, 2).astype(np.float64)
    kmax = max(4, int(math.log2(A + W)))
    H = np.zeros(W)
    H5 = np.zeros(W)           # first draft: k = 3, 4, 5 only, no E(n)
    Tev = np.zeros(W)          # sum over even k >= 4 of (2/k) T_k: p = r^j counted with weight 1/j
    T4 = None
    for k in range(3, kmax + 1):
        tk = gs.window_Tk(bm, A, W, k)
        H += (2.0 / k) * tk
        if k % 2 == 0:
            Tev += (2.0 / k) * tk
        if k <= 5:
            H5 += (2.0 / k) * tk
        if k == 4:
            T4 = tk.astype(np.float64)
    Ew = both_proper(A, W, ppl)
    S = gs.window_S(bm, A, W)
    eps = gs.window_eps(A, W)
    # von Mangoldt by class mod 3 on the window, for psi(n-1)
    lam = np.zeros((3, W))
    idx = np.nonzero(bm.get(A, A + W))[0]
    lg, c = np.log(idx + A), (idx + A) % 3
    lam[0, idx] = lg
    lam[1, idx[c == 1]] = lg[c == 1]
    lam[2, idx[c == 2]] = lg[c == 2]
    for pk, p, k in ppl:
        if pk >= A + W:
            break
        if pk >= A:
            lp, cc = math.log(p), pk % 3
            lam[0, pk - A] += lp
            if cc:
                lam[cc, pk - A] += lp
    cum = np.cumsum(lam, axis=1)
    psi_prev = psi0[:, None] + np.concatenate([np.zeros((3, 1)), cum[:, :-1]], axis=1)
    carry = psi0 + cum[:, -1]
    n = np.arange(A, A + W, dtype=np.int64)
    keep = (n % 2 == 0) & (n >= nmin)
    q = dict(n=n[keep], R=R[keep], T=T[keep], T4=T4[keep], Tev=Tev[keep], H5=H5[keep],
             H=H[keep] + Ew[keep], S=S[keep], eps=eps[keep],
             psi=psi_prev[0, keep], psi1=psi_prev[1, keep], psi2=psi_prev[2, keep])
    nf = q["n"].astype(np.float64)
    q["I"] = Ifun(nf)
    q["IT"] = ITfun(nf)
    q["secs"] = time.time() - t0
    return q, carry


STAT_NAMES = []


def stat_columns(q):
    """Yield (name, per-n column) for every summed statistic, one column at a time."""
    n = q["n"]
    nf = n.astype(np.float64)
    R, T, H, S, eps = q["R"], q["T"], q["H"], q["S"], q["eps"]
    HLi = S * q["I"]
    Tpred = S * eps * q["IT"]
    Po = HLi - Tpred
    Pg = HLi * (1.0 - 4.0 * eps / np.sqrt(nf))
    Rw = R + T + H
    m1 = nf - 1.0
    E1 = q["psi1"] - m1 / 2
    E2 = q["psi2"] - m1 / 2
    L = 2.0 * (q["psi"] - m1) / nf
    c3 = n % 3
    L3 = np.where(c3 == 1, 4.0 * E2 / nf, np.where(c3 == 2, 4.0 * E1 / nf, 2.0 * (E1 + E2) / nf))
    del E1, E2
    base = dict(ri=R / HLi, rc=(R + T) / HLi, rw=Rw / HLi, tp=Tpred / HLi, L=L, L3=L3,
                ro=R / Po - 1.0, rg=R / Pg - 1.0)
    base["yres"] = base["rw"] - 1.0 - L3
    zo = (R - Po) / np.sqrt(Po)
    zg = (R - Pg) / np.sqrt(Pg)
    zr = (Rw - HLi * (1.0 + L3)) / np.sqrt(HLi)
    yield "cnt", np.ones_like(nf)
    for k in ("ri", "rc", "rw", "tp", "L", "L3", "ro", "rg", "yres"):
        yield k, base[k]
    yield "t", T / HLi
    yield "h", H / HLi
    yield "LoverS", L / S
    yield "Traw", T
    yield "Tpraw", Tpred
    yield "T4h", T + 0.5 * q["T4"]
    yield "T4raw", q["T4"]
    yield "Tpw", T + q["Tev"]
    yield "h5", q["H5"] / HLi
    yield "eps", eps
    zc = (R + T - HLi) / np.sqrt(HLi)
    yield "zc", zc
    yield "zc2", zc ** 2
    del zc
    for nm, z in (("zo", zo), ("zg", zg), ("zr", zr)):
        yield nm, z
        yield nm + "2", z ** 2
        if nm != "zr":
            yield nm + "3", z ** 3
            yield nm + "4", z ** 4
    Ares = Rw - HLi * (1.0 + L3)
    yield "Ares", Ares
    AresI = Ares / q["I"]
    yield "AresI", AresI
    yield "sqrtn", np.sqrt(nf)
    for qq in QS:
        chi = gs.legendre(n, qq)
        for s_, tag in ((1, "p"), (-1, "m")):
            msk = (chi == s_).astype(np.float64)
            yield f"g{qq}{tag}_cnt", msk
            for k in ("ri", "rc", "rw", "ro", "rg", "tp"):
                yield f"g{qq}{tag}_{k}", msk * base[k]
    zs = dict(zo=zo, zg=zg, zr=zr)
    for cc in (0, 1, 2):
        msk = (c3 == cc).astype(np.float64)
        yield f"c{cc}_cnt", msk
        for k in ("rw", "L3", "L", "yres"):
            yield f"c{cc}_{k}", msk * base[k]
        for k, z in zs.items():
            yield f"c{cc}_{k}", msk * z
    d3, d5 = (n % 3 == 0), (n % 5 == 0)
    for a in (0, 1):
        for b in (0, 1):
            msk = ((d3 == bool(a)) & (d5 == bool(b))).astype(np.float64)
            yield f"s{a}{b}_cnt", msk
            yield f"s{a}{b}_yres", msk * base["yres"]
            yield f"s{a}{b}_AresI", msk * AresI
            yield f"s{a}{b}_S", msk * S
    yield "__hist__", dict(zo=zo, zg=zg, zr=zr)


def accumulate(q, accs):
    """Add the statistics of the per-n arrays q to every RangeAcc in accs (masks by range)."""
    masks = []
    for acc in accs:
        m = (q["n"] >= acc.lo) & (q["n"] < acc.hi)
        if m.any():
            bid = ((q["n"][m] - acc.lo) // (2 * acc.block_even)).astype(np.int64)
            masks.append((acc, m, bid))
    for name, col in stat_columns(q):
        if name == "__hist__":
            for acc, m, _ in masks:
                for k, z in col.items():
                    acc.hists[k] = acc.hists.get(k, 0) + np.histogram(z[m], HBINS)[0]
            continue
        if name not in STAT_NAMES:
            STAT_NAMES.append(name)
        for acc, m, bid in masks:
            acc.add_col(name, bid, col[m])


class RangeAcc:
    """Block sums for one range [lo, hi) of even n."""

    def __init__(self, label, lo, hi, block_even=None):
        self.label, self.lo, self.hi = label, lo, hi
        n_even = (hi - lo) // 2
        if block_even is None:
            block_even = 1024
            while block_even > 16 and n_even // block_even < 64:
                block_even //= 2
        self.block_even = block_even
        self.nblocks = -(-n_even // block_even) + 1
        self.cols = {}
        self.hists = {}
        self.extra = {}

    def add_col(self, name, bid, col):
        if name not in self.cols:
            self.cols[name] = np.zeros(self.nblocks)
        self.cols[name] += np.bincount(bid, weights=col, minlength=self.nblocks)[: self.nblocks]

    def save_dict(self, prefix):
        sums = np.column_stack([self.cols[k] for k in STAT_NAMES])
        nz = sums[:, STAT_NAMES.index("cnt")] > 0
        d = {f"{prefix}sums": sums[nz],
             f"{prefix}meta": np.array([self.lo, self.hi, self.block_even], dtype=np.int64),
             f"{prefix}label": np.array(self.label)}
        for k, v in self.hists.items():
            d[f"{prefix}hist_{k}"] = v
        for k, v in self.extra.items():
            d[f"{prefix}x_{k}"] = np.array(v)
        return d


def run_window(A, W):
    A -= A % 2
    t0 = time.time()
    bm = gs.PrimeBitmap(A + W)
    ppl = proper_prime_powers(A + W, bm.base)
    psi0 = psi_before(bm, A, ppl)
    Ifun = gs.spline_over(A, A + W, gs.I_exact, gs.base_I, 9)
    ITfun = gs.spline_over(A, A + W, gs.IT_exact, gs.base_IT, 9)
    print(f"[{A:,}, {A + W:,}): sieve+psi {time.time() - t0:.1f}s, psi(A-1)-(A-1) = {psi0[0] - (A - 1):.1f}",
          flush=True)
    q, _ = window_quantities(bm, A, W, Ifun, ITfun, psi0, ppl)
    # brute-force spot checks of R and T
    rng = np.random.default_rng(A % 1000003)
    for i in rng.choice(len(q["n"]), 3, replace=False):
        nn = int(q["n"][i])
        assert gs.brute_R(bm, nn) == q["R"][i], ("R mismatch", nn)
        assert gs.brute_T(bm, nn) == q["T"][i], ("T mismatch", nn)
    acc = RangeAcc(f"window {A}", A, A + W, block_even=1024)
    accumulate(q, [acc])
    acc.extra = dict(E_start=psi0[0] - (A - 1), secs=time.time() - t0, W=W)
    d = acc.save_dict("w_")
    d["stat_names"] = np.array(STAT_NAMES)
    np.savez_compressed(f"{OUT}/window_{A}.npz", **d)
    print(f"window {A:,} done in {time.time() - t0:.1f}s (quantities {q['secs']:.1f}s)", flush=True)


def run_full(N, W=W_DEFAULT):
    t0 = time.time()
    bm = gs.PrimeBitmap(N + 1)
    ppl = proper_prime_powers(N + 1, bm.base)
    Ifun = gs.spline_over(1000, N + 2, gs.I_exact, gs.base_I, 400)
    ITfun = gs.spline_over(1000, N + 2, gs.IT_exact, gs.base_IT, 200)
    accs, j = [], 10
    while 2 ** j < N:
        accs.append(RangeAcc(f"[2^{j}, 2^{j + 1})", 2 ** j, min(2 ** (j + 1), N + 1)))
        j += 1
    carry = np.zeros(3)
    for A in range(0, N + 1, W):
        Wa = min(W, N + 1 - A)
        q, carry = window_quantities(bm, A, Wa, Ifun, ITfun, carry, ppl)
        secs = q["secs"]
        if len(q["n"]):
            accumulate(q, accs)
        del q
        print(f"  [{A:,}, {A + Wa:,}) {secs:.1f}s", flush=True)
    d = {"stat_names": np.array(STAT_NAMES)}
    for i, acc in enumerate(accs):
        if acc.cols:
            d.update(acc.save_dict(f"r{i:02d}_"))
    np.savez_compressed(f"{OUT}/full_{N}.npz", **d)
    print(f"full {N:,} done in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "window"])
    ap.add_argument("A", type=int)
    ap.add_argument("W", nargs="?", type=int, default=W_DEFAULT)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "full":
        run_full(a.A)
    else:
        run_window(a.A, a.W)


if __name__ == "__main__":
    main()
