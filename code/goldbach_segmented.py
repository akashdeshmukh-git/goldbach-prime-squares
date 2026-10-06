#!/usr/bin/env python3
"""
Segmented computation of Goldbach counts R(n), prime-square counts T(n), and
Hardy-Littlewood predictions, for all even n up to N or for a window of even n
near a large A.

Definitions
-----------
R(n)   = #{(p, q) ordered : p, q prime, p + q = n}
T(n)   = #{p prime : n - p^2 is prime}
T_k(n) = #{p prime : n - p^k is prime}            (k = 3, 4, 5 also computed)
S(n)   = 2*C2 * prod_{l | n, l > 2} (l-1)/(l-2)   (Goldbach singular series)
I(n)   = int_2^{n-2} dt / (ln t ln(n-t))
HLi(n) = S(n) * I(n)

Heuristic prediction for T(n) (see the accompanying note):
eps(n)   = prod_{l odd prime, l does not divide n} (1 - (n|l)/(l-2))   (truncated at l <= EPS_LMAX)
I_T(n)   = int_2^{sqrt(n-3)} dt / (ln t ln(n - t^2))
Tpred(n) = S(n) * eps(n) * I_T(n)

Method (no full-range FFT, memory ~ N/8 bytes)
----------------------------------------------
1. Segmented Sieve of Eratosthenes, stored as a packed bitmap (1 bit per integer).
2. For a window of n in [A, A+W): every pair p + q = n has min(p, q) < A/2 or both
   p, q >= A/2. The first kind is handled block by block: for primes p in a block
   [a, a+B) with a+B <= A/2, the counts for all n in the window are one convolution
   of the block's prime indicator with the indicator of q in (A-a-B, A+W-a].
   Each convolution is a small FFT of length ~2B+W, so memory stays O(B+W) and
   total work is O(A log B). The second kind is one self-convolution of length ~W.
   Every FFT output is rounded to an integer; values are far below 2^53.
3. T_k(n) for the window: for each prime p with p^k < A+W, add the indicator of
   [A - p^k, A + W - p^k).
4. S(n) by sieving the window with prime powers; I(n) and I_T(n) by high-accuracy
   quadrature on a grid plus cubic-spline interpolation.

Usage
-----
    pip install numpy scipy
    python3 goldbach_segmented.py full 100000000          # all even n <= 1e8
    python3 goldbach_segmented.py window 1000000000 4194304   # 2^21 even n above 1e9
    python3 goldbach_segmented.py check                   # brute-force self-test
    python3 goldbach_segmented.py average 30000000        # averaged (RH-provable) check
    python3 goldbach_segmented.py zeta goldbach_window_1000000000.json   # adds the 2(psi(n)-n)/n term
    python3 goldbach_segmented.py granville 100000000 4194304   # Granville's I* (2007, eq. 1.1) vs ours

Each run prints a table and writes a JSON file of per-range statistics.
Rough cost on a 2-core laptop: full 1e8 ~ 10 min; window at 1e9 ~ 5 min;
window at 1e10 ~ 40 min and ~2 GB RAM.

This is computational evidence only; nothing here proves anything.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy import fft as sfft
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

C2 = 0.6601618158468695739278121100145   # twin-prime constant
QS = (3, 5, 7, 11, 13)                     # moduli for the residue-class bias
EPS_LMAX = 1000                            # truncation of the eps(n) product
WORKERS = os.cpu_count() or 1


# ----------------------------------------------------------------- sieving
def simple_sieve(n):
    """Primes <= n (small n only)."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


class PrimeBitmap:
    """Packed bitmap of primes in [0, N], built by a segmented sieve."""

    def __init__(self, N, chunk=1 << 24):
        self.N = N
        self.base = simple_sieve(math.isqrt(N) + 1)
        self.packed = np.zeros(N // 8 + 1, dtype=np.uint8)
        for lo in range(0, N + 1, chunk):              # lo is a multiple of 8
            hi = min(lo + chunk, N + 1)
            seg = np.ones(hi - lo, dtype=bool)
            if lo == 0:
                seg[:min(2, hi)] = False
            for p in self.base:
                pp = int(p) * int(p)
                if pp >= hi:
                    break
                start = max(pp, -(-lo // p) * p)
                seg[start - lo::p] = False
            bits = np.packbits(seg, bitorder="little")
            self.packed[lo // 8: lo // 8 + len(bits)] = bits

    def get(self, lo, hi):
        """Boolean array is_prime[m] for m in [lo, hi); zero outside [0, N]."""
        out = np.zeros(max(hi - lo, 0), dtype=bool)
        a, b = max(lo, 0), min(hi, self.N + 1)
        if a < b:
            b0, b1 = a // 8, (b + 7) // 8
            bits = np.unpackbits(self.packed[b0:b1], bitorder="little")
            out[a - lo: b - lo] = bits[a - 8 * b0: b - 8 * b0].astype(bool)
        return out


# ----------------------------------------------------------------- R(n) on a window
def _conv(x, y):
    L = sfft.next_fast_len(len(x) + len(y) - 1, real=True)
    fx = sfft.rfft(x, L, workers=WORKERS)
    fy = fx if y is x else sfft.rfft(y, L, workers=WORKERS)
    return sfft.irfft(fx * fy, L, workers=WORKERS)


def window_R(bm, A, W, B=1 << 22):
    """Ordered R(n) for every n in [A, A+W)."""
    L1 = A // 2
    acc = np.zeros(W, dtype=np.int64)
    a = 0
    while a < L1:
        b = min(a + B, L1)
        x = bm.get(a, b)
        if x.any():
            qlo, qhi = A - b + 1, A + W - a
            z = _conv(x.astype(np.float64), bm.get(qlo, qhi).astype(np.float64))
            k0 = A - a - qlo
            part = z[k0:k0 + W]
            r = np.rint(part)
            assert np.max(np.abs(part - r)) < 0.1, "FFT rounding error"
            acc += r.astype(np.int64)
        a = b
    m = bm.get(L1, A + W - L1).astype(np.float64)      # both p, q >= L1
    z = _conv(m, m)
    k0 = A - 2 * L1
    mid = np.rint(z[k0:k0 + W]).astype(np.int64)
    return 2 * acc + mid


def window_Tk(bm, A, W, k):
    """T_k(n) = #{p prime : n - p^k prime} for n in [A, A+W)."""
    T = np.zeros(W, dtype=np.int32)
    for p in bm.base:
        pk = int(p) ** k
        if pk >= A + W:
            break
        T += bm.get(A - pk, A + W - pk)
    return T


# ----------------------------------------------------------------- arithmetic factors
def window_S(bm, A, W):
    """Singular series S(n) for n in [A, A+W) (meaningful for even n)."""
    rem = np.arange(A, A + W, dtype=np.int64)
    S = np.full(W, 2 * C2)
    top = A + W - 1
    for p in bm.base:
        p = int(p)
        if p * p > top:
            break
        st = (-A) % p
        if st >= W:
            continue
        if p > 2:
            S[st::p] *= (p - 1) / (p - 2)
        pk = p
        while pk <= top:
            s2 = (-A) % pk
            if s2 < W:
                rem[s2::pk] //= p
            pk *= p
    big = rem > 1                                   # one prime factor > sqrt(top)
    r = rem[big].astype(np.float64)
    S[big] *= (r - 1) / (r - 2)
    return S


def window_eps(A, W, lmax=EPS_LMAX):
    """eps(n) = prod over odd primes l <= lmax, l not dividing n, of 1 - (n|l)/(l-2)."""
    eps = np.ones(W)
    idx = np.arange(W, dtype=np.int64)
    for l in simple_sieve(lmax)[1:]:
        l = int(l)
        chi = np.zeros(l)
        for x in range(1, l):
            chi[x * x % l] = 1.0
        chi[1:][chi[1:] == 0] = -1.0                 # non-residues
        chi[0] = 0.0
        r = (A % l + idx) % l
        eps *= 1.0 - chi[r] / (l - 2)
    return eps


def legendre(n, q):
    r = n % q
    out = np.zeros(len(n), dtype=np.int8)
    sq = np.zeros(q, dtype=bool)
    sq[[x * x % q for x in range(1, q)]] = True
    out[(r != 0) & sq[r]] = 1
    out[(r != 0) & ~sq[r]] = -1
    return out


# ----------------------------------------------------------------- integrals
def I_exact(n):
    """2 * int_2^{n/2} dt/(ln t ln(n-t)), via t = e^u, piecewise quadrature."""
    f = lambda u: math.exp(u) / (u * math.log(n - math.exp(u)))
    pts = np.linspace(math.log(2), math.log(n / 2), 60)
    return 2 * sum(quad(f, a, b, epsabs=0, epsrel=1e-13, limit=200)[0]
                   for a, b in zip(pts[:-1], pts[1:]))


def IT_exact(n):
    """int_2^{sqrt(n-3)} dt/(ln t ln(n - t^2)), split finely near the upper end."""
    top = math.sqrt(n - 3)
    f = lambda t: 1.0 / (math.log(t) * math.log(n - t * t))
    cuts = [2.0, 3.0] + list(np.geomspace(3.0, top / 2, 30)[1:])
    j = 1
    while n - 3 * 10 ** j > (top / 2) ** 2 and 10 ** j < n:
        j += 1
    cuts += sorted(math.sqrt(n - 3 * 10 ** i) for i in range(j - 1, 0, -1)) + [top]
    cuts = sorted(set(c for c in cuts if 2 <= c <= top))
    return sum(quad(f, a, b, epsabs=0, epsrel=1e-10, limit=200)[0]
               for a, b in zip(cuts[:-1], cuts[1:]))


def spline_over(lo, hi, func, base, npts):
    """Cubic spline in ln n of func(n)/base(n) on [lo, hi]."""
    xs = np.linspace(math.log(lo), math.log(hi), npts)
    ys = [func(math.exp(x)) / base(math.exp(x)) for x in xs]
    cs = CubicSpline(xs, ys)
    return lambda n: cs(np.log(n)) * base(n)


base_I = lambda n: n / np.log(n) ** 2
base_IT = lambda n: 2 * np.sqrt(n) / np.log(n) ** 2


# ----------------------------------------------------------------- statistics
class Acc:
    """Running sums for one range of n."""

    def __init__(self, label, lo, hi):
        self.label, self.lo, self.hi = label, lo, hi
        self.s = {}

    def add(self, key, v):
        self.s[key] = self.s.get(key, 0.0) + float(v)

    def mean(self, key, cnt="count"):
        return self.s.get(key, 0.0) / max(self.s.get(cnt, 0.0), 1.0)

    def update(self, n, R, T, Tpp, HLi, Tpred, D):
        self.add("count", len(n))
        ri, rc, rpp = R / HLi, (R + T) / HLi, (R + Tpp) / HLi
        z = (R + T - HLi) / np.sqrt(HLi)
        for k, v in (("ri", ri), ("rc", rc), ("rpp", rpp), ("t", T / HLi),
                     ("tp", Tpred / HLi), ("z", z), ("z2", z * z),
                     ("D", D), ("T", T), ("Tpp", Tpp), ("Tpred", Tpred)):
            self.add(k, v.sum())
        for q in QS:
            chi = legendre(n, q)
            for c, name in ((1, "qr"), (-1, "nr")):
                m = chi == c
                self.add(f"{q}{name}_n", m.sum())
                self.add(f"{q}{name}_ri", ri[m].sum())
                self.add(f"{q}{name}_rc", rc[m].sum())
                self.add(f"{q}{name}_tp", (Tpred / HLi)[m].sum())

    def summary(self):
        out = dict(label=self.label, lo=self.lo, hi=self.hi, count=int(self.s["count"]))
        for k in ("ri", "rc", "rpp", "t", "tp", "z"):
            out[k] = self.mean(k)
        out["z_std"] = math.sqrt(max(self.mean("z2") - self.mean("z") ** 2, 0))
        out["D_over_T"] = self.s["D"] / max(self.s["T"], 1)
        out["D_over_Tpp"] = self.s["D"] / max(self.s["Tpp"], 1)
        out["T_over_Tpred"] = self.s["T"] / max(self.s["Tpred"], 1e-9)
        for q in QS:
            g = lambda key: (self.mean(f"{q}qr_{key}", f"{q}qr_n")
                             - self.mean(f"{q}nr_{key}", f"{q}nr_n"))
            out[f"gap{q}_ri"] = g("ri")
            out[f"gap{q}_rc"] = g("rc")
            out[f"gap{q}_pred"] = -g("tp")     # model: R/HLi ~ 1 - Tpred/HLi
        return out


def process_window(bm, A, W, accs, Ifun, ITfun, nmin=1000):
    t0 = time.time()
    R = window_R(bm, A, W)
    T = window_Tk(bm, A, W, 2)
    Tpp = T + sum((2 / k) * window_Tk(bm, A, W, k) for k in (3, 4, 5))
    S = window_S(bm, A, W)
    eps = window_eps(A, W)
    n = np.arange(A, A + W, dtype=np.int64)
    keep = (n % 2 == 0) & (n >= nmin)
    n, R, T, Tpp, S, eps = n[keep], R[keep], T[keep], Tpp[keep], S[keep], eps[keep]
    nf = n.astype(np.float64)
    HLi = S * Ifun(nf)
    Tpred = S * eps * ITfun(nf)
    D = HLi - R
    for acc in accs:
        m = (n >= acc.lo) & (n < acc.hi)
        if m.any():
            acc.update(n[m], R[m], T[m], Tpp[m], HLi[m], Tpred[m], D[m])
    return time.time() - t0, n, R, T


# ----------------------------------------------------------------- self-test
def brute_R(bm, n):
    c = 0
    half = n // 2
    for lo in range(0, half, 1 << 24):
        hi = min(lo + (1 << 24), half)              # p in [lo, hi), p < n/2
        x = bm.get(lo, hi)
        y = bm.get(n - hi + 1, n - lo + 1)[::-1]     # y[i] = is_prime(n - lo - i)
        c += int(np.count_nonzero(x & y))
    return 2 * c + (1 if n % 2 == 0 and bm.get(half, half + 1)[0] else 0)


def brute_T(bm, n):
    return sum(1 for p in bm.base if p * p < n and bm.get(n - p * p, n - p * p + 1)[0])


def self_test():
    N = 3_000_000
    bm = PrimeBitmap(N)
    assert int(sum(np.count_nonzero(bm.get(lo, min(lo + 10 ** 6, N + 1)))
                   for lo in range(0, N + 1, 10 ** 6))) == 216816, "pi(3e6) mismatch"
    rng = np.random.default_rng(1)
    for A, W in ((0, 4096), (1_000_000, 50_000), (2_000_000, 1 << 16)):
        R = window_R(bm, A, W, B=1 << 15)
        T = window_Tk(bm, A, W, 2)
        for i in rng.choice(np.arange(W)[(np.arange(W) + A) % 2 == 0], 15, replace=False):
            n = A + int(i)
            if n < 4:
                continue
            assert R[i] == brute_R(bm, n), (n, R[i], brute_R(bm, n))
            assert T[i] == brute_T(bm, n), (n, T[i], brute_T(bm, n))
    print("self-test passed: pi(3e6), R(n) and T(n) match brute force on 45 values")


def zeta_term(stats_json):
    """For each range in a run's JSON, the mean over even n of 2(psi(n) - n)/n, the local
    zero-driven offset predicted by differentiating Fujii's formula, next to the observed
    mean of (R + sum_k (2/k) T_k)/HLi - 1. Both are also reported times sqrt(n)."""
    rows = json.load(open(stats_json))
    top = max(r["hi"] for r in rows)
    bm = PrimeBitmap(top)
    C = 1 << 24

    def theta_parts(idx):
        """sum of ln p over the given primes: total, class 1 mod 3, class 2 mod 3."""
        if len(idx) == 0:
            return 0.0, 0.0, 0.0
        L, c = np.log(idx), idx % 3
        return float(L.sum()), float(L[c == 1].sum()), float(L[c == 2].sum())

    pos, run = 0, np.zeros(3)                 # running theta, theta(;3,1), theta(;3,2) on [0, pos)
    for r in sorted(rows, key=lambda r: r["lo"]):
        lo, hi = r["lo"], r["hi"]
        while pos + C <= lo:
            run += theta_parts(np.nonzero(bm.get(pos, pos + C))[0] + pos)
            pos += C
        before = run + theta_parts(np.nonzero(bm.get(pos, lo))[0] + pos)
        lam = np.zeros((3, hi - lo))          # Lambda restricted to: all, class 1, class 2
        idx = np.nonzero(bm.get(lo, hi))[0]
        vals, cls = np.log(idx + lo), (idx + lo) % 3
        lam[0, idx] = vals
        lam[1, idx[cls == 1]] = vals[cls == 1]
        lam[2, idx[cls == 2]] = vals[cls == 2]
        k = 2
        while 2 ** k < hi:
            for p in np.nonzero(bm.get(0, int(hi ** (1 / k)) + 2))[0]:
                pk, lp = int(p) ** k, math.log(p)
                c = pk % 3
                if pk < lo:
                    before[0] += lp
                    if c:
                        before[c] += lp
                elif pk < hi:
                    lam[0, pk - lo] += lp
                    if c:
                        lam[c, pk - lo] += lp
            k += 1
        psi = before[:, None] + np.cumsum(lam, axis=1)
        n = np.arange(lo, hi, dtype=np.float64)
        ev = (np.arange(lo, hi) % 2 == 0) & (n >= 1000)
        r["zeta_term"] = float(np.mean(2 * (psi[0, ev] - n[ev]) / n[ev]))
        # mod 3 race: predicted leftover gap (squares minus non-squares) after adding T
        r["race3_gap"] = float(np.mean(4 * (psi[2, ev] - psi[1, ev]) / n[ev]))
        nm = math.sqrt(lo * hi) if hi > 1.5 * lo else (lo + hi) / 2
        print(f"{r['label']:>24}  sqrt(n)*[(R+sum T_k)/HLi - 1] = {math.sqrt(nm) * (r['rpp'] - 1):+.3f}"
              f"   sqrt(n)*2(psi(n)-n)/n = {math.sqrt(nm) * r['zeta_term']:+.3f}"
              f"   mod 3 gap after T: {100 * r['gap3_rc']:+.5f}%  race prediction {100 * r['race3_gap']:+.5f}%",
              flush=True)
    with open(stats_json, "w") as f:
        json.dump(rows, f, indent=1)


def moments(z):
    m, s = z.mean(), z.std()
    u = (z - m) / s
    return dict(mean=float(m), std=float(s), skew=float(np.mean(u ** 3)),
                exkurt=float(np.mean(u ** 4) - 3))


def granville_compare(A, W):
    """Compare, on the window [A, A+W) of even n, three predictions for R(n):
    plain S*I; ours, S*I - S*eps*I_T (subtract the expected T(n) once); and Granville's
    I* = S*I*(1 - 4 eps/sqrt n) from Funct. Approx. 37 (2007), eq. (1.1).
    Reports sqrt(n)*mean(R/pred - 1) and the mean, std, skewness and excess kurtosis of
    z = (R - pred)/sqrt(pred), the statistic Granville asks about on p. 165."""
    A -= A % 2
    bm = PrimeBitmap(A + W)
    R = window_R(bm, A, W)
    T = window_Tk(bm, A, W, 2)
    S = window_S(bm, A, W)
    eps = window_eps(A, W)
    n = np.arange(A, A + W, dtype=np.int64)
    keep = n % 2 == 0
    n, R, T, S, eps = n[keep], R[keep], T[keep], S[keep], eps[keep]
    nf = n.astype(np.float64)
    Ifun = spline_over(A, A + W, I_exact, base_I, 9)
    ITfun = spline_over(A, A + W, IT_exact, base_IT, 9)
    HLi = S * Ifun(nf)
    preds = {
        "plain HL, S*I": HLi,
        "ours, S*I - S*eps*I_T": HLi - S * eps * ITfun(nf),
        "Granville (1.1), S*I*(1 - 4 eps/sqrt n)": HLi * (1 - 4 * eps / np.sqrt(nf)),
    }
    out = dict(A=A, W=W, count=int(len(n)))
    print(f"window [{A:,}, {A + W:,}), {len(n):,} even n")
    print(f"{'prediction':>42} {'sqrt(n)*mean(R/pred - 1)':>26} {'mean z':>8} {'std z':>7} {'skew':>7} {'ex.kurt':>8}")
    for k, P in preds.items():
        rel = float(np.mean(R / P - 1) * math.sqrt(A + W / 2))
        mo = moments((R - P) / np.sqrt(P))
        out[k] = dict(rel_sqrt_n=rel, **mo)
        print(f"{k:>42} {rel:>+26.3f} {mo['mean']:>+8.3f} {mo['std']:>7.3f} {mo['skew']:>+7.3f} {mo['exkurt']:>+8.3f}")
    # where does the difference between the two corrections show up: by n mod 3
    for k, P in list(preds.items())[1:]:
        z = (R - P) / np.sqrt(P)
        by_class = {c: float(z[n % 3 == c].mean()) for c in (1, 2, 0)}
        out[k]["mean_z_by_n_mod_3"] = by_class
        print(f"  {k}: mean z for n = 1, 2, 0 mod 3: "
              + ", ".join(f"{by_class[c]:+.3f}" for c in (1, 2, 0)))
    return out


def averaged_check(N):
    """(N^2/2 - sum_{p+q<=N} ln p ln q)/N^1.5 versus the prime-power prediction
    2 sum_{k>=2} sum_{p^k<=N} ln p (N - p^k) / N^1.5 (Fujii's formula with prime-only weights)."""
    bm = PrimeBitmap(N)
    isp = bm.get(0, N + 1)
    theta = np.cumsum(np.where(isp, np.log(np.maximum(np.arange(N + 1), 1)), 0.0))
    small = [int(x) for x in np.nonzero(isp[:math.isqrt(N) + 2])[0]]
    rows = []
    M = 10 ** 5
    while M <= N:
        for m in (M, 3 * M):
            if m > N:
                break
            p = np.nonzero(isp[:m + 1])[0]
            obs = (m * m / 2 - float(np.sum(np.log(p) * theta[m - p]))) / m ** 1.5
            pred = sum(2 * math.log(q) * (m - q ** k) for k in range(2, 64)
                       for q in small if q ** k <= m) / m ** 1.5
            rows.append(dict(N=m, observed=obs, prime_powers=pred, residual=obs - pred))
            print(f"N = {m:>11,}: observed {obs:.4f}  prime-power term {pred:.4f}  "
                  f"residual {obs - pred:+.4f}  (RH bound on zero sum: 0.0924)")
        M *= 10
    return rows


# ----------------------------------------------------------------- main
def print_table(rows):
    hdr = (f"{'range':>24} {'count':>9} {'R/HLi':>8} {'(R+T)/HLi':>9} {'all pow':>8} "
           f"{'z mean':>7} {'z std':>6} {'D/T':>6} {'T/Tpred':>7} "
           + " ".join(f"gap{q}:ri/rc/pred" for q in (3, 5)))
    print(hdr)
    for r in rows:
        print(f"{r['label']:>24} {r['count']:>9} {r['ri']:8.5f} {r['rc']:9.5f} {r['rpp']:8.5f} "
              f"{r['z']:+7.3f} {r['z_std']:6.3f} {r['D_over_T']:6.3f} {r['T_over_Tpred']:7.3f} "
              + " ".join(f"{100*r[f'gap{q}_ri']:+.4f}/{100*r[f'gap{q}_rc']:+.4f}/{100*r[f'gap{q}_pred']:+.4f}"
                         for q in (3, 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "window", "check", "average", "zeta", "granville"])
    ap.add_argument("N_or_A", nargs="?", default="10000000")
    ap.add_argument("W", nargs="?", type=int, default=1 << 22)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "check":
        self_test()
        return
    if args.mode == "zeta":
        zeta_term(args.N_or_A)          # here N_or_A is the path of a run's JSON file
        return
    args.N_or_A = int(args.N_or_A)
    if args.mode == "granville":
        out = granville_compare(args.N_or_A, args.W)
        with open(args.out or f"goldbach_granville_{args.N_or_A}.json", "w") as f:
            json.dump(out, f, indent=1)
        return
    if args.mode == "average":
        rows = averaged_check(args.N_or_A)
        with open(args.out or f"goldbach_average_{args.N_or_A}.json", "w") as f:
            json.dump(rows, f, indent=1)
        return
    t0 = time.time()
    if args.mode == "full":
        N = args.N_or_A
        W = args.W
        bm = PrimeBitmap(N + 1)
        print(f"sieve to {N:,}: {time.time() - t0:.1f}s, primes: "
              f"{sum(int(np.unpackbits(bm.packed[i:i + 1 << 20]).sum()) for i in range(0, len(bm.packed), 1 << 20)):,}")
        Ifun = spline_over(1000, N + 2, I_exact, base_I, 400)
        ITfun = spline_over(1000, N + 2, IT_exact, base_IT, 200)
        accs, j = [], 10
        while 2 ** j < N:
            accs.append(Acc(f"[2^{j}, 2^{j + 1})", 2 ** j, min(2 ** (j + 1), N + 1)))
            j += 1
        for A in range(0, N + 1, W):
            Wa = min(W, N + 1 - A)
            dt, *_ = process_window(bm, A, Wa, accs, Ifun, ITfun)
            print(f"  window [{A:,}, {A + Wa:,}) done in {dt:.1f}s", flush=True)
    else:
        A, W = args.N_or_A, args.W
        A -= A % 2
        bm = PrimeBitmap(A + W)
        print(f"sieve to {A + W:,}: {time.time() - t0:.1f}s", flush=True)
        Ifun = spline_over(A, A + W, I_exact, base_I, 9)
        ITfun = spline_over(A, A + W, IT_exact, base_IT, 9)
        accs = [Acc(f"[{A:.3g}, {A + W:.3g})", A, A + W)]
        dt, n, R, T = process_window(bm, A, W, accs, Ifun, ITfun)
        print(f"window done in {dt:.1f}s; brute-force spot checks:", flush=True)
        rng = np.random.default_rng(2)
        for i in rng.choice(len(n), 3, replace=False):
            b = brute_R(bm, int(n[i]))
            print(f"    n = {int(n[i]):,}: R = {int(R[i]):,}, brute force {b:,}, "
                  f"T = {int(T[i])}, brute force {brute_T(bm, int(n[i]))}")
            assert b == R[i]
    rows = [a.summary() for a in accs if a.s.get("count")]
    print_table(rows)
    out = args.out or f"goldbach_{args.mode}_{args.N_or_A}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"total {time.time() - t0:.1f}s; wrote {out}")


if __name__ == "__main__":
    main()
