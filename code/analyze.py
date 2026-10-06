#!/usr/bin/env python3
"""
Tables, tests and figure data for the revision, from the block sums written by goldbach_rev.py.

Every mean/gap is a ratio of block sums; standard errors come from a block bootstrap over
blocks of consecutive even n (1024 even n in windows; smaller blocks in the small dyadic
ranges, at least 32 blocks per range), 2000 resamples, seed 12345.
"sqrt(n)-scaled" means: each block's sum is multiplied by sqrt(n) at that block before
summing, so sqrt(n) * x is averaged per n (x ~ n^(-1/2) varies inside a dyadic range).
"""
import glob
import itertools
import json
import math
import os

import numpy as np

QS = (3, 5, 7, 11, 13)
B = 2000
SEED = 12345
FRESH = [125894656, 199524352, 316227584, 501186560, 794329088, 1258921984, 1995259904, 3162275840]


class Rg:
    def __init__(self, sums, names, meta, label, hists=None, extra=None):
        self.sums = sums
        self.idx = {k: i for i, k in enumerate(names)}
        self.lo, self.hi, self.be = (int(x) for x in meta)
        self.label = label
        self.hists = hists or {}
        self.extra = extra or {}
        c = sums[:, self.idx["cnt"]]
        self.sqn = sums[:, self.idx["sqrtn"]] / c
        self.ssc = sums * self.sqn[:, None]
        self.nmid = float((sums[:, self.idx["sqrtn"]].sum() / c.sum()) ** 2)

    def agg(self, k):
        """Merge k consecutive blocks (block-length sensitivity)."""
        nb = (len(self.sums) // k) * k
        s = self.sums[:nb].reshape(-1, k, self.sums.shape[1]).sum(1)
        names = sorted(self.idx, key=self.idx.get)
        return Rg(s, names, (self.lo, self.hi, self.be * k), self.label)

    def boot(self, f, b=B, seed=SEED):
        """Point estimate and bootstrap SE of f(v, vs) (v: sums, vs: sqrt(n)-scaled sums)."""
        est = f(Acc(self.sums.sum(0), self.idx), Acc(self.ssc.sum(0), self.idx))
        rng = np.random.default_rng(seed)
        nb = len(self.sums)
        C = np.zeros((b, nb))
        for i in range(b):
            C[i] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        reps = f(Acc(C @ self.sums, self.idx), Acc(C @ self.ssc, self.idx))
        return float(est), float(np.std(reps, ddof=1))


class Acc:
    def __init__(self, v, idx):
        self.v, self.idx = v, idx

    def __getitem__(self, k):
        return self.v[..., self.idx[k]]


LW_KEYS = ("lw", "lpsi", "l3w", "c0_l3w", "c1_l3w", "c2_l3w")


def merge_lw(sums, names, lwd, prefix=""):
    """Append the block sums of the log-weighted linear terms (lw.py), checking block alignment."""
    cnt = sums[:, names.index("cnt")]
    assert np.array_equal(cnt, lwd[prefix + "cnt"]), "block mismatch with lw file"
    for c in range(3):
        assert np.array_equal(sums[:, names.index(f"c{c}_cnt")], lwd[prefix + f"c{c}_cnt"])
    # psi cross-check: uniform L recomputed independently in lw.py
    dl = np.max(np.abs(sums[:, names.index("L")] - lwd[prefix + "lpsi"]) / np.maximum(cnt, 1))
    assert dl < 1e-9, ("psi mismatch", dl)
    return np.column_stack([sums] + [lwd[prefix + k] for k in LW_KEYS]), names + list(LW_KEYS)


def load_full(path):
    f = np.load(path)
    names = list(f["stat_names"])
    N = path.split("_")[-1].split(".")[0]
    lwf = np.load(f"out/lw_full_{N}.npz") if os.path.exists(f"out/lw_full_{N}.npz") else None
    out = []
    for p in sorted({k.split("_")[0] for k in f.files if k.startswith("r") and k[1:3].isdigit()}):
        sums, nm = f[p + "_sums"], names
        if lwf is not None:
            sums, nm = merge_lw(sums, names, lwf, p + "_")
        out.append(Rg(sums, nm, f[p + "_meta"], str(f[p + "_label"]),
                      {k.split("hist_")[1]: f[k] for k in f.files if k.startswith(p + "_hist_")}))
    return out


def load_window(A):
    f = np.load(f"out/window_{A}.npz")
    names = list(f["stat_names"])
    sums, nm = f["w_sums"], names
    if os.path.exists(f"out/lw_window_{A}.npz"):
        sums, nm = merge_lw(sums, names, np.load(f"out/lw_window_{A}.npz"))
    if os.path.exists(f"out/lwq_window_{A}.npz"):
        g = np.load(f"out/lwq_window_{A}.npz")
        extra = []
        for q in QS:
            for t in ("p", "m"):
                assert np.array_equal(sums[:, nm.index(f"g{q}{t}_cnt")], g[f"g{q}{t}_cnt"]), "lwq block mismatch"
                extra.append(g[f"g{q}{t}_lq"])
                nm = nm + [f"g{q}{t}_lq"]
        sums = np.column_stack([sums] + extra)
    return Rg(sums, nm, f["w_meta"], f"window {A}",
              {k.split("hist_")[1]: f[k] for k in f.files if k.startswith("w_hist_")},
              {k[4:]: float(f[k]) for k in f.files if k.startswith("w_x_")})


# ---------------------------------------------------------------- statistic definitions
def mean(x):
    return lambda v, vs: v[x] / v["cnt"]


def smean(x):
    return lambda v, vs: vs[x] / v["cnt"]


def smean_lin(terms):
    """sqrt(n)-scaled mean of a linear combination sum c * x (x may be 'cnt' for constants)."""
    return lambda v, vs: sum(c * vs[x] for c, x in terms) / v["cnt"]


def ratio(a, b):
    return lambda v, vs: v[a] / v[b]


def ngap(q, x, scale=True):
    """(q-2) * sqrt(n) * [mean over n with (n|q)=+1 of x  -  mean over (n|q)=-1]."""
    def f(v, vs):
        src = vs if scale else v
        return (q - 2) * (src[f"g{q}p_{x}"] / v[f"g{q}p_cnt"] - src[f"g{q}m_{x}"] / v[f"g{q}m_cnt"])
    return f


def cgap(x):
    """sqrt(n) * [mean over n = 1 mod 3 of x - mean over n = 2 mod 3]."""
    return lambda v, vs: vs[f"c1_{x}"] / v["c1_cnt"] - vs[f"c2_{x}"] / v["c2_cnt"]


def cmean(c, x):
    return lambda v, vs: v[f"c{c}_{x}"] / v[f"c{c}_cnt"]


def zsd(z):
    return lambda v, vs: np.sqrt(v[z + "2"] / v["cnt"] - (v[z] / v["cnt"]) ** 2)


def zskew(z):
    def f(v, vs):
        n = v["cnt"]; m1, m2, m3 = v[z] / n, v[z + "2"] / n, v[z + "3"] / n
        var = m2 - m1 ** 2
        return (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5
    return f


def zkurt(z):
    def f(v, vs):
        n = v["cnt"]; m1, m2, m3, m4 = (v[z] / n, v[z + "2"] / n, v[z + "3"] / n, v[z + "4"] / n)
        var = m2 - m1 ** 2
        return (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2 - 3
    return f


def pm(x):
    return [round(x[0], 4), round(x[1], 4)]


def cramer_sd(n):
    """Analytic within-window spread of (R-E)/sqrt(E) in the Cramer model (quadratic part)."""
    a = np.linspace(3, n - 3, 400001)
    p = np.minimum(1, 2 / np.log(a)); pb = p[::-1]
    return float(np.sqrt(2 * np.sum(p * pb * (1 - p) * (1 - pb)) / np.sum(p * pb)))


# ---------------------------------------------------------------- main
def row_stats(r):
    d = dict(label=r.label, lo=r.lo, hi=r.hi, count=int(r.sums[:, r.idx["cnt"]].sum()), nmid=r.nmid,
             blocks=len(r.sums), block_even=r.be)
    d["shortfall_R"] = pm(r.boot(smean_lin([(1, "cnt"), (-1, "ri")])))
    d["tau"] = pm(r.boot(smean("t")))
    d["tau_pred"] = pm(r.boot(smean("tp")))
    d["granville_term"] = round(4 * float(r.sums[:, r.idx["eps"]].sum() / r.sums[:, r.idx["cnt"]].sum()), 4)
    d["T_over_Tpred"] = pm(r.boot(ratio("Traw", "Tpraw")))
    d["Tpw_over_Tpred"] = pm(r.boot(ratio("Tpw", "Tpraw")))
    d["offset_rw"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt")])))
    d["zeta"] = pm(r.boot(smean("L")))                      # exact linear term 2(psi(n-1)-(n-1))/n
    d["zeta_L3"] = pm(r.boot(smean("L3")))                  # class-resolved (pre-registered definition)
    d["resid"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "L")])))
    d["resid_draft"] = pm(r.boot(smean_lin([(1, "ri"), (1, "t"), (1, "h5"), (-1, "cnt"), (-1, "L")])))
    # powers of 2 can never pair for even n: removing them from the linear term adds 2 psi_2(n)/n
    c = r.sums[:, r.idx["cnt"]]
    nb = r.sqn ** 2
    adj = float(np.sum(c * 2 * np.floor(np.log2(nb)) * math.log(2) / r.sqn) / c.sum())
    d["resid_odd"] = [round(d["resid"][0] + adj, 4), d["resid"][1]]
    d["two_power_adj"] = round(adj, 4)
    if "lw" in r.idx:
        d["zeta_w"] = pm(r.boot(smean("lw")))
        d["resid_w"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "lw")])))
        d["race_pred_w"] = pm(r.boot(cgap("l3w")))
        d["race_resid_w"] = pm(r.boot(lambda v, vs: cgap("rw")(v, vs) - cgap("l3w")(v, vs)))
    d["higher"] = pm(r.boot(smean("h")))
    d["sd_zo"] = pm(r.boot(zsd("zo")))
    d["sd_zc"] = pm(r.boot(zsd("zc")))
    d["sd_zr"] = pm(r.boot(zsd("zr")))
    d["gaps"] = {}
    for q in QS:
        d["gaps"][q] = dict(raw=pm(r.boot(ngap(q, "ri"))), rw=pm(r.boot(ngap(q, "rw"))),
                            ours=pm(r.boot(ngap(q, "ro"))), granville=pm(r.boot(ngap(q, "rg"))),
                            pred=round(-ngap(q, "tp")(Acc(r.sums.sum(0), r.idx), Acc(r.ssc.sum(0), r.idx)), 4),
                            pp=pm(r.boot(lambda v, vs, q=q: ngap(q, "ri")(v, vs) - ngap(q, "rw")(v, vs))))
        if f"g{q}p_lq" in r.idx:
            d["gaps"][q]["race_w"] = pm(r.boot(ngap(q, "lq")))
            d["gaps"][q]["rw_minus_race_w"] = pm(r.boot(lambda v, vs, q=q: ngap(q, "rw")(v, vs) - ngap(q, "lq")(v, vs)))
    d["race_obs"] = pm(r.boot(cgap("rw")))
    d["race_pred"] = pm(r.boot(cgap("L3")))
    return d


def window_extra(r):
    d = {}
    d["off_plain"] = pm(r.boot(smean_lin([(1, "ri"), (-1, "cnt")])))
    d["off_ours"] = pm(r.boot(smean("ro")))
    d["off_granville"] = pm(r.boot(smean("rg")))
    for c in (1, 2, 0):
        d[f"z_ours_c{c}"] = pm(r.boot(cmean(c, "zo")))
        d[f"z_gran_c{c}"] = pm(r.boot(cmean(c, "zg")))
    d["decomp_total"] = pm(r.boot(smean_lin([(1, "ri"), (-1, "cnt"), (1, "tp")])))
    d["decomp_Tshort"] = pm(r.boot(smean_lin([(1, "tp"), (-1, "t")])))
    d["decomp_higher"] = pm(r.boot(smean_lin([(-1, "h")])))
    d["decomp_zeta"] = pm(r.boot(smean("L")))
    d["decomp_resid"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "L")])))
    if "lw" in r.idx:
        d["decomp_zeta_w"] = pm(r.boot(smean("lw")))
        d["decomp_resid_w"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "lw")])))
        d["X_w"] = pm(r.boot(smean("lw")))
    d["mom_zo"] = dict(mean=pm(r.boot(mean("zo"))), sd=pm(r.boot(zsd("zo"))),
                       skew=pm(r.boot(zskew("zo"))), exkurt=pm(r.boot(zkurt("zo"))))
    d["mom_zg"] = dict(mean=pm(r.boot(mean("zg"))), sd=pm(r.boot(zsd("zg"))),
                       skew=pm(r.boot(zskew("zg"))), exkurt=pm(r.boot(zkurt("zg"))))
    d["Y"] = pm(r.boot(smean_lin([(1, "rw"), (-1, "cnt")])))
    d["X_mult"] = pm(r.boot(smean("L3")))          # pre-registered definition
    d["X_mult_L"] = pm(r.boot(smean("L")))
    d["X_add"] = pm(r.boot(smean("LoverS")))
    d["E_start_over_sqrtA"] = r.extra.get("E_start", float("nan")) / math.sqrt(r.lo)
    d["resid_by_S"] = {f"{a}{b}": pm(r.boot(lambda v, vs, a=a, b=b: vs[f"s{a}{b}_yres"] / v[f"s{a}{b}_cnt"]))
                       for a in (0, 1) for b in (0, 1)}
    d["cramer_sd"] = round(cramer_sd(r.nmid), 4)
    # block-length sensitivity for two key statistics
    d["blocklen"] = {}
    for k in (1, 4, 16):
        rr = r.agg(k)
        d["blocklen"][k] = dict(race=round(rr.boot(cgap("rw"))[1], 5),
                                race_minus_pred=round(rr.boot(lambda v, vs: cgap("rw")(v, vs) - cgap("L3")(v, vs))[1], 5),
                                race_minus_pred_w=(round(rr.boot(lambda v, vs: cgap("rw")(v, vs) - cgap("l3w")(v, vs))[1], 5)
                                                   if "l3w" in rr.idx else None),
                                resid_w=(round(rr.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "lw")]))[1], 5)
                                         if "lw" in rr.idx else None),
                                resid=round(rr.boot(smean_lin([(1, "rw"), (-1, "cnt"), (-1, "L")]))[1], 5),
                                gap3=round(rr.boot(ngap(3, "ri"))[1], 5))
    return d


def wls_origin(y, x, se):
    w = 1 / se ** 2
    beta = np.sum(w * x * y) / np.sum(w * x * x)
    return float(beta), float(1 / math.sqrt(np.sum(w * x * x)))


def main():
    res = dict(ranges=[], windows={})
    full = load_full("out/full_100000000.npz")
    for r in full:
        res["ranges"].append(row_stats(r))
        print(res["ranges"][-1]["label"], res["ranges"][-1]["resid"], res["ranges"][-1]["resid_draft"], flush=True)
    wins = sorted(int(p.split("_")[1].split(".")[0]) for p in glob.glob("out/window_*.npz"))
    for A in wins:
        r = load_window(A)
        d = row_stats(r)
        d.update(window_extra(r))
        res["windows"][A] = d
        print(A, "resid", d["resid"], "race", d["race_obs"], d["race_pred"], flush=True)

    # ---------------- pre-registered tests on the fresh windows
    have = [A for A in FRESH if A in res["windows"]]
    if len(have) == len(FRESH):
        G = np.array([res["windows"][A]["race_obs"][0] for A in FRESH])
        SE = np.array([res["windows"][A]["race_obs"][1] for A in FRESH])
        P = np.array([res["windows"][A]["race_pred"][0] for A in FRESH])
        beta, sb = wls_origin(G, P, SE)
        lo, hi = beta - 1.96 * sb, beta + 1.96 * sb
        if lo > 0 and lo >= 0.75 and hi <= 1.25:
            verdict = "SUPPORTED"
        elif lo <= 0 <= hi or hi < 0.75 or lo > 1.25:
            verdict = "NOT SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
        r_obs = float(np.corrcoef(G, P)[0, 1])
        perms = [float(np.corrcoef(np.array(p), P)[0, 1]) for p in itertools.permutations(G)]
        p_perm = float(np.mean(np.array(perms) >= r_obs - 1e-12))
        chi2 = float(np.sum(((G - P) / SE) ** 2))
        # secondary: multiplicative vs additive
        Y = np.array([res["windows"][A]["Y"][0] for A in FRESH])
        SY = np.array([res["windows"][A]["Y"][1] for A in FRESH])
        out2 = {}
        for name, key in (("multiplicative", "X_mult"), ("additive", "X_add")):
            X = np.array([res["windows"][A][key][0] for A in FRESH])
            w = 1 / SY ** 2
            a = float(np.sum(w * (Y - X)) / np.sum(w))
            sa = float(1 / math.sqrt(np.sum(w)))
            c2 = float(np.sum(w * (Y - X - a) ** 2))
            Xc = np.column_stack([np.ones_like(X), X])
            Wm = np.diag(w)
            cov = np.linalg.inv(Xc.T @ Wm @ Xc)
            ab = cov @ Xc.T @ Wm @ Y
            out2[name] = dict(intercept=[a, sa], chi2_7dof=c2, free_fit_a=[float(ab[0]), float(math.sqrt(cov[0, 0]))],
                              free_fit_b=[float(ab[1]), float(math.sqrt(cov[1, 1]))], X=X.tolist())
        # exploratory (post hoc): the log-weighted versions, not part of the pre-registered decision
        expl = {}
        if all("race_pred_w" in res["windows"][A] for A in FRESH):
            Pw = np.array([res["windows"][A]["race_pred_w"][0] for A in FRESH])
            bw, sbw = wls_origin(G, Pw, SE)
            Xw = np.array([res["windows"][A]["X_w"][0] for A in FRESH])
            w = 1 / SY ** 2
            aw = float(np.sum(w * (Y - Xw)) / np.sum(w))
            expl = dict(Pw=Pw.tolist(), beta_w=bw, se_beta_w=sbw, chi2_race_w=float(np.sum(((G - Pw) / SE) ** 2)),
                        Xw=Xw.tolist(), intercept_w=[aw, float(1 / math.sqrt(np.sum(w)))],
                        chi2_zeta_w_7dof=float(np.sum(w * (Y - Xw - aw) ** 2)),
                        chi2_zeta_w_8dof_noint=float(np.sum(w * (Y - Xw) ** 2)))
        res["prereg"] = dict(exploratory=expl, G=G.tolist(), SE=SE.tolist(), P=P.tolist(), beta=beta, se_beta=sb,
                             ci=[lo, hi], verdict=verdict, pearson=r_obs, perm_p_one_sided=p_perm,
                             chi2_8dof=chi2, Y=Y.tolist(), SY=SY.tolist(), zeta=out2)
        print("PRE-REGISTERED:", json.dumps(res["prereg"], indent=1))
    json.dump(res, open("out/results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
