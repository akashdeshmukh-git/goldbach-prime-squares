#!/usr/bin/env python3
"""Table bodies and numbers for the manuscript, from out/results.json and out/averaged.json.
Writes out/tables.json (LaTeX table bodies) and out/text_numbers.json (numbers quoted in the text)."""
import json
import math

import numpy as np

R = json.load(open("out/results.json"))
AV = json.load(open("out/averaged.json"))
RG = R["ranges"]
WIN = {int(k): v for k, v in R["windows"].items()}
DRAFT = [10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10]
FRESH = [125894656, 199524352, 316227584, 501186560, 794329088, 1258921984, 1995259904, 3162275840]
ALLW = sorted(WIN)


def pe(x, digits=None):
    """value(se) with the SE given to two significant digits (one if the leading digit is >= 3)."""
    v, s = x
    if not math.isfinite(v):
        return "--"
    if s <= 0 or not math.isfinite(s):
        return mm(f"{v:.4f}")
    e = math.floor(math.log10(s))
    nd = 2 if (s / 10 ** e < 2 and e >= -3) else 1
    dec = max(0, -(e - nd + 1))
    if digits is not None:
        dec = digits
    se_digits = round(s * 10 ** dec)
    if se_digits == 0:
        se_digits = 1
    return mm(f"{v:.{dec}f}({se_digits})")


def mm(t):
    """Typeset a leading sign as a math minus."""
    return "$-$" + t[1:] if t.startswith("-") else t


def num(v, d=3):
    return mm(f"{v:.{d}f}")


def sci(A):
    e = int(math.floor(math.log10(A)))
    m = A / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else f"${m:.2f}\\cdot10^{{{e}}}$"


def mlab(A):
    return sci(A) + ("$^\\dagger$" if A in FRESH else "")


def rlabel(d):
    lab = d["label"]                       # "[2^10, 2^11)"
    a, b = lab.strip("[)").split(", ")
    j = int(a.split("^")[1])
    if d["hi"] > 2 ** 26 and j == 26:
        return "$[2^{26},10^{8}]$"
    return f"$[2^{{{j}}},2^{{{j + 1}}})$"


out = {}


def macro(name, body):
    out[name] = body


# ------------------------------------------------------------------ Table 1
rows = []
for d in RG:
    rows.append([rlabel(d), f"{d['count']:,}".replace(",", "\\,"), pe(d["shortfall_R"]), pe(d["tau"]),
                 num(d["tau_pred"][0]), num(d["Tpw_over_Tpred"][0]), pe(d["offset_rw"]), pe(d["resid"]),
                 pe(d["resid_w"]) if "resid_w" in d else "--", num(d["sd_zo"][0])])
rows.append(None)
for A in ALLW:
    d = WIN[A]
    rows.append([mlab(A), f"{d['count']:,}".replace(",", "\\,"), pe(d["shortfall_R"]), pe(d["tau"]),
                 num(d["tau_pred"][0]), num(d["Tpw_over_Tpred"][0], 4), pe(d["offset_rw"]), pe(d["resid"]),
                 pe(d["resid_w"]) if "resid_w" in d else "--", num(d["sd_zo"][0])])
body = []
for r in rows:
    body.append("\\midrule" if r is None else " & ".join(r) + " \\\\")
macro("TableOneBody", "\n".join(body))

# ------------------------------------------------------------------ Table 2
body = []
for A in DRAFT:
    if A not in WIN:
        continue
    d = WIN[A]
    body.append(" & ".join([sci(A), pe(d["off_plain"]), pe(d["off_ours"]), pe(d["off_granville"]),
                            pe(d["z_ours_c0"], 3), pe(d["z_ours_c1"], 3), pe(d["z_ours_c2"], 3),
                            pe(d["z_gran_c0"], 3), pe(d["z_gran_c1"], 3), pe(d["z_gran_c2"], 3)]) + " \\\\")
macro("TableTwoBody", "\n".join(body))

# ------------------------------------------------------------------ Table 2b: decomposition
body = []
for A in ALLW:
    d = WIN[A]
    over = 4 * 0 + d["granville_term"] - d["tau_pred"][0]
    tsh = d["decomp_Tshort"]
    hi = d["decomp_higher"]
    zw = d.get("decomp_zeta_w", [float("nan"), 0])
    rw = d.get("decomp_resid_w", [float("nan"), 0])
    body.append(" & ".join([mlab(A), pe(d["off_granville"]), num(over), pe(tsh), pe(hi), pe(d["decomp_zeta"]),
                            pe(zw), pe(rw)]) + " \\\\")
macro("TableDecompBody", "\n".join(body))

# ------------------------------------------------------------------ Table 3: residue gaps
body = []
for q in (3, 5, 7, 11, 13):
    cells = [f"{q}"]
    for A in (10 ** 9, 10 ** 10):
        if A not in WIN:
            cells += ["--"] * 5
            continue
        g = WIN[A]["gaps"][str(q)]
        cells += [pe(g["raw"]), num(g["pred"], 2), pe(g["ours"]),
                  num(g["race_w"][0], 2) if "race_w" in g else "--", pe(g["granville"])]
    body.append(" & ".join(cells) + " \\\\")
macro("TableThreeBody", "\n".join(body))

# ------------------------------------------------------------------ Table 4
body = []
for r in AV["rows"]:
    N = r["N"]
    e = int(math.floor(math.log10(N)))
    lab = f"$10^{{{e}}}$" if N == 10 ** e else f"$3\\cdot10^{{{e}}}$"
    body.append(" & ".join([lab, f"{r['A']:.5f}", f"{r['Pan']:.5f}", f"{4 / 3 + 1.5 * N ** (-1 / 6):.5f}",
                            mm(f"{r['B']:.6f}"), mm(f"{r['Z']:.6f}"),
                            mm(f"{r['B_minus_pred'] * math.sqrt(N):.2f}")]) + " \\\\")
macro("TableFourBody", "\n".join(body))

# ------------------------------------------------------------------ Table 5: pre-registered
P = R.get("prereg")
if P:
    ex = P.get("exploratory", {})
    body = []
    for j, A in enumerate(FRESH):
        d = WIN[A]
        cells = [sci(A), "$+$" if d["E_start_over_sqrtA"] > 0 else "$-$",
                 pe([P["G"][j], P["SE"][j]]), num(P["P"][j]),
                 num(ex["Pw"][j]) if ex else "--",
                 pe([P["Y"][j], P["SY"][j]]), num(P["zeta"]["multiplicative"]["X"][j]),
                 num(P["zeta"]["additive"]["X"][j]), num(ex["Xw"][j]) if ex else "--"]
        body.append(" & ".join(cells) + " \\\\")
    macro("TableFiveBody", "\n".join(body))
    macro("nBeta", f"{P['beta']:.3f}")
    macro("nBetaSE", f"{P['se_beta']:.3f}")
    macro("nBetaLo", f"{P['ci'][0]:.3f}")
    macro("nBetaHi", f"{P['ci'][1]:.3f}")
    macro("nVerdict", P["verdict"].lower())
    macro("nPearson", f"{P['pearson']:.3f}")
    macro("nPermP", f"{P['perm_p_one_sided']:.1e}".replace("e-0", "\\cdot10^{-") + "}")
    macro("nChiRace", f"{P['chi2_8dof']:.0f}")
    zm, za = P["zeta"]["multiplicative"], P["zeta"]["additive"]
    macro("nChiMult", f"{zm['chi2_7dof']:.0f}")
    macro("nChiAdd", f"{za['chi2_7dof']:.0f}")
    macro("nBMult", pe(zm["free_fit_b"]))
    macro("nBAdd", pe(za["free_fit_b"]))
    macro("nAMult", pe(zm["intercept"]))
    if ex:
        macro("nBetaW", pe([ex["beta_w"], ex["se_beta_w"]]))
        macro("nChiRaceW", f"{ex['chi2_race_w']:.1f}")
        macro("nChiZetaW", f"{ex['chi2_zeta_w_8dof_noint']:.1f}")
        macro("nAW", pe(ex["intercept_w"]))
        macro("nChiZetaWint", f"{ex['chi2_zeta_w_7dof']:.1f}")

json.dump(out, open("out/tables.json", "w"), indent=1)
for k, v in out.items():
    print(k, v[:200])


# ------------------------------------------------------------------ numbers quoted in the text
def rng(vals, d=2, unit=""):
    lo, hi = min(vals), max(vals)
    return f"${lo:.{d}f}$ and ${hi:.{d}f}${unit}".replace("$-", "$-")


T = {}
top = [A for A in ALLW if A >= 10 ** 9]
T["TRATIO_TOP"] = f"${min(WIN[A]['T_over_Tpred'][0] for A in top):.3f}$--${max(WIN[A]['T_over_Tpred'][0] for A in top):.3f}$"
big = [A for A in ALLW if A > 10 ** 8]
T["TPI_RANGE"] = rng([WIN[A]["Tpw_over_Tpred"][0] for A in big], 4)
T["GOFF_RANGE"] = f"${min(WIN[A]['off_granville'][0] for A in ALLW):.2f}$ to ${max(WIN[A]['off_granville'][0] for A in ALLW):.2f}$ times $1/\\sqrt n$"
T["OOFF_RANGE"] = f"offsets between ${min(WIN[A]['off_ours'][0] for A in ALLW):.2f}$ and ${max(WIN[A]['off_ours'][0] for A in ALLW):+.2f}$, which are the zeta term (Table~\\ref{{tab:decomp}})"
T["ZETA_RANGE"] = f"${min(WIN[A]['decomp_zeta_w'][0] for A in ALLW if 'decomp_zeta_w' in WIN[A]):.2f}$ and ${max(WIN[A]['decomp_zeta_w'][0] for A in ALLW if 'decomp_zeta_w' in WIN[A]):+.2f}$"
rw = [WIN[A]["decomp_resid_w"] for A in ALLW if "decomp_resid_w" in WIN[A]]
T["RESW_MAX"] = f"${max(abs(x[0]) for x in rw):.3f}$"
T["RESW_WIN"] = f"${max(abs(x[0]) for x in rw):.3f}$"
wv = np.array([1 / x[1] ** 2 for x in rw])
xv = np.array([x[0] for x in rw])
mw = float(np.sum(wv * xv) / np.sum(wv))
T["RESW_WIN"] = (f"${max(abs(x[0]) for x in rw):.3f}$ (weighted mean ${mw:+.4f}({round(1e4 / math.sqrt(wv.sum()))})$, "
                 f"with the largest deviation, ${max(rw, key=lambda x: abs(x[0]) / x[1])[0]:+.3f}$, "
                 f"at {max(abs(x[0]) / x[1] for x in rw):.1f} standard errors)")
# fit resid_w = c log n / sqrt n over the dyadic ranges (sqrt(n) units)
y = np.array([d["resid_w"][0] for d in RG]); e = np.array([d["resid_w"][1] for d in RG])
x = np.array([math.log(d["nmid"]) / math.sqrt(d["nmid"]) for d in RG])
c = float(np.sum(x * y / e ** 2) / np.sum(x * x / e ** 2))
T["RESW_C"] = f"{c:.1f}"
fitc = json.load(open("out/fig4_fit.json"))["fig4_fit_c"] if __import__("os").path.exists("out/fig4_fit.json") else float("nan")
T["FIT_C"] = f"{fitc:.2f}"
# prime-power part of the residue gaps vs prediction
agree = []
for A in (10 ** 9, 10 ** 10):
    if A in WIN:
        for q in ("3", "5", "7", "11", "13"):
            g = WIN[A]["gaps"][q]
            if "pp" in g:
                agree.append(abs(g["pp"][0] / g["pred"] - 1))
T["PP_AGREE"] = f"${100 * max(agree):.1f}\\%$" if agree else "--"
# race-term comparison for all q (exploratory)
dev = []
for A in ALLW:
    for q in ("3", "5", "7", "11", "13"):
        g = WIN[A]["gaps"][q]
        if "rw_minus_race_w" in g:
            dev.append(g["rw_minus_race_w"][0] / g["rw_minus_race_w"][1])
if dev:
    dev = np.array(dev)
    T["RACEQ_SENTENCE"] = (f"over the {len(dev) // 5} windows and five moduli, the observed gap of $R_w/\\HLi$ minus this "
                           f"prediction has $\\chi^2={np.sum(dev ** 2):.1f}$ on {len(dev)} degrees of freedom, "
                           f"and its largest deviation is {np.max(np.abs(dev)):.1f} standard errors.")
else:
    T["RACEQ_SENTENCE"] = ""
if P:
    T["BETA"] = pe([P["beta"], P["se_beta"]])
    T["BETALO"] = f"{P['ci'][0]:.3f}"
    T["BETAHI"] = f"{P['ci'][1]:.3f}"
    T["PEARSON"] = f"{P['pearson']:.3f}"
    cnt = round(P["perm_p_one_sided"] * 40320)
    T["PERMCOUNT"] = {1: "one", 2: "two", 3: "three"}.get(cnt, str(cnt))
    T["PERMP"] = f"{P['perm_p_one_sided']:.1e}".replace("e-05", "\\cdot10^{-5}").replace("e-04", "\\cdot10^{-4}")
    T["CHIRACE"] = f"{P['chi2_8dof']:.0f}"
    zm, za = P["zeta"]["multiplicative"], P["zeta"]["additive"]
    T["CHIMULT"] = f"{zm['chi2_7dof']:.0f}"
    T["CHIADD"] = f"{za['chi2_7dof']:.0f}"
    T["BMULT"] = pe(zm["free_fit_b"])
    T["BADD"] = pe(za["free_fit_b"])
    ex = P.get("exploratory", {})
    T["BETAW"] = pe([ex["beta_w"], ex["se_beta_w"]]) if ex else "--"
    T["CHIRACEW"] = f"{ex['chi2_race_w']:.1f}" if ex else "--"
    T["CHIZETAW"] = f"{ex['chi2_zeta_w_8dof_noint']:.1f}" if ex else "--"
# block-length sensitivity
rr, rc = [], []
for A in ALLW:
    bl = WIN[A]["blocklen"]
    for k in ("4", "16"):
        if bl[k].get("resid_w"):
            rr.append(bl[k]["resid_w"] / bl["1"]["resid_w"])
        if bl[k].get("race_minus_pred_w"):
            rr.append(bl[k]["race_minus_pred_w"] / bl["1"]["race_minus_pred_w"])
        rc.append(bl[k]["race"] / bl["1"]["race"])
T["BL_RESID"] = f"${max(rr):.1f}$" if rr else "--"
T["BL_RACE"] = f"${max(rc):.1f}$"
T["SD_TOP"] = f"${WIN[max(ALLW)]['sd_zo'][0]:.2f}$"
# tables
T["TABLE1"] = out["TableOneBody"]
T["TABLE2"] = out["TableTwoBody"]
T["TABLE3"] = out["TableThreeBody"]
T["TABLE4"] = out["TableFourBody"]
T["TABLE5"] = out.get("TableFiveBody", "")
T["TABLEDECOMP"] = out["TableDecompBody"]
json.dump(T, open("out/text_numbers.json", "w"), indent=1)
for k, v in T.items():
    if not k.startswith("TABLE"):
        print(k, "=", v)
