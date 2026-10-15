#!/usr/bin/env python3
"""Merge CNN shard CSVs and produce paper-ready tables/figures.

  python analyze_cnn.py --in experiments/cnn_cifar10/results --out experiments/cnn_cifar10/analysis
Outputs: summary.csv (ASR/acc/TPR/FPR per alpha x method, bootstrap CI), lift_vs_oracle.csv (per seed),
paired_tests.csv (Wilcoxon, Holm), auroc_by_phase.csv (closed-loop score AUROC), table_cnn.tex, fig_cnn.pdf.
Interpretation guard: ASR is only meaningful when the benign-only oracle actually learns; the script
reports the oracle's mean clean accuracy and warns below --min-oracle-acc.
"""
import argparse, glob, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def boot(x, rng, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1); return tuple(np.percentile(m, [2.5, 97.5]))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-oracle-acc", type=float, default=0.5); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20261013)
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.inp}/raw_shard*.csv"))], ignore_index=True)
    raw = raw.drop_duplicates(["seed", "alpha", "method"])
    S = []
    for (al, m), g in raw.groupby(["alpha", "method"]):
        lo, hi = boot(g.asr, rng)
        S.append(dict(alpha=al, method=m, n=len(g), asr=100 * g.asr.mean(), asr_lo=100 * lo, asr_hi=100 * hi,
                      acc=100 * g.clean_accuracy.mean(), tpr=pd.to_numeric(g.cum_tpr, errors="coerce").mean(),
                      fpr=pd.to_numeric(g.cum_fpr, errors="coerce").mean()))
    S = pd.DataFrame(S); S.to_csv(out / "summary.csv", index=False)
    W = raw.pivot_table(index=["alpha", "seed"], columns="method", values="asr")
    if "oracle" in W:
        lift = (W.sub(W["oracle"], axis=0) * 100).drop(columns="oracle").reset_index().melt(id_vars=["alpha", "seed"], var_name="method", value_name="lift_pp")
        lift.to_csv(out / "lift_vs_oracle.csv", index=False)
        oa = S[S.method == "oracle"][["alpha", "acc"]]
        for _, r in oa.iterrows():
            if r.acc / 100 < a.min_oracle_acc:
                print(f"WARNING alpha={r.alpha}: benign-only oracle clean accuracy {r.acc:.1f}% < {100*a.min_oracle_acc:.0f}% -> ASR is hard to interpret; train longer or raise lr before using these numbers")
    T = []
    for al, g in W.groupby(level=0):
        g = g.droplevel(0)
        for x, y in [("topf", "mad"), ("topf", "fedavg"), ("mad", "fedavg"), ("topf", "multikrum"), ("topf", "coord_median"), ("topf", "oracle")]:
            if x in g and y in g:
                d = (g[x] - g[y]).dropna()
                p = wilcoxon(d).pvalue if (d != 0).any() and len(d) > 5 else np.nan
                T.append(dict(alpha=al, a=x, b=y, n=len(d), mean_diff_pp=100 * d.mean(), p=p))
    T = pd.DataFrame(T)
    if len(T): T["p_holm"] = holm(T.p.fillna(1)); T.to_csv(out / "paired_tests.csv", index=False)
    rr = sorted(glob.glob(f"{a.inp}/rounds_shard*.csv"))
    if rr:
        R = pd.concat([pd.read_csv(f) for f in rr]); rows = []
        for (al, m), g in R.groupby(["alpha", "method"]):
            for ph, (lo, hi) in {"early_1_5": (1, 5), "mid_6_10": (6, 10), "late_11_15": (11, 15)}.items():
                x = g[g["round"].between(lo, hi)].groupby("seed").auroc.mean()
                rows.append(dict(alpha=al, method=m, phase=ph, n=len(x), auroc=x.mean(), lo=boot(x, rng)[0], hi=boot(x, rng)[1]))
        pd.DataFrame(rows).to_csv(out / "auroc_by_phase.csv", index=False)
    order = [m for m in ["fedavg", "coord_median", "trimmed_mean", "norm_clip", "multikrum", "fltrust", "mad", "topf", "oracle"] if m in set(S.method)]
    names = dict(fedavg="FedAvg", coord_median="Coord. median", trimmed_mean="Trimmed mean", norm_clip="Norm clipping", multikrum="Multi-Krum", fltrust="FLTrust", mad="MAD rule", topf="Top-$f$ rule", oracle="Oracle")
    alphas = sorted(S.alpha.unique())
    L = ["\\begin{tabular}{@{}l" + "c" * len(alphas) + "@{}}", "\\toprule", "Defence & " + " & ".join(f"$\\alpha{{=}}{x:g}$" for x in alphas) + "\\\\", "\\midrule"]
    for m in order:
        cells = []
        for al in alphas:
            r = S[(S.alpha == al) & (S.method == m)]
            cells.append("--" if r.empty else f"{r.asr.iloc[0]:.1f} [{r.asr_lo.iloc[0]:.1f}, {r.asr_hi.iloc[0]:.1f}]")
        L.append(names[m] + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; (out / "table_cnn.tex").write_text("\n".join(L))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 2.4), sharey=True)
    ax = np.atleast_1d(ax)
    for axi, al in zip(ax, alphas):
        s = S[S.alpha == al].set_index("method").reindex(order)
        axi.bar(range(len(order)), s.asr, yerr=[s.asr - s.asr_lo, s.asr_hi - s.asr], capsize=2, color="#888")
        axi.set_xticks(range(len(order))); axi.set_xticklabels([names[m].replace("$", "") for m in order], rotation=60, ha="right", fontsize=6)
        axi.set_title(f"alpha = {al:g}", fontsize=8); axi.set_ylabel("ASR (%)" if al == alphas[0] else "")
    fig.tight_layout(); fig.savefig(out / "fig_cnn.pdf"); print(S.round(2).to_string())

if __name__ == "__main__": main()
