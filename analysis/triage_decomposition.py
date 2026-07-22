"""Triage error decomposition by symptom checker (manuscript Table 3 / S4).

Decomposes each recommendation versus the physician consensus into under-triage, exact
match (= appropriateness), and over-triage, plus the non-under-triage rate (= over + exact).
Triage is on the study's four-level scale; exact-match equals appropriateness and non-under
equals the non-under-triage rate.

The ordinal mean absolute error (MAE) treats the four-level scale as ordinal and averages
|recommendation - ground truth| per app, with a normal-approximation 95% CI (mean +/- 1.96*SE),
matching the interval style used elsewhere in the manuscript. Unmappable recommendations
(level 0) are excluded from the MAE only.

Exact McNemar tests are reported for under-triage and for exact-match appropriateness.
Following the Top-K table, p-values are Holm-adjusted within each metric across all pairwise
comparisons among the applications (the six-application family, 15 pairs); the GPT-4o-vs-SCA
subset is reported here. Non-under-triage is the complement of under-triage, so its paired
p-value equals under-triage's and is not repeated.
"""
import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from common import APP_ORDER, load_checked_response, output_dir, wilson_ci

Z = 1.959964  # 95% normal quantile
COMPARATORS = [a for a in APP_ORDER if a != "GPT-4o"]


def holm_mcnemar_vs_gpt4o(df, col):
    """Holm-adjusted exact-McNemar p on a per-case binary outcome.

    Adjusts across all 15 pairwise comparisons among the six applications (matching the
    Top-K table's family), then returns the GPT-4o-vs-each-SCA subset.
    """
    long = df[["Dx", "App", col]].copy()
    long[col] = long[col].astype(int)
    long = long.groupby(["Dx", "App"], as_index=False)[col].max()
    wide = long.pivot(index="Dx", columns="App", values=col)
    pairs = list(itertools.combinations(APP_ORDER, 2))
    raw = []
    for a, b in pairs:
        pr = wide[[a, b]].dropna().astype(int)
        tbl = (pd.crosstab(pr[a], pr[b])
               .reindex(index=[0, 1], columns=[0, 1], fill_value=0))
        raw.append(float(mcnemar(tbl.to_numpy(), exact=True, correction=False).pvalue))
    _, padj, _, _ = multipletests(raw, alpha=0.05, method="holm")
    adj = {frozenset(p): pa for p, pa in zip(pairs, padj)}
    return {s: adj[frozenset(("GPT-4o", s))] for s in COMPARATORS}


def main():
    cr = load_checked_response().copy()
    df = cr[cr["standard_triage_num"].notna() & cr["pool_result"].notna()].copy()
    df["app"] = df["standard_triage_num"]       # 0 = unmappable, 1=Emergency .. 4=Routine
    df["gt"] = df["pool_result"]                # ground-truth consensus level (1..4)

    df["non_under"] = df["app"] <= df["gt"]     # non-under-triage
    df["exact"] = df["app"] == df["gt"]         # appropriateness
    df["under"] = df["app"] > df["gt"]
    df["over"] = df["non_under"] & ~df["exact"]

    p_under = holm_mcnemar_vs_gpt4o(df, "under")
    p_exact = holm_mcnemar_vs_gpt4o(df, "exact")

    rows = []
    for app in APP_ORDER:
        g = df[df["App"] == app]
        n = len(g)
        u, ov, e, nu = g["under"].sum(), g["over"].sum(), g["exact"].sum(), g["non_under"].sum()
        # Ordinal MAE, excluding unmappable (level 0) recommendations.
        gg = g[g["app"] > 0]
        ae = (gg["app"] - gg["gt"]).abs().to_numpy(dtype=float)
        mae = ae.mean()
        se = ae.std(ddof=1) / np.sqrt(len(ae))
        rows.append({
            "App": app, "N": n, "N_MAE": len(ae),
            "Under%": round(100 * u / n, 2), "Under_CI": wilson_ci(u, n),
            "Under_Holm_p": p_under.get(app),
            "Over%": round(100 * ov / n, 2), "Over_CI": wilson_ci(ov, n),
            "Exact%": round(100 * e / n, 2), "Exact_CI": wilson_ci(e, n),
            "Exact_Holm_p": p_exact.get(app),
            "NonUnder%": round(100 * nu / n, 2), "NonUnder_CI": wilson_ci(nu, n),
            "MAE": round(mae, 3),
            "MAE_CI": (round(mae - Z * se, 3), round(mae + Z * se, 3)),
        })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print("Triage decomposition (Under + Exact + Over = 100; Non-under = Exact + Over)")
    print("MAE = ordinal mean absolute error on the 4-level scale (normal-approx 95% CI).")
    print("Holm p = exact McNemar vs GPT-4o, Holm-adjusted within metric across the 5 SCAs.\n")
    print(res.to_string(index=False))

    out = output_dir()
    res.to_csv(f"{out}/triage_decomposition.csv", index=False)
    print(f"\nSaved {out}/triage_decomposition.csv")


if __name__ == "__main__":
    main()
