"""Paired Top-K comparison of GPT-4o versus each SCA (manuscript Supplementary Table S3b).

For each K and SCA: paired absolute difference in Top-K accuracy (GPT-4o - SCA, percentage
points), its 95% confidence interval (normal-approximation CI for the difference of paired
proportions, i.e., the interval associated with the McNemar comparison), discordant-pair
counts, and the exact McNemar p-value (Holm-adjusted within each K across all pairwise
comparisons, matching the main Top-K table markers).
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from common import (APP_ORDER, gpt4o_holm_p, load_checked_response, output_dir,
                    topk_success_wide)

COMPARATORS = [a for a in APP_ORDER if a != "GPT-4o"]
Z = 1.959964  # 95% normal quantile


def paired_diff_ci(b, c, n, z=Z):
    """95% CI for the difference of paired proportions p1-p2 = (b-c)/n.

    Normal-approximation (Wald) interval based on the discordant counts, where b and c are
    the two discordant cells of the paired 2x2 table. This is the CI conventionally paired
    with the McNemar test.
    """
    diff = (b - c) / n
    se = np.sqrt(b + c - (b - c) ** 2 / n) / n
    return diff, diff - z * se, diff + z * se


def main():
    cr = load_checked_response()
    holm = gpt4o_holm_p(cr)

    rows = []
    for k in range(1, 6):
        wide = topk_success_wide(cr, k)
        for sca in COMPARATORS:
            pair = wide[["GPT-4o", sca]].dropna().astype(int)
            g, s = pair["GPT-4o"].to_numpy(), pair[sca].to_numpy()
            a = int(((g == 1) & (s == 1)).sum())
            b = int(((g == 1) & (s == 0)).sum())  # GPT correct only
            c = int(((g == 0) & (s == 1)).sum())  # SCA correct only
            d = int(((g == 0) & (s == 0)).sum())
            n = a + b + c + d
            diff, lo, hi = paired_diff_ci(b, c, n)
            p = float(mcnemar(np.array([[a, b], [c, d]]), exact=True, correction=False).pvalue)
            hp = holm[(k, sca)]
            rows.append({
                "K": k, "SCA": sca, "N": n,
                "SCA_acc": round(100 * (a + c) / n, 2),
                "GPT4o_acc": round(100 * (a + b) / n, 2),
                "Diff_pp": round(100 * diff, 2),
                "CI_low": round(100 * lo, 2), "CI_high": round(100 * hi, 2),
                "disc_GPT_only": b, "disc_SCA_only": c,
                "McNemar_p": p, "Holm_p": hp, "sig": "*" if hp < 0.05 else "",
            })

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    for k in range(1, 6):
        print(f"\n=== Top-{k}: GPT-4o vs each SCA (Diff = GPT-4o - SCA, pp) ===")
        print(res[res["K"] == k][[
            "SCA", "SCA_acc", "GPT4o_acc", "Diff_pp", "CI_low", "CI_high",
            "disc_GPT_only", "disc_SCA_only", "Holm_p", "sig"]].to_string(index=False))

    out = output_dir()
    res.to_csv(f"{out}/top_k_gpt4o_paired_diff.csv", index=False)
    print(f"\nSaved {out}/top_k_gpt4o_paired_diff.csv")


if __name__ == "__main__":
    main()
