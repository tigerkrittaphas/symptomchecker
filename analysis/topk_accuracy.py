"""Reproduce the main Top-K diagnostic accuracy table (manuscript Table 1).

Per application: Top-1..5 accuracy with 95% Wilson confidence intervals, and the
significance markers versus GPT-4o (exact McNemar, Holm-adjusted across all pairwise
comparisons within each K).
"""
import pandas as pd

from common import (APP_ORDER, gpt4o_holm_p, load_checked_response, output_dir,
                    topk_success_wide, wilson_ci)


def main():
    cr = load_checked_response()
    holm = gpt4o_holm_p(cr)

    rows = []
    for app in APP_ORDER:
        row = {"Symptom Checker": app}
        for k in range(1, 6):
            wide = topk_success_wide(cr, k)
            s = wide[app].dropna().astype(int)
            n, c = len(s), int(s.sum())
            acc = round(100 * c / n, 2)
            lo, hi = wilson_ci(c, n)
            sig = ""
            if app != "GPT-4o":
                p = holm.get((k, app))
                sig = "*" if (p is not None and p < 0.05) else ""
            row[f"Top-{k}"] = f"{acc:.2f}{sig}"
            row[f"Top-{k}_CI"] = f"({lo:.2f}-{hi:.2f})"
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("Top-K diagnostic accuracy (%). * = Holm-adjusted McNemar p<0.05 vs GPT-4o.\n")
    print(res[["Symptom Checker"] + [f"Top-{k}" for k in range(1, 6)]].to_string(index=False))

    out = output_dir()
    res.to_csv(f"{out}/top_k_accuracy.csv", index=False)
    print(f"\nSaved {out}/top_k_accuracy.csv")


if __name__ == "__main__":
    main()
