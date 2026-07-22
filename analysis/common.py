"""Shared data-loading pipeline for the symptom-checker benchmark analyses.

Rebuilds the per-case / per-application table used by all main-table analyses from the
files under ``stc_code/data``. This reproduces the pipeline in the study's analysis
notebook so that the published Top-K and triage numbers can be regenerated end-to-end.

Outputs a tidy DataFrame ``checked_response`` with, per (case, app):
  - location   : 0-indexed rank of the correct diagnosis in the app's ordered list
                 (-1 if the correct diagnosis is absent); from the human-checked file
  - standard_triage_num : app triage recommendation mapped to the 4-level scale
                 (1=Emergency, 2=1-day, 3=1-week, 4=Routine/self-care; 0=unmappable)
  - pool_result: physician consensus triage level (ground truth)

App name convention: 'GPT-4o' is the LLM operating on PreceptorAI-collected input (labelled
'GPT-4o' in the benchmark file; the native PreceptorAI STC rows are not part of this analysis).
"""
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

APP_ORDER = ["Ada", "Agnos", "Buoy", "DoctorAtHome", "GPT-4o", "Symptomate"]

_RENAME_APP = {
    "Ada Health": "Ada", "Agnos": "Agnos", "Buoy Health": "Buoy",
    "DoctorAtHome": "DoctorAtHome", "Symptomate (Infermedica)": "Symptomate",
}

_STANDARD_TRIAGE_NUM = {
    "emergency": 1, "1 day": 2, "1 week": 3, "routine appointment/self care": 4,
}


def load_triage_label():
    """Physician triage labels: pre-consensus (triage_1/triage_2) and consensus (pool_result)."""
    label = pd.read_csv(os.path.join(DATA, "triage_labels_raw", "case_vignette_pool1st.csv"))
    readj = pd.read_csv(os.path.join(DATA, "triage_labels_raw", "triage_diff_case_readjusted.csv"))
    readj = readj[["disease", "triage_s_final", "triage_w_final", "pool_result"]].dropna()
    label = label.merge(readj, on="disease", how="left")
    label["pool_result"] = label["pool_result"].fillna(label["triage_1"]).astype(int)
    return label


def _load_triage_maps():
    tm = os.path.join(DATA, "triage_mapping")
    standard_map = pd.read_csv(os.path.join(tm, "4_level_mapping.csv"))
    files = {
        "Ada": "Ada.csv", "Agnos": "Agnos.csv", "Buoy": "bouy.csv",
        "DoctorAtHome": "doc-at-home.csv", "Symptomate": "symptomate.csv",
        "GPT-4o": "GPT4o_preceptorAI.csv",
    }
    maps = {}
    for app, fn in files.items():
        m = pd.read_csv(os.path.join(tm, fn))
        maps[app] = m.merge(standard_map, on="intermediate_triage", how="left")
    return maps, standard_map


def load_checked_response():
    """Build the per-case / per-app table (human-checked diagnosis matching + triage mapping)."""
    case_info = pd.read_csv(os.path.join(DATA, "case_info.csv"))
    filter_cases = list(case_info["disease"].unique())

    cr = pd.read_excel(
        os.path.join(DATA, "benchmark", "sca_llm_results_checked.xlsx"), header=0)
    cr = cr[["Dx", "App", "ddx1", "ddx2", "ddx3", "ddx4", "ddx5",
             "triage", "location", "correctness"]].copy()
    cr["Dx"] = cr["Dx"].str.strip().str.lower()
    cr = cr.loc[cr["Dx"].isin(filter_cases)]
    cr["App"] = cr["App"].replace(_RENAME_APP)

    maps, standard_map = _load_triage_maps()

    def to_intermediate(row):
        m = maps.get(row["App"])
        if m is None:
            return None
        hit = m[m["original_triage"].str.lower() == str(row["triage"]).strip().lower()]
        return hit.iloc[0]["intermediate_triage"] if not hit.empty else None

    cr["intermediate_mappings"] = cr.apply(to_intermediate, axis=1)

    def to_standard(row):
        hit = standard_map[standard_map["intermediate_triage"] == row["intermediate_mappings"]]
        return hit.iloc[0]["4_level_triage"] if not hit.empty else None

    cr["standard_triage"] = cr.apply(to_standard, axis=1)
    cr["standard_triage_num"] = (
        cr["standard_triage"].map(_STANDARD_TRIAGE_NUM).fillna(0).astype(int))

    label = load_triage_label()
    cr = cr.merge(label[["disease", "pool_result"]],
                  left_on="Dx", right_on="disease", how="left")
    return cr


def wilson_ci(count, nobs, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return round(lo * 100, 2), round(hi * 100, 2)


def topk_success_wide(cr, k, apps=APP_ORDER):
    """Wide table (index=case, columns=app) of Top-K success (1/0) for the given k."""
    d = cr[["Dx", "App", "location"]].copy()
    d = d[d["App"].isin(apps)]
    d["succ"] = ((d["location"] >= 0) & (d["location"] < k)).astype(int)
    d = d.groupby(["Dx", "App"], as_index=False)["succ"].max()
    return d.pivot(index="Dx", columns="App", values="succ")


def topk_pairwise_holm(cr, apps=APP_ORDER):
    """All pairwise exact-McNemar comparisons per K with Holm adjustment within each K.

    This is the multiple-comparison family used for the significance markers in the main
    Top-K table (Holm applied across all pairwise comparisons within each K).
    Returns a long DataFrame with one row per (K, App A, App B).
    """
    from itertools import combinations
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    rows = []
    for k in range(1, 6):
        wide = topk_success_wide(cr, k, apps)
        pvals, idx = [], []
        for a, b in combinations(apps, 2):
            pair = wide[[a, b]].dropna().astype(int)
            tbl = (pd.crosstab(pair[a], pair[b])
                   .reindex(index=[0, 1], columns=[0, 1], fill_value=0))
            p = float(mcnemar(tbl.to_numpy(), exact=True, correction=False).pvalue)
            rows.append({"K": k, "App A": a, "App B": b,
                         "n10": int(tbl.loc[1, 0]), "n01": int(tbl.loc[0, 1]),
                         "McNemar_p": p})
            pvals.append(p)
            idx.append(len(rows) - 1)
        _, padj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for i, pa in zip(idx, padj):
            rows[i]["Holm_p"] = float(pa)
    return pd.DataFrame(rows)


def gpt4o_holm_p(cr, apps=APP_ORDER):
    """Holm-adjusted McNemar p-values for GPT-4o vs each SCA, keyed by (K, SCA).

    Uses the same 15-pairwise-per-K family as ``topk_pairwise_holm`` so the p-values match
    the significance markers reported in the main Top-K table.
    """
    pw = topk_pairwise_holm(cr, apps)
    g = pw[(pw["App A"] == "GPT-4o") | (pw["App B"] == "GPT-4o")].copy()
    g["SCA"] = g.apply(lambda r: r["App B"] if r["App A"] == "GPT-4o" else r["App A"], axis=1)
    return {(int(r["K"]), r["SCA"]): float(r["Holm_p"]) for _, r in g.iterrows()}


def output_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(d, exist_ok=True)
    return d
