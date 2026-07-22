"""Triage agreement: Cohen's kappa for the internist ceiling and each LLM vs physician (Table 4).

One combined table on the study's four-level triage scale. The first row is the reference
ceiling -- the two internists' independent pre-consensus labels against each other -- and the
remaining rows are each LLM's triage labels against the physician standard. Every row uses
Cohen's kappa (unweighted) with a 95% confidence interval from the McHugh (2012) asymptotic-SE
method, so the model rows are directly comparable to the human reliability ceiling.

Internist labels come from ``triage_labels_raw`` (via ``common.load_triage_label``). LLM labels
come from the promptfoo evaluation outputs under ``alignment/results/*.json``; the prompt's
five levels (1 emergency ... 5 self care) are collapsed to four by merging levels 4 and 5.

McHugh ML. Interrater reliability: the kappa statistic. Biochem Med (Zagreb). 2012;22(3):276-282.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import DATA, load_triage_label, output_dir

RESULTS = os.path.join(os.path.dirname(DATA), "alignment", "results")

# promptfoo result files to combine (base gemini/gpt-4o/gpt-5 run + GPT-5 reasoning sweep).
RESULT_FILES = ["results.json", "results_gpt5.json"]

# Provider id (+ reasoning-effort suffix for GPT-5) -> display name used in the manuscript.
MODEL_NAMES = {
    "azure:chat:gpt-4o-research": "GPT-4o",
    "azure:responses:gpt-5": "GPT-5, medium reasoning",
    "azure:responses:gpt-5_reasoning_high": "GPT-5, high reasoning",
    "azure:responses:gpt-5_reasoning_low": "GPT-5, low reasoning",
    "vertex:gemini-2.5-flash": "Gemini 2.5 Flash",
    "vertex:gemini-2.5-pro": "Gemini 2.5 Pro",
}

# Errored provider run (Vertex 404); excluded from the analysis.
DROP_MODELS = {"vertex:claude-sonnet-4"}


def kappa_ci_mchugh(y_true, y_pred, z=1.96):
    """Cohen's kappa and a McHugh (2012) asymptotic 95% CI: kappa +/- z * SE_kappa."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.union1d(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float(np.sum((cm.sum(axis=1) / n) * (cm.sum(axis=0) / n)))
    kappa = cohen_kappa_score(y_true, y_pred)
    se = np.sqrt(po * (1.0 - po) / (n * (1.0 - pe) ** 2))
    return kappa, max(-1.0, kappa - z * se), min(1.0, kappa + z * se), int(n)


def _row(name, y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    kappa, lo, hi, n = kappa_ci_mchugh(yt, yp)
    return {
        "Comparison": name,
        "Cohen's kappa": round(kappa, 3),
        "95% CI": f"[{lo:.3f}, {hi:.3f}]",
        "Exact agreement %": round((yt == yp).mean() * 100, 1),
        "Within-1 %": round((np.abs(yt - yp) <= 1).mean() * 100, 1),
        "N": n,
    }


def _parse_results(file_path):
    """Flatten one promptfoo JSON results file to rows of (physician, model-label) per case.

    GPT-5 was run at several reasoning efforts under the same provider id; the effort is
    recovered from the provider config via ``promptIdx`` and appended to the model id.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    provider_effort = {}
    providers = data.get("config", {}).get("providers") or data.get("results", {}).get("providers", [])
    for i, provider in enumerate(providers):
        provider_effort[i] = (provider["id"], provider.get("config", {}).get("reasoning_effort"))

    rows = []
    for result in data["results"]["results"]:
        v = result.get("testCase", {}).get("vars", {})
        model = result.get("provider", {}).get("id")
        idx = result.get("promptIdx")
        if idx in provider_effort:
            pid, effort = provider_effort[idx]
            model = f"{pid}_reasoning_{effort}" if effort and "gpt-5" in pid else pid
        resp = result.get("response", {})
        label = resp.get("output") if result.get("success") else resp.get("error")
        rows.append({"physician_triage": v.get("physician_triage"),
                     "llm_model": model, "llm_triage": label})
    return pd.DataFrame(rows)


def load_llm_labels():
    """Combined LLM/physician triage labels on the collapsed four-level scale."""
    df = pd.concat([_parse_results(os.path.join(RESULTS, f)) for f in RESULT_FILES],
                   ignore_index=True)
    df = df[~df["llm_model"].isin(DROP_MODELS)]
    df["llm_model"] = df["llm_model"].map(MODEL_NAMES)
    df = df.dropna(subset=["llm_model"])

    df["physician"] = pd.to_numeric(df["physician_triage"], errors="coerce")
    df["model"] = pd.to_numeric(df["llm_triage"], errors="coerce")
    df = df.dropna(subset=["physician", "model"]).copy()

    # Collapse level 5 (self care) into level 4 (routine appointment) -> four-level scale.
    df.loc[df["physician"] == 5, "physician"] = 4
    df.loc[df["model"] == 5, "model"] = 4
    return df


def main():
    rows = []

    # Reference ceiling: the two internists' independent pre-consensus labels.
    label = load_triage_label()
    rows.append(_row("Internists (ceiling)",
                     label["triage_1"].astype(int), label["triage_2"].astype(int)))

    # Each LLM vs the physician triage standard.
    llm = load_llm_labels()
    model_rows = [_row(m, d["physician"], d["model"])
                  for m, d in llm.groupby("llm_model", sort=False)]
    model_rows.sort(key=lambda r: r["Cohen's kappa"], reverse=True)
    rows.extend(model_rows)

    res = pd.DataFrame(rows)
    print("Triage agreement -- Cohen's kappa, four-level scale (McHugh 2012 95% CI)\n")
    print(res.to_string(index=False))

    out = os.path.join(output_dir(), "triage_agreement_kappa.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
