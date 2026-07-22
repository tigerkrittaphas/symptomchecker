# Enhancing Diagnostic Precision and Patient Triage in the Digital Age: A Quantitative Evaluation of LLM and Traditional Symptom Checker Applications

This repository contains the code and data for the study comparing the diagnostic accuracy
and triage performance of a Large Language Model (GPT-4o) against five publicly available
symptom checker applications (SCAs) using 213 clinical vignettes grounded in Thai clinical
practice.

## Repository layout

```
data/
  disease_list.txt                     Seed list of diagnoses used to generate vignettes
  case_info.csv                        Case attributes (disease, age, gender, body_systems,
                                       underlying_diseases)
  triage_label.csv                     Final physician consensus triage label per case
  vignettes/                           Structured case inputs, one JSON per disease, used to
                                       drive the LLM pipeline and SCA data collection

  benchmark/
    sca_llm_results_checked.xlsx       PRIMARY analysis file: every SCA's and GPT-4o's
                                       differential diagnoses in the exact order returned
                                       (ddx1..ddx5), the original triage wording, and the
                                       human-verified diagnosis-matching decision
                                       (location = 0-indexed rank of the correct diagnosis,
                                       -1 if absent; correctness flag).

  triage_mapping/                      Triage-mapping decisions (original wording -> 4 levels)
    Ada.csv Agnos.csv bouy.csv         Per-app maps: original_triage -> intermediate_triage
      doc-at-home.csv symptomate.csv
      GPT4o_preceptorAI.csv
    4_level_mapping.csv                intermediate_triage -> 4-level scale
                                       (emergency, 1 day, 1 week, routine appointment/self care)

  triage_labels_raw/                   Physician triage labelling, including pre-consensus
    case_vignette_pool1st.csv          Two internists' INDEPENDENT first-round labels
                                       (triage_1, triage_2) before consensus
    triage_diff_case_readjusted.csv    Re-adjudication of disagreements -> pool_result
                                       (consensus ground truth)

analysis/                              Statistical analysis scripts (reproduce all main tables)
  common.py                            Shared loader: rebuilds the per-case/per-app table
  topk_accuracy.py                     Main Top-K accuracy table (Table 1)
  topk_paired_diff.py                  GPT-4o vs each SCA: paired diffs, 95% CIs, discordant
                                       pairs, McNemar (Supplementary Top-K paired table)
  triage_decomposition.py             Under/over/exact-triage + non-under-triage rate (Table 3)
  triage_agreement_kappa.py           Cohen's kappa: internist ceiling + each LLM (Table 4)
  plot_topk_accuracy.py               Top-K accuracy figure with confidence intervals
  output/                              Generated CSVs

plots/                                 Generated figures (Top-K accuracy, PDF/PNG/TIFF)
vignette_generator.py                  Generates clinical vignettes from the disease list
case_information.py                    Extracts structured case attributes from vignettes
alignment/                             LLM triage-labelling (promptfoo eval that produces labels)
  promptfooconfig*.yaml prompt/        promptfoo eval config + prompt
  results/                             promptfoo eval outputs (results*.json), scored by
                                       analysis/triage_agreement_kappa.py
```

## Triage scale

App triage recommendations are mapped to a common four-level scale in two hops:

1. Each app's raw wording is normalised to an `intermediate_triage` phrase
   (`data/triage_mapping/<app>.csv`).
2. Each intermediate phrase maps to one of four standard levels
   (`data/triage_mapping/4_level_mapping.csv`):

   | Level | Numeric code |
   |---|---|
   | emergency | 1 |
   | 1 day | 2 |
   | 1 week | 3 |
   | routine appointment/self care | 4 |

This mapping is applied by `analysis/common.py`; unmappable wording is coded `0`.

## Reproducing the manuscript results

Requires Python >= 3.12. Install dependencies (declared in `pyproject.toml`):

```bash
uv sync            # or: pip install -e .
```

Run each analysis from the `analysis/` directory (outputs are written to `analysis/output/`):

```bash
cd analysis
uv run python topk_accuracy.py        # Table 1: Top-K accuracy + significance vs GPT-4o
uv run python topk_paired_diff.py     # Paired differences, 95% CIs, discordant-pair counts
uv run python triage_decomposition.py # Under/over/exact-triage, non-under-triage rate
uv run python triage_agreement_kappa.py # Table 4: internist-ceiling + per-LLM Cohen's kappa
uv run python plot_topk_accuracy.py   # Top-K accuracy figure -> ../plots/
```

`triage_agreement_kappa.py` reports both the internist reference ceiling and each LLM's
agreement with the physician standard in one table. It scores the LLM triage labels generated
by the promptfoo eval under `alignment/` (`alignment/results/*.json`); see `alignment/README.md`
for how those label files are produced.

### Which artifact reproduces which manuscript element

| Manuscript element | Script | Key inputs |
|---|---|---|
| Table 1 — Top-K diagnostic accuracy | `analysis/topk_accuracy.py` | `benchmark/sca_llm_results_checked.xlsx` |
| Top-K paired differences / CIs / discordant pairs (S3) | `analysis/topk_paired_diff.py` | same |
| Triage distribution & error decomposition (Tables 2-3) | `analysis/triage_decomposition.py` | benchmark file + `triage_mapping/` + `triage_labels_raw/` |
| Triage agreement — internist ceiling + LLM alignment (Table 4) | `analysis/triage_agreement_kappa.py` | `triage_labels_raw/case_vignette_pool1st.csv` + `alignment/results/*.json` |
| Top-K accuracy figure | `analysis/plot_topk_accuracy.py` | benchmark file |

## Statistical methods

- Proportions and 95% confidence intervals: Wilson score interval.
- Paired comparisons (GPT-4o vs each SCA): exact McNemar test on matched Top-K
  success/failure, Holm-adjusted across all pairwise comparisons within each K.
- Paired difference confidence intervals: normal approximation for the difference of paired
  proportions (the interval associated with the McNemar comparison).
- Triage: error decomposition into under-triage, over-triage, and exact-match
  (appropriateness) rates on the four-level scale, plus the non-under-triage rate.
- Triage agreement (internist ceiling and each LLM vs physician): Cohen's kappa (unweighted)
  with McHugh (2012) asymptotic 95% CIs, on the four-level scale (LLM triage levels 4 and 5
  collapsed), so model agreement is directly comparable to the human reliability ceiling.

## Data-collection provenance

Data were collected between 20 May 2024 and 20 July 2024. The specific access dates and the
version / interface state of each SCA during the study period are documented in the
manuscript's supplementary materials (Supplementary Table: "Symptom checker applications:
access dates and version / interface state"). The versions available during the study window
were used; SCAs were accessed via web browser or mobile application.

## Known limitations of the released data

- **GPT-4o inputs.** The manuscript describes GPT-4o input as a PreceptorAI STC
  question-and-answer transcript. The structured case inputs are provided under
  `data/vignettes/`. If the raw PreceptorAI Q&A transcripts are to be released verbatim,
  confirm and add them here.
- **Personal identifiers.** All vignettes are synthetic; no real patient data are included.
- **Secrets.** API keys and service-account credentials are excluded via `.gitignore`.
