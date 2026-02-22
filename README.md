# Enhancing Diagnostic Precision and Patient Triage in the Digital Age: A Quantitative Evaluation of LLM and Traditional Symptom Checker Applications.

This repository contains the code and data for the research paper titled "Enhancing Diagnostic Precision and Patient Triage in the Digital Age: A Quantitative Evaluation of LLM and Traditional Symptom Checker Applications." The study compares the diagnostic accuracy and triage capabilities of Large Language Models (LLMs) and traditional symptom checker applications.

## Dataset
- `data/disease_list.txt`: A initial list of diseases used as a seed for generating clinical vignettes
- `data/all_vignettes.csv`: contains the clinical vignettes used as patient represtnations for SCAs and LLM input in the study
- `data/case_info.csv`: Case information extracted from the vignettes, use for analysis
- `data/triage_label.csv`: Triage labels assigned by physicians for each case

## Code
- `vignette_generator.py`: Code for generating clinical vignettes using LLMs based on the disease list
- `case_information.py`: Code for extracting case information from generated clinical vignettes
- `alignment`: Directory containing all codes related to LLM automated labeling comparing to physician labels.