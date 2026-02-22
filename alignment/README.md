# Symptom Checker Physician-LLM Alignment Check

This repo contains code that check LLM grading alignment to physician gold-standard answer.

This expect "all_vignettes.csv" in ../data/

Use promptfoo with VertexAI to run, using service account json in secret/

```
export GOOGLE_APPLICATION_CREDENTIALS="path_to_json_file"
promptfoo eval --output results/results.json --output results/results/csv
promptfoo view #optionaL
```