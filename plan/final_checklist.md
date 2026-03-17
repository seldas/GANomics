# Manuscript Final Checklist

Use this checklist every time before submitting or updating the GANomics manuscript. It captures the minimum steps that guarantee the figures, tables, and datasets referenced in R1 remain reproducible and documented.

## ✅ Data & Results
- [ ] Confirm `plan/aggregate_and_plot_results.py` pulls the latest `dashboard/backend/results_ms/` CSVs and regenerates any figures or intermediate tables.
- [ ] Verify every table/figure in the manuscript references a CSV inside `results_ms/` (e.g., `3_ComparativeAnalysis/Test_performance.csv`, `4_Biomarkers/DEG/Jaccard_Curve_GANomics.csv`). Update the README/architecture doc if additional files are used.
- [ ] Double-check `plan/test_sync_ms.py` and related scripts to ensure sync data exports (`microarray_fake.csv`, `rnaseq_fake.csv`, etc.) match the metrics reported in the paper.
- [ ] Re-run `plan/biomarker_batch.py` when key DEG/pathway/frequency thresholds change, and stash outputs in `results_ms/4_Biomarkers/` before submission.

## 🧪 Code & Configuration
- [ ] Confirm `dashboard/backend/main.py` points to the updated `results_ms/` directory when serving `/api/manuscript/*` endpoints.
- [ ] Ensure any new scripts introduced in `plan/` or `dashboard/backend/scripts/` have clear CLI args and log outputs (copy log files to `results_ms/1_Training/logs/` when relevant).
- [ ] Tag the commit that represents the manuscript version (e.g., `mnstrpt-v1`) so you can trace code/backends/plots to the exact state.

## 🧾 Documentation
- [ ] Update `README.md`, `architecture.md` (backend/frontend), and this checklist with any new directories, endpoint changes, or dataset sources referenced in the manuscript.
- [ ] Note in `README` or a release note which `results_ms/` run IDs produced each figure/table (e.g., `NB_Size_50_run_3 → Fig 4`).
- [ ] Share the QR code / tunnel info for `dashboard/frontend` if reviewers need live access (the architecture file now explains how to derive `API_BASE`).

## 📦 Submission Prep
- [ ] Package the canonical `results_ms/` folder (or its relevant CSV subset) with the manuscript bundle so reviewers replicate the numeric results.
- [ ] Archive any long-running CLI output (`plan/results/comparative_deg_curve.csv`, etc.) referenced in the paper under `plan/results/` or `dashboard/backend/results_ms/` before merging.
- [ ] Run `python plan/aggregate_and_plot_results.py` one final time and commit the generated artifacts alongside the checklist.

Add any extra items below as needed for the next submission round.
