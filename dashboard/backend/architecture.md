# GANomics Backend Architecture

This folder hosts the FastAPI service that powers the GANomics dashboard. It orchestrates dataset ingestion, training orchestration, analysis steps, and the delivery of both interactive dashboard data and manuscript-grade records.

## Core Structure

```
dashboard/backend/
├── dataset/           # Per‑project raw inputs (df_ag/df_rs, metadata, labels, samples/genelist exports)
├── results/           # Live experiment outputs used by the dashboard (training, sync, comparative, DEG, pathway, prediction, figures)
├── results_ms/        # Manuscript-grade CSV exports (canonical figures/tables for the paper)
├── scripts/           # Standalone Python jobs (run_ablation, test_sync, comparative/deg/pathway/prediction analysis, inference, helpers)
├── src/               # Core GANomics library (models, layers, datasets, utils, database helpers)
├── main.py            # FastAPI entry point wiring HTTP endpoints, orchestrating scripts, exposing run status + manuscript APIs
├── migrate_configs.py # Utility for updating legacy project configs
├── generate_samples.py# Sample-generation helper called by the frontend
└── temp/              # Scratch directory for downloads/exports
```

## Entry Point (`main.py`)

- `app = FastAPI(...)` configures CORS + uvicorn server (`python main.py` launches on port 8832).
- Defines absolute paths (`BACKEND_DIR`, `DATASET_DIR`, `RESULTS_DIR`, `RESULTS_MS_DIR`, etc.) so every helper can pivot off the backend root.
- Provides client-facing endpoints such as:
  - `/api/projects` → list available datasets, metadata, sample/genes counts.
  - `/api/projects/create` → uploads paired Microarray/RNA-seq files, seeds config, label, sample/genelist exports.
  - `/api/runs/{run_id}/...` → start/stop training (`/api/train`), run analysis steps (`/api/runs/{run_id}/run_step`), stream logs, download sync/DEG/comparative/pathway/prediction assets.
  - `/api/results` → dashboard status view for training sync/DEG/comparative updates.
  - Manuscript surface: `/api/manuscript/tasks`, `/api/manuscript/download/{...}`, `/api/manuscript/logs/{run_id}` to browse the `results_ms/` tier.
- Uses `subprocess.Popen` to launch scripts (with `PYTHONPATH` pointing back at the backend) so long-running jobs run outside FastAPI’s loop.
- A tiny process tracker (`running_processes`) enables clean shutdown with `psutil` helper `kill_proc_tree`.

## `dataset/`

Every project lives here under `dataset/<project_id>/`. Each project directory contains:
- `<project>_config.yaml`: training hyperparameters, metadata, dataset paths (the dashboard writes these through `/api/projects/create`).
- `df_ag.tsv`, `df_rs.tsv`: Matrices for Microarray (A) and RNA-seq (B).
- Optional `label.txt`, `samples.tsv`, `genelist.tsv`, and external testing folders `ext_<id>/` with their own metadata.

AI navigation tip: start by scanning `dataset/` names to understand available cohorts, then open the matching `_config.yaml` to see model settings before running scripts.

## `results/` vs `results_ms/`

This backend maintains two parallel result tiers:

1. **`results/` (dashboard live data)**
   - `1_Training`: training checkpoints + logs.
   - `2_SyncData`: synthetic matrix outputs per run + external syncs.
   - `3_ComparativeAnalysis`: multi-method performance (GANomics vs QN/TDM/etc.) behind `Test_performance.csv`.
   - `4_Biomarkers`: DEG (`Jaccard_Curve_*`), Pathway (`Pathway_*`), Prediction (`Classifier_Performance_*`).
   - `5_Figures`: aggregated image exports consumed by the UI.

2. **`results_ms/` (manuscript-grade records)**
   - Mirrors the dashboard structure but is treated as the record for the manuscript (§ in README). These CSVs are the canonical data for tables/figures cited in the paper. `/api/manuscript/...` endpoints expose them for downloads.

Guidance: When an AI assistant needs to reproduce or inspect published figures, always prefer `results_ms/` for deterministic outputs. Any fresh runs should be copied into `results_ms/` before referencing the manuscript.

## `scripts/`

Scripts are modular CLI jobs invoked by `main.py`:

- `run_ablation.py`: orchestrates size/beta/lambda sweeps and writes logs/checkpoints to `results/1_Training`.
- `test_sync.py`: generates synthetic data, t-SNE, etc.
- `comparative_analysis.py`: generates `Test_performance.csv` across algorithms.
- `deg_analysis.py`, `pathway_analysis.py`, `prediction_analysis.py`: compute biomarker metrics stored under `results/4_Biomarkers` or `results_ms/4_Biomarkers`.
- `inference.py`: translates external datasets using saved checkpoints.
- `train.py`, `preprocess.py`, utilities for data conversion, sample generation, gene mapping, etc.

AI pointer: When tasked with adding a new feature or debugging, search this folder by step name (e.g., `deg_analysis`, `inference`) to see the command-line contract used by `main.py`.

## `src/`

This is the reusable GANomics core library:

- `datasets/`: data loaders, balanced pairing utilities.
- `models/`, `layers/`, `layers_compatible/`: GAN/translator architectures + helper blocks.
- `core/`: training loop, loss definitions, checkpoint management.
- `bio_utils/`: pathway/DEG helpers, enrichment calculators.
- `db/`: result persistence helpers (if any) for later UI queries.

When AI needs to reason about model behavior, start within `src/` and tie it back to `scripts/run_ablation.py` or `train.py`—those scripts import from `src` to execute training.

## Navigation Advice for AI Strategists

1. **Start at `main.py`** to understand which API endpoints orchestrate which scripts and directories. The `scripts` → `results` relationship is spelled out in `run_step`.
2. **Follow the data flow**: `dataset/` inputs → `scripts/run_ablation.py` → `results/1_Training` → downstream analysis scripts → `results/2-5`, then optionally copy into `results_ms/` for manuscript exports.
3. **Use `results_ms/` as canonical evidence** when reasoning about published tables/figures. The `README` now states this folder is the record for the manuscript.
4. **Search for `<run_id>`** inside `results/` when tracing a specific experiment; `main.py` exposes `run_statuses` by looking at logs/checkpoints.
5. **For new API endpoints**, update `main.py` and add CLI scripts as needed, then update `architecture.md`/README so humans and AI can find them.

Keeping these landmarks in mind makes it easier for you to craft new FastAPI routes, hook in training scripts, or explain how to extract results for downstream reporting.