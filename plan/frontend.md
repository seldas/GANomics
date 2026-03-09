# GANomics Web Dashboard: Complete Lifecycle Plan

## 1. Goal
To provide a simple, intuitive web interface for the entire GANomics workflow: from raw data training to biological validation and comparative reporting.

## 2. Technical Stack
- **Frontend:** React (Vite) + Vanilla CSS.
- **Backend:** FastAPI (Python).
- **Execution:** Wrapper for `scripts/` (train, test_sync, comparative_analysis, biomarker).

---

## 3. UI Modules

### Module 1: Training & Ablation (Step 1)
- **Data Selection:** Select project (NB, NCI60, TCGA) or upload `ag.tsv`/`rs.tsv`.
- **Parameter Matrix:** Chip-based multi-select for Sample Sizes, Beta, Lambda, and Repeats.
- **Smart Monitor:** Scan `results/checkpoints/` to highlight existing runs. Real-time log streaming from `results/logs/`.

### Module 2: Post-Training Generation (Step 2)
- **Script:** `test_sync.py`.
- **Function:** Generate `microarray_fake.csv` and `rnaseq_fake.csv` for trained models.
- **UI:** A "Generate" button on completed job cards. Shows train/test split audit.

### Module 3: Comparative Benchmarking (Step 3)
- **Script:** `comparative_analysis.py`.
- **Function:** Benchmark GANomics vs ComBat, TDM, QN, YuGene, and CuBlock.
- **UI Visualization:**
    - **Performance Table:** Sortable table of Pearson, Spearman, and MAE metrics.
    - **Scatter Explorer:** Select a gene to see Real vs. Syn correlation across all methods.

### Module 4: Biological Validation (Step 4)
- **Script:** `biomarker.py`.
- **Input:** Clinical labels (`SEQC_NB_...txt`) and Gene Sets (`h.all.gmt`).
- **Sub-Modules:**
    - **DEG Analysis:** Interactive **Jaccard Overlap Curve** (Figure 9a).
    - **Pathway Concordance:** Bar chart of **Spearman Rho** comparing pathway ranks across algorithms.
    - **Cross-Platform RF:** Bar chart comparing **MCC** scores for `Real->Syn` and `Syn->Real` scenarios.

---

## 4. Logical Workflow & Data Mapping

| Phase | Action | Input | Output Folder |
| :--- | :--- | :--- | :--- |
| **1. Train** | Run `train.py` | `dataset/{Project}/` | `results/checkpoints/` |
| **2. Sync** | Run `test_sync.py` | `results/checkpoints/` | `results/sync_data/` |
| **3. Compare**| Run `comparative_analysis.py`| `results/sync_data/` | `results/tables/` (Table 2) |
| **4. Bio** | Run `biomarker.py` | `results/sync_data/` | `results/tables/` (Table 4, Fig 9) |

## 5. Implementation Priorities
1. **Dashboard Frame:** Project selection and basic filesystem navigation.
2. **Training Matrix:** Integrating `run_ablation.py` with real-time log streaming.
3. **Evaluation Pipeline:** Sequential triggering of Sync -> Compare -> Biomarker.
4. **Visualizations:** Integrating Chart.js or D3 for Jaccard and MCC plots.
