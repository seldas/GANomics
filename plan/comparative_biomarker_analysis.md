# Plan: Comparative Biological Signal Preservation Analysis

## 1. Objective
To address Reviewer Comment (5), which requires comparing GANomics' biological fidelity against standard harmonization techniques (ComBat, TDM, CuBlock, etc.) in the context of Differential Expression Gene (DEG) overlap and pathway enrichment preservation.

## 2. Methodology
We will compare the following methods using the Neuroblastoma (NB) dataset with 50 training samples:
- **Ground Truth:** Real Microarray (MA) profiles.
- **Experimental Methods:**
    - GANomics (Full Model)
    - ComBat
    - TDM (Training Distribution Mapping)
    - CuBlock
    - YuGene
    - Quantile Normalization (QN)

Each method will translate RNA-seq (RS) test samples into the Microarray (MA) domain. We will then assess how well the biological signal (High-Risk vs. Low-Risk) is preserved in the translated MA space compared to the real MA space.

## 3. Required Data & Assets
- **Expression Data:** `dataset/NB/NB_AG.csv` (MA) and `dataset/NB/NB_NGS.csv` (RS).
- **Clinical Labels:** `dataset/NB/clinical_info.csv` (Expected columns: `sample_id`, `label` where label is High-Risk/Low-Risk).
    - *Note:* If missing, this must be generated from the SEQC clinical TXT file found in the original dataset.
- **Pathway Sets:** `dataset/RAW/h.all.v2025.1.Hs.symbols.gmt`.
- **Model:** `results/checkpoints/NB_Ablation_Size_50_Run_0/` (Latest checkpoint).

## 4. Implementation Steps

### Step 1: Baseline Synthesis
Run a modified version of `scripts/comparative_analysis.py` to save the full synthetic dataframes for each baseline method.
- **Task:** Update `scripts/comparative_analysis.py` to export CSVs for each baseline (e.g., `results/sync_data/baselines/NB_50_0/microarray_combat.csv`).

### Step 2: Comparative DEG Analysis
Develop/Run `scripts/comparative_biomarker.py` to:
1. Load Real Microarray DEGs (Reference).
2. Load Synthetic Microarray DEGs for **every** method.
3. Calculate **Jaccard Overlap Curves** for each method across top-$N$ genes (10 to 1000).
4. Save the results to `results/tables/Table_Fig9a_Comparative_Jaccard.csv`.

### Step 3: Pathway Enrichment Concordance
1. Run pathway enrichment (hypergeometric or GSEA) on the DEG lists for each method.
2. Compute the **Spearman Rank Correlation** of pathway p-values between Real MA and each Synthetic MA method.
3. Generate a comparative bar plot showing the Spearman $\rho$ for each algorithm.

### Step 4: Final Synthesis (Figure 9 Extension)
- Generate a new multi-panel figure:
    - **Panel A:** Jaccard Overlap Curve (GANomics vs. all baselines).
    - **Panel B:** Pathway Rank Concordance (Bar plot showing $\rho$ for all baselines).

## 5. Execution Commands (Once Environment is Ready)

```powershell
# 1. Generate full synthetic profiles for all baselines

python scripts/comparative_analysis.py --config configs/nb_config.yaml --sample_size 50 --save_full

## data structures
0. task identifier:
NB_Ablation_Size_50_Run_0

1. GANomics model checkpoint:
dashboard/backend/results/1_Training/checkpoints/NB_Ablation_Size_50_Run_0/net_latest.pth 
train_samples.txt is under the same folder.

2. Sync-data output template:
GANomics test: dashboard/backend/results/2_SyncData/NB_Ablation_Size_50_Run_0/test/microarray_fake.csv 
other algorithms: dashboard/backend/results/2_SyncData/NB_Ablation_Size_50_Run_0/algorithms/microarray_fake_combat.csv
combat; cublock; qn; tdm; yugene

3. comparative analysis result:
dashboard/backend/results/3_ComparativeAnalysis/NB_Ablation_Size_50_Run_0/Test_performance.csv

4. DEG results:
dashboard/backend/results/4_Biomarkers/DEG/NB_Ablation_Size_50_Run_0/Jaccard_Curve_ComBat.csv

5. Pathway results:
dashboard/backend/results/4_Biomarkers/Pathways/NB_Ablation_Size_50_Run_0/enriched_pathways_Combat.csv

6. Pred. Model results:
dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_0/Classifier_Performance_ComBat.csv

# 2. Run the comparative biomarker analysis
python scripts/biomarker.py \
    --real_A results/sync_data/NB_50_0/test/microarray_real.csv \
    --real_B results/sync_data/NB_50_0/test/rnaseq_real.csv \
    --syn_A results/sync_data/NB_50_0/test/microarray_fake.csv \
    --syn_B results/sync_data/NB_50_0/test/rnaseq_fake.csv \
    --gmt dataset/RAW/h.all.v2025.1.Hs.symbols.gmt

python scripts/biomarker.py \
    --real_A results/sync_data/NB_50_0/test/microarray_real.csv \
    --real_B results/sync_data/NB_50_0/test/rnaseq_real.csv \
    --syn_A results/sync_data/NB_50_0/algorithms/microarray_fake_tdm.csv \
    --syn_B results/sync_data/NB_50_0/algorithms/rnaseq_fake_tdm.csv

python scripts/biomarker.py \
    --real_A results/sync_data/NB_Sensitivity_Beta_0.0_Run_0/test/microarray_real.csv \
    --real_B results/sync_data/NB_Sensitivity_Beta_0.0_Run_0/test/rnaseq_real.csv \
    --syn_A results/sync_data/NB_Sensitivity_Beta_0.0_Run_0/test/microarray_fake.csv \
    --syn_B results/sync_data/NB_Sensitivity_Beta_0.0_Run_0/test/rnaseq_fake.csv


# 3. Generate plots
python scripts/plot.py --type comparative_fidelity --table results/tables/Comparative_Biological_Fidelity.csv
```

## 6. Expected Results
We anticipate that **GANomics** will exhibit the highest Jaccard overlap and Spearman rank concordance, as its non-linear translation capability and feedback-loss supervision better preserve the platform-independent biological signals compared to linear harmonization methods like ComBat.


