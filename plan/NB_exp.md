# Neuroblastoma (NB) Reproducibility Guide

This document provides a detailed walkthrough of the experiments performed on the Neuroblastoma dataset to reproduce the findings in the GANomics manuscript.

## Step 1: Data Preprocessing
Starting from raw GEO downloads to create the refined 10,042 gene dataset.

- **Inputs:**
  - `dataset/RAW/GSE49710_series_matrix.txt` (Microarray)
  - `dataset/RAW/GSE49711_series_matrix.txt` (RNA-seq)
  - `configs/preprocess_config.yaml`
- **Command:**
  ```bash
  python scripts/preprocess.py --config configs/preprocess_config.yaml
  ```
- **Expected Outputs:**
  - `dataset/NB/NB_A.csv`: Processed Microarray matrix (Samples x 10,042 Genes).
  - `dataset/NB/NB_B.csv`: Processed RNA-seq matrix (Samples x 10,042 Genes).
  - `results/tables/Table_1_Dataset_Overview.csv`: Statistics showing 498 samples and 10,042 common genes.

---

## Step 2: Core Model Training
Training the bidirectional translation model using the pair-aware feedback loss.

- **Inputs:**
  - `dataset/NB/NB_A.csv`
  - `dataset/NB/NB_B.csv`
  - `configs/nb_config.yaml`
- **Command:**
  ```bash
  python scripts/train.py --config configs/nb_config.yaml --name NB_Full_Training
  ```
- **Expected Outputs:**
  - `results/checkpoints/NB_Full_Training/net_latest.pth`: Trained model weights.
  - `results/logs/`: Training loss history showing convergence of GAN, Cycle, and Feedback losses.
- **Manuscript Alignment:** Supports **Figure 2** (Loss curves).

---

## Step 3: Ablation & Sensitivity Studies
Investigating model behavior across sample sizes and hyperparameter settings (beta and lambda).

### A. Sample Size Study
- **Command:**
  ```bash
  python scripts/run_ablation.py --config configs/nb_config.yaml --sizes 10 20 50 100 400
  ```

### B. Hyperparameter Sensitivity (Reviewer Request)
To examine the impact of feedback weight ($\beta$) and cycle weight ($\lambda$):
```bash
# Beta Sensitivity (varying feedback contribution)
python scripts/run_ablation.py --config configs/nb_config.yaml --betas 1.0 5.0 10.0 50.0

# Lambda Sensitivity (varying cycle consistency contribution)
python scripts/run_ablation.py --config configs/nb_config.yaml --lambdas 1.0 5.0 10.0 50.0
```
- **Expected Outputs:** Unique checkpoints in `results/checkpoints/` for each parameter value, allowing for comparative performance plotting.

---

## Step 4: Testing & Sync Data Generation
Generating synthetic profiles for the hold-out test set and benchmarking against legacy methods.

- **Inputs:**
  - `dataset/NB/NB_A.csv` / `NB_B.csv`
  - `results/checkpoints/NB_GANomics_400/net_latest.pth`
- **Command:**
  ```bash
  python scripts/test_sync.py --config configs/nb_config.yaml --checkpoint results/checkpoints/NB_GANomics_400/net_latest.pth
  ```
- **Expected Outputs:**
  - `results/tables/Table_2_Benchmarks.csv`: Pearson correlations (>0.96) and L1 errors.
  - Synthetic expression matrices used for visualization.
- **Manuscript Alignment:** Supports **Table 2** and **Figures 5, 6, 7**.

---

## Step 5: Biomarker Discovery & Modeling
Biological validation through DEG analysis, pathway enrichment, and cross-platform classification.

- **Inputs:**
  - `dataset/NB/NB_A.csv` (Real)
  - `results/tables/NB_syn_A.csv` (Synthetic)
  - `dataset/NB/clinical_info.csv` (Clinical labels: Favorable vs. Unfavorable)
  - `datasets/h.all.v2023.gmt` (Hallmark Gene Sets)
- **Command:**
  ```bash
  python scripts/biomarker.py --real_A dataset/NB/NB_A.csv --real_B dataset/NB/NB_B.csv --syn_A results/tables/NB_syn_A.csv --syn_B results/tables/NB_syn_B.csv --labels dataset/NB/clinical_info.csv --gmt datasets/h.all.v2023.gmt
  ```
- **Expected Outputs:**
  - `results/tables/Table_4_Classifier_Performance.csv`: Comparison of Real->Real, Real->Syn, Syn->Real scenarios (MCC scores).
  - `results/tables/Table_Fig9a_Jaccard_Curve.csv`: DEG overlap stats.
  - `results/tables/Table_Fig9_Null_Dist.csv`: Permutation test results for pathways.
- **Manuscript Alignment:** Supports **Table 4** and **Figures 9, 10**.

---

## Step 6: Figure Generation
Creating the final manuscript-ready visualizations.

- **Commands:**
  ```bash
  # Figure 7: t-SNE Alignment
  python scripts/plot.py --type tsne --real dataset/NB/NB_A.csv --syn results/tables/NB_syn_A.csv --out results/figures/Figure_7_tSNE.png

  # Figure 10: MCC Performance Bars
  python scripts/plot.py --type performance --table results/tables/Table_4_Classifier_Performance.csv --out results/figures/Figure_10_MCC.png
  ```
- **Expected Outputs:**
  - High-resolution images in `results/figures/`.
