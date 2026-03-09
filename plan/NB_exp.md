# Neuroblastoma (NB) Reproducibility Guide

This document provides a detailed walkthrough of the experiments performed on the Neuroblastoma dataset to reproduce the findings in the GANomics manuscript.

## Step 1: Data Preprocessing
Starting from raw GEO downloads to create the refined 10,042 gene dataset.

- **Command:**
  ```bash
  python scripts/preprocess.py --config configs/preprocess_config.yaml
  ```
- **Audit:** Checks `results/tables/Table_1_Dataset_Overview.csv` for correct sample/gene counts.

---

## Step 2: Core Model Training
Training the bidirectional translation model using the pair-aware feedback loss.

- **Command:**
  ```bash
  python scripts/train.py --config configs/nb_config.yaml --name NB_Full_Training
  ```

  ```bash
  python scripts/run_ablation.py --config configs/nb_config.yaml --sizes 50 --repeats 1 --gpu_ids 0,1,2,3,4,5,6,7
  ```
- **Audit:** Verify `net_latest.pth` and `train_samples.txt` are created in `results/checkpoints/NB_Full_Training/`.

---

## Step 3: Ablation & Sensitivity Studies
Investigating model behavior across sample sizes, hyperparameter settings, and directions.

### A. Parallel Sample Size Study (8-GPU example)
- **Command:**
  ```bash
  python scripts/run_ablation.py --config configs/nb_config.yaml --sizes 10 20 50 100 400 --repeats 5 --gpu_ids 0,1,2,3,4,5,6,7
  ```

### B. Hyperparameter Sensitivity
```bash
python scripts/run_ablation.py --config configs/nb_config.yaml --betas 1.0 5.0 10.0 50.0 --repeats 3 --gpu_ids 0,1,2,3,4,5,6,7
```

```bash
python scripts/run_ablation.py --config configs/nb_config.yaml --lambdas 1.0 5.0 10.0 50.0 --repeats 3 --gpu_ids 0,1,2,3,4,5,6,7
```

### C. One-Directional Architecture Ablation
```bash
python scripts/run_ablation.py --config configs/nb_config.yaml --direction AtoB --repeats 3 --gpu_ids 0,1,2,3,4,5,6,7
```

---

## Step 4: Testing & Sync Data Generation
Generating synthetic profiles. Automatically splits results into `train/` and `test/` based on the audit trail.

- **Command:**
  ```bash
  python scripts/test_sync.py --config configs/nb_config.yaml --project NB --sample_size 50 --run_id 0
  ```
- **Expected Outputs:** `results/sync_data/NB_50_0/test/microarray_fake.csv`, etc.

---

## Step 5: Comparative Analysis
Benchmarking GANomics against legacy bioinformatics methods.

- **Command:**
  ```bash
  python scripts/comparative_analysis.py --config configs/nb_config.yaml --project NB --sample_size 50
  ```
- **Expected Outputs:** `results/tables/Table_2_Comparison_NB.csv`.

---

## Step 6: Biomarker Discovery & Modeling
Biological validation through DEG analysis and cross-platform classification.

- **Command:**
  ```bash
  python scripts/biomarker.py --real_A dataset/NB/NB_A.csv --real_B dataset/NB/NB_B.csv --syn_A results/sync_data/NB_50_0/test/microarray_fake.csv --syn_B results/sync_data/NB_50_0/test/rnaseq_fake.csv --labels dataset/NB/clinical_info.csv
  ```

---

## Step 7: Figure Generation
Creating the final manuscript-ready visualizations.

- **Commands:**
  ```bash
  # Figure 7: t-SNE Alignment
  python scripts/plot.py --type tsne --real dataset/NB/NB_A.csv --syn results/sync_data/NB_50_0/test/microarray_fake.csv --out results/figures/Figure_7_tSNE.png

  # Figure 10: MCC Performance Bars
  python scripts/plot.py --type performance --table results/tables/Table_4_Classifier_Performance.csv --out results/figures/Figure_10_MCC.png
  ```
