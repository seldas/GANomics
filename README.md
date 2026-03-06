# GANomics: Bridging Legacy and Modern Transcriptomic Platforms

GANomics is a Generative Adversarial Network (GAN) framework designed for bidirectional translation between legacy (Microarray) and modern (RNA-seq) transcriptomic platforms. By integrating paired and unpaired samples through a novel **pair-aware feedback loss**, GANomics enforces one-to-one transcript mappings while preserving global gene expression distributions.

This repository contains the refactored, modular Python implementation supporting the methodologies and results described in the manuscript: **"GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications"**.

---

## 🚀 Key Features
- **Bidirectional Translation:** Seamlessly convert Microarray to RNA-seq and vice versa.
- **Data Efficiency:** Achieve high per-sample correlations (>0.96) with as few as 10–50 paired profiles.
- **Biological Fidelity:** Preserves Differential Expressed Genes (DEGs), pathway-level rankings, and clinical classifier performance.
- **Reproducible Pipeline:** Pure Python implementation with systematic result storage in `results/`.

---

## 📁 Project Structure

```text
GANomics/
├── configs/            # YAML configurations for experiments (e.g., nb_config.yaml)
├── data/               # Raw and processed genomic datasets
├── results/            # Systematic output storage
│   ├── checkpoints/    # Trained model weights (.pth)
│   ├── figures/        # Manuscript-ready plots (PDF/PNG)
│   ├── tables/         # Generated CSV/Excel stats for the paper
│   └── logs/           # Training and evaluation logs
├── src/                # Core GANomics Package
│   ├── bio_utils/      # Normalization (ComBat, QN, TDM, YuGene, CuBlock)
│   ├── core/           # Training loops, evaluation metrics, and analysis logic
│   ├── datasets/       # Genomics-specific data loaders and preprocessing
│   ├── layers/         # GANomics architecture (1x1 Conv dense layers)
│   └── models/         # GANomicsModel implementation with Feedback Loss
└── scripts/            # Executable Entry Points (Pipeline Steps 1-4)
```

---

## 🛠️ Pipeline & Manuscript Alignment

The following four scripts form the core GANomics pipeline. Each script is directly responsible for generating specific results found in the manuscript.

### 1. Data Preprocessing (`scripts/preprocess.py`)
**Function:** Normalizes raw GEO/Synapse data and aligns Microarray/RNA-seq platforms.
- **Manuscript Support:** 
  - Generates **Table 1** (Dataset Overview for NB, TCGA, NCI-60, METSIM).
  - Generates **Figure S1** (Normalization impact on METSIM).

### 2. Model Training (`scripts/train.py`)
**Function:** Executes GANomics training using the pair-aware feedback loss. Supports variable sample sizes (10, 50, 100).
- **Manuscript Support:** 
  - Generates **Figure 2** (Loss convergence and L1 distance monitoring).
  - Supports **Figure 3** (Sample size requirement analysis).
  - Supports **Figure 4** (Benchmarking against standard CycleGAN).

### 3. Testing & Sync Generation (`scripts/test_sync.py`)
**Function:** Generates synthetic "Sync Data" and benchmarks against legacy methods (TDM, QN, ComBat, etc.).
- **Manuscript Support:** 
  - Generates **Table 2 & 3** (Global performance metrics).
  - Generates **Figure 5 & 6** (Concordance and cross-dataset generalization).
  - Generates **Figure 7 & S2** (t-SNE co-localization plots).
  - Generates **Figure 8** (PAM50 clinical signature alignment).

### 4. Biomarker Discovery & Modeling (`scripts/biomarker.py`)
**Function:** Performs DEG analysis, Pathway Enrichment, and Cross-Platform Classifier validation.
- **Manuscript Support:** 
  - Generates **Figure 9** (DEG overlaps and KEGG/Hallmark pathway enrichment).
  - Generates **Figure 10** (Random Forest MCC performance histograms).
  - Generates **Table 4 & S1** (Classifier diagnostic metrics: Recall, Precision, MCC).

---

## ⚙️ Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### 1. Preprocess Data
```bash
python scripts/preprocess.py --datasets NB METSIM
```

### 2. Train GANomics
```bash
python scripts/train.py --config configs/nb_config.yaml
```

### 3. Evaluate and Generate Sync Data
```bash
python scripts/test_sync.py --config configs/nb_config.yaml --checkpoint results/checkpoints/NB_GANomics_50/net_latest.pth
```

### 4. Run Biomarker Analysis
```bash
python scripts/biomarker.py --real_A data/processed/NB_A.csv --syn_A results/tables/NB_syn_A.csv --labels dataset/NB/clinical_info.csv
```

---

## 📝 Citation
If you use GANomics in your research, please cite:
> Wu, L., Salman, H., & Tong, W. (2026). GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications. (Manuscript R1).
