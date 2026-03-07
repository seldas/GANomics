# GANomics Project Restructuring Plan

This document outlines the plan to refactor the GANomics project from its current state (inherited from CycleGAN boilerplate) into a clean, modular, and genomics-focused codebase. This refactoring supports the methodologies described in `plan\GANomics_R1.docx`.

## 1. Current State (old_ver/)

The `old_ver` directory contains the legacy implementation, which is heavily based on the PyTorch-CycleGAN-and-pix2pix framework.

### Root Files
- `train_new.py`: The main training script. It includes manual overrides for genomic experiments and a robust training loop with "loss explosion" detection and auto-recovery.
- `GANomics_Common.ipynb`, `DEGs_Common.ipynb`, `ComparativeAnalysis_Common.ipynb`, `NBFav_DEGs.ipynb`, `NBFav_KNNs.ipynb`: Jupyter notebooks used for downstream genomic analysis, differential expression analysis (DEGs), and KNN-based evaluation.

### Directories

#### `old_ver/data/`
Contains data loading logic. Many files are boilerplate from image-to-image translation and are likely unused for genomics.
- `text_folder.py`: Likely the primary loader for genomic text/CSV data.
- `unalignedpaired_dataset.py`: Handles unpaired data across domains.
- `base_dataset.py`, `aligned_dataset.py`, `unaligned_dataset.py`: Core dataset classes.
- `colorization_dataset.py`, `image_folder.py`, `single_dataset.py`, `template_dataset.py`, `visualize.py`: Boilerplate from the original framework.

#### `old_ver/model/`
Contains model architectures and training logic.
- `MMD_cycle_gan_model.py`: CycleGAN with Maximum Mean Discrepancy (MMD) loss.
- `trans_cycle_gan_model.py`: Transformer-based CycleGAN variant.
- `cycle_gan_model.py`: Standard CycleGAN implementation.
- `networks.py`: Contains definitions for Generators and Discriminators.
- `base_model.py`: Abstract base class for all models.
- `test_model.py`: Model for inference/testing.
- `unet/`: U-Net sub-components (likely for image-based baselines).

#### `old_ver/options/`
Handles command-line arguments using `argparse`.
- `base_options.py`, `train_options.py`, `test_options.py`: Configuration management.

#### `old_ver/util/`
A mix of generic utilities and specialized bioinformatics methods.
- **Bioinformatics/Genomics:** `ComBat.py`, `CuBlock.py`, `QN.py` (Quantile Normalization), `TDM.py` (Training Distribution Matching), `YuGene.py`.
- **Boilerplate/Utilities:** `html.py` (HTML reporting), `image_pool.py`, `visualizer.py` (Visdom integration), `util.py`, `get_data.py`.

---

## 2. Identified Issues

1.  **Redundant Boilerplate:** High volume of unused image-specific code (e.g., `image_folder.py`, `html.py`) makes the codebase difficult to navigate.
2.  **Coupled Logic:** Training logic (manual overrides) is hardcoded into `train_new.py`, making it difficult to run different experiment variants without manual edits.
3.  **Scattered Analysis:** Notebooks are in the same folder as scripts, and core bioinformatics utilities are mixed with UI/boilerplate utilities.
4.  **Configuration Management:** Argparse is used for complex experiment steering, which can be prone to error compared to structured config files (YAML/JSON).

---

## 3. Proposed New Structure (GANomics v2)

The goal is to move towards a structure that separates "framework logic" from "domain logic."

```text
GANomics/
├── configs/            # YAML/JSON configurations for different experiments
├── data/               # Raw and processed datasets (kept separate from code)
├── plan/               # Research papers and restructuring plans
├── src/                # Core implementation
│   ├── datasets/       # Genomics-specific data loaders (CSV, TSV, AnnData)
│   ├── models/         # Clean implementations of GANomics, MMD-GAN, etc.
│   ├── layers/         # Shared network components (Transformers, FC layers)
│   ├── core/           # Training loops, loss functions, and engine logic
│   ├── bio_utils/      # Domain-specific logic (ComBat, QN, TDM, YuGene)
│   └── utils/          # Generic logging, checkpointing, and visualization
├── scripts/            # Entry point scripts
│   ├── preprocess.py   # (1) Data Pre-process
│   ├── train.py        # (2) Training
│   ├── test_sync.py    # (3) Testing / Sync Data
│   └── biomarker.py    # (4) Biomarker / Modeling
├── results/            # Systematic output storage
│   ├── checkpoints/    # Model weights
│   ├── figures/        # Manuscript-ready plots (PDF/PNG)
│   ├── tables/         # CSV/Excel stats for paper
│   └── logs/           # Training and evaluation logs
├── requirements.txt    # Updated dependencies
└── README.md           # Project overview and setup instructions
```

---

## 4. Manuscript-Driven Script Mapping

To ensure full reproducibility and to completely replace Jupyter notebooks with pure Python code, the four main pipeline functions are mapped directly to specific scripts. These scripts will automatically save output tables and figures to the `results/` folder, directly supporting `GANomics_R1.docx`.

### (1) Data Pre-process (`scripts/preprocess.py`)
- **Function:** Normalizes raw data downloaded from GEO/Synapse, matches microarray and RNA-seq pairs via platform mapping, and applies necessary scaling (e.g., log2 transformation for RNA-seq FPKM/TPM). 
- **Manuscript Support:** 
  - **Table 1:** Generates dataset overviews (sample counts, mapped genes for NB, TCGA, NCI-60, METSIM).
  - **Figure S1:** Demonstrates log2 vs. non-normalized FPKM transformations on METSIM.

### (2) Training & Performance (`scripts/train.py`)
- **Function:** Executes GANomics core training, handles varying training sample sizes (e.g., 10 to 400), tracks iterative loss convergence (generator, discriminator, cycle consistency, feedback), and saves checkpoints.
- **Manuscript Support:**
  - **Figure 2 (a-f):** Loss curves tracking and baseline L1 distance performance monitoring over epochs.
  - **Figure 3 (a-c):** Performance benchmarking using differing numbers of paired subsets (e.g., 10-50, 50-200, 100-400).
  - **Figure 4 (a-g):** Generates comparative training performance metrics and feature-space representations against standard CycleGAN.

### (3) Testing / Sync Data Generation (`scripts/test_sync.py`)
- **Function:** Applies the trained GANomics model to hold-out test sets to generate synthetic expression profiles. Computes robust comparative benchmarks (Pearson, Spearman, L1, L2, MAE, RMSE) against baseline approaches (TDM, QN, ComBat, YuGene, CuBlock).
- **Manuscript Support:**
  - **Table 2 & 3:** Comparison benchmarks vs other algorithms and sample-size evaluations.
  - **Table S2:** Comprehensive benchmarking across additional test datasets (TCGA, NCI-60, METSIM).
  - **Figure 5 (a-f):** Expression profile concordance plots, Pearson distribution maps, and general correlation between real and synthetic profiles.
  - **Figure 6:** Cross-dataset generalization metrics.
  - **Figure 7 & Figure S2:** t-SNE clustering visualizations of true vs synthetic datasets for BRCA, NCI60, and METSIM.
  - **Figure 8:** PAM50 clinical signature alignment using correlation and Bland-Altman biases.

### (4) Biomarker / Modeling (`scripts/biomarker.py`)
- **Function:** Conducts downstream biological validation and classification. Analyzes Differential Expressed Genes (DEGs), performs permutation testing and pathway enrichment (Hallmark, KEGG via DAVID). Evaluates cross-platform model viability (e.g., train on Real -> predict on Synthetic) using Random Forest.
- **Manuscript Support:**
  - **Figure 9 (a-f):** DEG overlap computations, enrichment correlation histograms with null permutations, bootstrap resampling analyses, and KEGG pathway consistency maps.
  - **Figure 10 (a-d):** Cross-platform Random Forest predictive classifier logic schemas and performance histograms tracking MCC.
  - **Table 4 & Table S1:** Detailed diagnostic breakdowns (Recall, Precision, MCC, False Positives/Negatives) for different experimental permutations (Real->Real, Real->Syn, Syn->Real, Syn->Syn).

---

## 5. Next Steps
1.  **Review:** Validate this script mapping against the remaining implicit needs of `GANomics_R1.docx`.
2.  **Scaffolding:** Create the `src/` directory and configure the pure Python entry points in `scripts/`.
3.  **Migration:** Iteratively migrate functions out of `old_ver` notebooks into `src/` modular code and execute via the mapped scripts, validating the `results/` against the manuscript.
