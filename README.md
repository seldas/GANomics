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

## 📊 Datasets & Data Sources

GANomics has been validated across multiple benchmark cohorts. Raw and processed gene expression profiles can be retrieved from the following sources:

### 1. Neuroblastoma (NB)
The primary benchmark dataset curated by the MAQC/SEQC consortium, consisting of 498 primary samples.
- **Microarray:** [GEO GSE49710](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49710)
- **RNA-seq:** [GEO GSE49711](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49711)

### 2. TCGA (BRCA, LUSC, LAML)
Validation on The Cancer Genome Atlas cohorts retrieved via Synapse.
- **BRCA Microarray:** [syn2319914](https://www.synapse.org/#!Synapse:syn2319914)
- **BRCA RNA-seq:** [syn2320006](https://www.synapse.org/#!Synapse:syn2320006) and [syn2320114](https://www.synapse.org/#!Synapse:syn2320114)
- **LUSC/LAML:** Retrieved from Synapse (Search by cohort ID).

### 3. NCI-60
A panel of 60 human cancer cell lines with multiple platform profiles.
- **Platforms:** Affymetrix HG-U133 Plus 2.0, HG-U133(A-B), and HG-U95(A-E).
- **Source:** [NCI CellMiner](https://discover.nci.nih.gov/cellminer/loadDownload.do)

### 4. METSIM
Adipose tissue gene expression from the Metabolic Syndrome in Men cohort.
- **Microarray:** [GEO GSE70353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70353)
- **RNA-seq:** [GEO GSE135134](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135134)

---

## 📁 Project Structure

```text
GANomics/
├── configs/            # YAML configurations for experiments and preprocessing
├── dataset/            # Genomic data root
│   ├── RAW/            # Raw GEO/Synapse downloads (GSExxxx_series_matrix.txt, etc.)
│   ├── NB/             # Processed Neuroblastoma dataset (CSV/TSV)
│   ├── METSIM/         # Processed METSIM dataset (CSV/TSV)
│   └── NCI60/          # Processed NCI-60 dataset (CSV/TSV)
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

### 1. Data Preprocessing (`scripts/preprocess.py`)
**Function:** Ingests raw files from `dataset/RAW`, aligns platforms, handles probe-to-symbol mapping, and saves processed matrices into cohort-specific folders (e.g., `dataset/NB/`).
```bash
python scripts/preprocess.py --config configs/preprocess_config.yaml
```
- **Note:** The files currently in `dataset/NB`, `dataset/METSIM`, and `dataset/NCI60` are already processed and ready for immediate use.
- **Manuscript Support:** Generates **Table 1** (Dataset Overview) and **Figure S1**.

### 2. Model Training (`scripts/train.py`)
**Function:** Bidirectional training using pair-aware feedback loss.
```bash
python scripts/train.py --config configs/nb_config.yaml
```
- **Manuscript Support:** Generates **Figure 2** (Loss curves) and **Figure 4** (CycleGAN benchmark).

### 3. Ablation Study (`scripts/run_ablation.py`)
**Function:** Investigates performance across multiple training sample sizes (10 to 400).
```bash
python scripts/run_ablation.py --config configs/nb_config.yaml --sizes 10 20 50 100 400
```
- **Manuscript Support:** Generates **Figure 3** and **Table 3**.

### 4. Testing & Sync Generation (`scripts/test_sync.py`)
**Function:** Generates synthetic profiles and benchmarks against legacy methods.
```bash
python scripts/test_sync.py --config configs/nb_config.yaml --checkpoint results/checkpoints/NB_GANomics/net_latest.pth
```
- **Manuscript Support:** Generates **Table 2** and **Figures 5, 6, 7, S2**.

### 5. Biomarker Discovery (`scripts/biomarker.py`)
**Function:** DEG analysis, Pathway Enrichment (Hallmark/KEGG), and Classifier validation.
```bash
python scripts/biomarker.py --real_A data/processed/NB_A.csv ... --gmt datasets/h.all.v2023.gmt
```
- **Manuscript Support:** Generates **Table 4** and **Figures 9, 10**.

---

## 📝 Citation
If you use GANomics in your research, please cite:
> Wu, L., Salman, H., & Tong, W. (2026). GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications. (Manuscript R1).
