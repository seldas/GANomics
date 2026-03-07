# GANomics: Bridging Legacy and Modern Transcriptomic Platforms

GANomics is a Generative Adversarial Network (GAN) framework designed for bidirectional translation between legacy (Microarray) and modern (RNA-seq) transcriptomic platforms. By integrating paired and unpaired samples through a novel **pair-aware feedback loss**, GANomics enforces one-to-one transcript mappings while preserving global gene expression distributions.

This repository contains the refactored, modular Python implementation supporting the methodologies and results described in the manuscript: **"GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications"**.

---

## 🚀 Key Features
- **Bidirectional Translation:** Seamlessly convert Microarray to RNA-seq and vice versa.
- **Multi-GPU Parallelization:** Automatically distribute independent trials and sensitivity sweeps across multiple GPUs (e.g., 8x V100).
- **Data Efficiency:** Achieve high per-sample correlations (>0.96) with as few as 10–50 paired profiles.
- **Architecture Ablation:** Support for one-directional (Supervised GAN) vs bidirectional (CycleGAN) baseline comparisons.
- **Reproducible Pipeline:** Pure Python implementation with systematic audit trails and result storage in `results/`.

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
│   ├── RAW/            # Raw GEO/Synapse downloads
│   ├── NB/             # Processed Neuroblastoma dataset (CSV/TSV)
│   ├── METSIM/         # Processed METSIM dataset (CSV/TSV)
│   └── NCI60/          # Processed NCI-60 dataset (CSV/TSV)
├── results/            # Systematic output storage
│   ├── checkpoints/    # Model weights and 'train_samples.txt' audit trails
│   ├── sync_data/      # Generated synthetic matrices (Split into train/test)
│   ├── figures/        # Manuscript-ready plots (PDF/PNG)
│   ├── tables/         # Generated CSV/Excel stats for the paper
│   └── logs/           # Training logs matching historical 'example.txt' format
├── src/                # Core GANomics Package
│   ├── bio_utils/      # Normalization (ComBat, QN, TDM, YuGene, CuBlock)
│   ├── core/           # Training loops, evaluation metrics, and analysis logic
│   ├── datasets/       # Genomics-specific data loaders with reproducible shuffling
│   └── models/         # GANomicsModel with Feedback Loss and direction control
└── scripts/            # Executable Entry Points
```

---

## 🛠️ Pipeline & Manuscript Alignment

### 1. Data Preprocessing (`scripts/preprocess.py`)
**Function:** Ingests raw files from `dataset/RAW`, aligns platforms, and handles mapping.
```bash
python scripts/preprocess.py --config configs/preprocess_config.yaml
```

### 2. Ablation & Sensitivity Studies (`scripts/run_ablation.py`)
**Function:** Automates large-scale experiments across multiple GPUs.
```bash
# Run 5 independent trials for size study across 8 GPUs
python scripts/run_ablation.py --config configs/nb_config.yaml --sizes 50 100 400 --repeats 5 --gpu_ids 0,1,2,3,4,5,6,7

# Run hyperparameter sensitivity sweeps
python scripts/run_ablation.py --config configs/nb_config.yaml --betas 1.0 10.0 50.0
python scripts/run_ablation.py --config configs/nb_config.yaml --lambdas 1.0 10.0 50.0
```

### 3. Testing & Sync Generation (`scripts/test_sync.py`)
**Function:** Generates synthetic profiles. Automatically uses the training audit trail to separate "Seen" (Train) and "Unseen" (Test) results.
```bash
python scripts/test_sync.py --config configs/nb_config.yaml --project NB --sample_size 50 --run_id 0
```

### 4. Comparative Analysis (`scripts/comparative_analysis.py`)
**Function:** Directly benchmarks GANomics against legacy methods (TDM, QN, ComBat, etc.).
```bash
python scripts/comparative_analysis.py --config configs/nb_config.yaml --project NB --sample_size 50
```

### 5. Biomarker Discovery (`scripts/biomarker.py`)
**Function:** Performs DEG analysis, Pathway Enrichment, and Cross-Platform Classifier validation (Scenario Q1-Q4).
```bash
python scripts/biomarker.py --real_A dataset/NB/NB_A.csv --syn_A results/sync_data/NB_50_0/test/microarray_fake.csv ...
```

---

## 📝 Citation
If you use GANomics in your research, please cite:
> Wu, L., Salman, H., & Tong, W. (2026). GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications. (Manuscript R1).
