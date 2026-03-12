# GANomics Suite: Bridging Legacy and Modern Transcriptomic Platforms

GANomics is a Generative Adversarial Network (GAN) framework for bidirectional translation between legacy (Microarray) and modern (RNA-seq) transcriptomic platforms. This suite features a comprehensive, interactive dashboard for project management, automated training (ablation studies), and deep biomarker analytics.

By integrating paired and unpaired samples through a novel **pair-aware feedback loss**, GANomics enforces one-to-one transcript mappings while preserving global gene expression distributions.

---

## 🚀 Key Features
- **Interactive Dashboard:** Manage projects, datasets, and experiments through a modern web interface.
- **Bidirectional GAN Engine:** Seamlessly convert Microarray to RNA-seq and vice versa.
- **Automated Training:** Easily configure and launch ablation and sensitivity studies with multi-GPU support.
- **Biomarker Analytics:** Integrated DEG analysis, Cross-Platform Classifier validation (Scenario Q1-Q4), and DAVID-like Pathway Enrichment (ORA) using Fisher's Exact Test to provide p-values and FDR.
- **Inference Hub:** Run inference on external datasets with ease.
- **Visual Performance Tracking:** Real-time logging, t-SNE visualizations, and performance metrics.
- **Reproducible Pipeline:** Pure Python core with systematic audit trails and result storage.

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (with `npm`)
- **CUDA-compatible GPU** (Recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/GANomics.git
   cd GANomics
   ```

2. **Set up the Backend (Python):**
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the Frontend (Vite/React):**
   ```bash
   cd dashboard/frontend
   npm install
   cd ../..
   ```

### Running the Application

Launch both the backend (FastAPI) and frontend (Vite) with a single command:
```bash
python run_dashboard.py
```
- **Frontend:** [http://localhost:8831](http://localhost:8831)
- **Backend API:** [http://localhost:8832](http://localhost:8832)
- **Logs:** Real-time session logs are saved to `app_logs/`.

---

## 📁 Dashboard Structure

- **Project Studio:** Create new projects by uploading Microarray and RNA-seq pairs.
- **Training Lab:** Launch training sessions with varying hyperparameters (Size study, Sensitivity, Ablations).
- **Session Overview:** Monitor active training runs, view real-time loss curves and performance logs.
- **Analytics Center:** Deep-dive into results:
    - **Sync Data:** Generate and download synthetic matrices.
    - **Comparative Analysis:** Benchmark against legacy methods (TDM, QN, ComBat).
    - **Biomarkers:** Visualize DEG Jaccard curves and Pathway Concordance.
    - **Prediction:** Evaluate cross-platform classifier performance.

---

## 📖 Example Workflow: Neuroblastoma (NB) Analysis

Follow these steps to reproduce the Neuroblastoma benchmark study using the dashboard:

### 1. Project Initialization
- **Action:** Navigate to the **"New Project"** section (Plus icon).
- **Input:** 
    - Name: `NB_Benchmarking`
    - Upload Microarray: `dataset/NB/df_ag.tsv`
    - Upload RNA-seq: `dataset/NB/df_rs.tsv`
    - Upload Labels (Optional): `dataset/NB/label.txt`
- **Click:** `Create Project`.
- **Expectation:** A new project card appears in your dashboard with summary statistics (498 samples, ~15k genes).

### 2. Launching a Size Study
- **Action:** Click **"Start New Session"** from the Project Dashboard.
- **Configure:** 
    - **Ablation Type:** `Size (N)`
    - **Sample Sizes:** Select `50`, `100`, and `400`.
    - **Repeats:** `5` (for statistical significance).
- **Click:** `Launch Experiment`.
- **Expectation:** The system initiates 15 independent training runs (3 sizes × 5 repeats). You can monitor progress in the **"Session Overview"** tab.

### 3. Synchronizing Data
- **Action:** Once a training run (e.g., `NB_Size_50_Run_0`) shows a `Completed` status, click the **"Task Dashboard"** (Eye icon).
- **Click:** The **"Sync"** button under Step 2 (Sync Data).
- **Expectation:** The backend generates synthetic Microarray/RNA-seq matrices. When finished, a **"Details"** button appears, allowing you to view t-SNE plots comparing Real vs. Synthetic distributions.

### 4. Comparative & Biomarker Analysis
- **Action:** Click the **"Start"** button under Step 3 (Comparative) and then Step 4 (Bio-markers).
- **Expectation:** 
    - **Comparative:** Displays Pearson correlation and RMSE benchmarks against TDM/QN.
    - **Bio-markers:** Navigate to the **"Results"** button in Step 4 to view:
        - **DEG Tab:** Jaccard curves showing gene overlap between platforms.
        - **Pathway Tab:** Concordance metrics across KEGG/Reactome libraries.
        - **Prediction Tab:** Accuracy of classifiers trained on synthetic data and tested on real samples.

---

## 📂 Repository Organization

```text
GANomics/
├── dashboard/          # Interactive Web Suite
│   ├── backend/        # FastAPI Server, API Logic & Project Data
│   │   ├── dataset/    # Genomic datasets and configurations
│   │   ├── results/    # Training checkpoints, logs, and analytics
│   │   ├── src/        # Core GANomics Package (Models & Layers)
│   │   └── scripts/    # Data processing and analysis tasks
│   └── frontend/       # React (Vite) User Interface
├── run_dashboard.py    # Unified application launcher (Windows/Linux)
├── requirements.txt    # Backend dependencies
└── app_logs/           # Captured logs from the dashboard sessions
```

---

## 📊 Datasets & Data Sources

GANomics has been validated across multiple benchmark cohorts. Key sources include:

### 1. Neuroblastoma (NB)
The primary benchmark dataset curated by the MAQC/SEQC consortium (498 primary samples).
- **Microarray:** [GEO GSE49710](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49710)
- **RNA-seq:** [GEO GSE49711](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49711)

### 2. TCGA (BRCA, LUSC, LAML)
- **Source:** Retrieved via [Synapse](https://www.synapse.org/).

### 3. NCI-60
- **Platforms:** Affymetrix HG-U133 Plus 2.0, HG-U133(A-B), and HG-U95(A-E).
- **Source:** [NCI CellMiner](https://discover.nci.nih.gov/cellminer/loadDownload.do)

### 4. METSIM
- **Microarray:** [GEO GSE70353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70353)
- **RNA-seq:** [GEO GSE135134](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135134)

---

## 📝 Citation
If you use GANomics in your research, please cite:
> Wu, L., Salman, H., & Tong, W. (2026). GANomics: Bridging Legacy and Modern Transcriptomic Platforms for Clinical Applications. (Manuscript R1).
