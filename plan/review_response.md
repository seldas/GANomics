# Reviewer Response Plan: GANomics Dashboard

This document tracks the status of addressing reviewer comments through the dashboard and backend functionality.

## Reviewer 1

### Major Concerns

**Comment (1): Hyperparameter sensitivity and justification (β, λ)**
> The feedback loss weight β is fixed at 10.0 throughout all experiments... a sensitivity analysis of β is necessary. Similarly, the cycle-consistency weight λ... should also be examined.
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** The dashboard now includes a dedicated "Sensitivity" ablation mode. Users can launch multiple runs with varying Beta and Lambda values. The "Ablation Analytics" modal provides dual-mode toggles, float-based ranking, and trend charts to justify the choice of $\beta=10$ and $\lambda=10$.

**Comment (2): Insufficient comparison with CycleGAN for larger paired sample sizes**
> While Figure 4 demonstrates that GANomics outperforms CycleGAN when training samples are limited (<50 pairs), the comparison for N ≥ 50 is not rigorously quantified... conduct multiple independent trials and apply paired statistical tests.
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** The "Sample Size" ablation dashboard automatically groups repeated runs (repeats state variable in frontend) and calculates Mean ± Std. Statistical p-values for N=50 vs baselines are included in the results processing.

**Comment (3): Unclear practical advantage when paired samples are sufficient**
> if the performance gap between GANomics and CycleGAN diminishes or becomes non-significant when N ≥ 50, the practical rationale... needs clarification.
*   **Status:** **NO DEVELOPMENT NEEDED**. This is a manuscript discussion point regarding the trade-off between data efficiency and model complexity.

**Comment (4): Missing one-directional GAN baselines with feedback loss**
> To isolate the contribution of the feedback loss itself, the authors should include two additional one-directional baselines: (1) a GAN for MA→RS trained with feedback loss only, and (2) a GAN for RS→MA with feedback loss only.
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** Developed the "Architecture" ablation mode. It supports `AtoB` and `BtoA` directions which explicitly skip cycle-consistency and identity losses. The dashboard analytics modal provides a 3-column table (Both vs A-B vs B-A) to visualize these results side-by-side.

**Comment (5): Lack of comparative baselines in biological signal preservation analysis**
> include at least one representative baseline (e.g., CycleGAN-50 or ComBat) in the DEG and pathway enrichment analyses to contextualize the results.
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** The Bio-marker Analysis panel (Step 4) now performs parallel analysis for GANomics, ComBat, CuBlock, QN, TDM, and YuGene. The "Start Analysis" button in Step 3 automatically triggers these baseline generations for comparison in the Jaccard and Pathway charts.

**Comment (6): Figure 1 (schematic) lacks clarity and detail**
> A revised figure should: 1) Clearly distinguish paired and unpaired data paths...
*   **Status:** **NO DEVELOPMENT NEEDED**. This is a graphical design task for the manuscript.

**Comment (8): Code and reproducibility**
> The manuscript does not provide a link to the source code... the authors must make the code publicly available.
*   **Status:** **NO DEVELOPMENT NEEDED**. Administrative task for submission.

---

### Minor Concerns

**1. Incomplete reference:**
*   **Status:** **NO DEVELOPMENT NEEDED**. Editorial fix.

**2. Figure readability:**
*   **Status:** **ADDRESSED**. Dashboard charts now use larger fonts, horizontal labels, and better color contrast.

**3. Terminology inconsistency:**
*   **Status:** **ADDRESSED**. Dashboard UI and logs have been standardized to "Paired" and "Unpaired".

---

## Reviewer 2

**Comment (1): Comprehensive benchmarking of biological signal preservation**
> Expanding the evaluation to include additional metrics specifically targeting biological conservation (beyond Pearson correlation)...
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** Added Step 4 (DEG Jaccard Similarity), Step 5 (Pathway Concordance), and Step 6 (Cross-platform Prediction MCC) to go beyond simple correlation.

**Comment (2): Experimental setup for predictive classifiers**
> ambiguous whether the (C1) Real→Real classifier corresponds to a model trained and tested within the same modality... does integrating data lead to improved performance?
*   **Status:** **DEVELOPMENT NEEDED**.
*   **Action:** Add a "Data Integration" mode to the prediction script where the classifier is trained on a *combined* dataset (Real + Synthetic) to prove "added value."

**Comment (3): Advantage of bidirectional translation**
> provide a concrete or even hypothetical use case where converting RNA-seq data into microarray-like profiles would be advantageous.
*   **Status:** **NO DEVELOPMENT NEEDED**. Discussion point (e.g., meta-analysis with legacy microarray-only datasets).

---

## Reviewer 3

**Comment (1): Overfitting vs Generalizability across cohorts**
> It remains unclear whether GANomics can generalize across cohorts in realistic scenarios.
*   **Status:** **SPECULATIVE DEVELOPMENT**.
*   **Action:** Could implement a "Project Merge" or "Cross-Project Validation" tool where a model trained on Project A is tested on Project B (e.g., NB trained model tested on NCI60).

**Comment (2): Practical benefits of synthetic data**
> do not show whether incorporating synthetic data can improve downstream analyses compared with using only real data.
*   **Status:** **ADDRESSED (Development Complete)**.
*   **Implementation:** The "Prediction" analysis includes the `Syn->Real` scenario, demonstrating that synthetic data encodes enough biological signal to train functional classifiers for real-world samples.

**Comment (3): Model architecture and training parameter details**
*   **Status:** **NO DEVELOPMENT NEEDED**. Documentation task.

**Comment (4): Public link to implementation**
*   **Status:** **NO DEVELOPMENT NEEDED**. Administrative task.
