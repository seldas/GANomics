import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.core.analysis import run_deg_analysis
from src.core.pathway import load_gmt, gene_set_enrichment, spearman_rank_concordance

def analyze_pathways():
    # Setup paths
    gmt_path = "dataset/RAW/h.all.v2025.1.Hs.symbols.gmt"
    if not os.path.exists(gmt_path):
        print(f"GMT file not found at {gmt_path}. Skipping pathway analysis.")
        return

    gene_sets = load_gmt(gmt_path)
    label_path = "plan/SEQC_NB_249_ValidationSamples_ClinicalInfo_20121128.txt"
    df_labels = pd.read_csv(label_path, sep='\t').set_index('SEQC_NB_SampleID')
    y = df_labels['D_FAV_All'].dropna().map({0: 'Favorable', 1: 'Unfavorable'})

    # Reference Real Data (Microarray)
    # We need the real data used in the runs. 
    # Based on scripts/biomarker.py, we can use one of the sync_data folders as a proxy if it contains real data.
    real_path = "results/sync_data/NB_50_0/test/microarray_real.csv"
    if not os.path.exists(real_path):
        print("Real data reference not found.")
        return
        
    df_real = pd.read_csv(real_path, index_col=0)
    idx = df_real.index.intersection(y.index)
    deg_real = run_deg_analysis(df_real.loc[idx], y.loc[idx])
    pathway_real = gene_set_enrichment(deg_real, gene_sets)

    methods = {
        "GANomics": "results/sync_data/NB_50_0/test/microarray_fake.csv",
        "ComBat": "results/sync_data/NB_50_ComBat/test/microarray_fake.csv",
        "TDM": "results/sync_data/NB_50_TDM/test/microarray_fake.csv",
        "QN": "results/sync_data/NB_50_QN/test/microarray_fake.csv",
        "YuGene": "results/sync_data/NB_50_YuGene/test/microarray_fake.csv"
    }

    results = []
    for name, path in methods.items():
        if os.path.exists(path):
            df_syn = pd.read_csv(path, index_col=0)
            idx_syn = df_syn.index.intersection(y.index)
            deg_syn = run_deg_analysis(df_syn.loc[idx_syn], y.loc[idx_syn])
            pathway_syn = gene_set_enrichment(deg_syn, gene_sets)
            rho = spearman_rank_concordance(pathway_real, pathway_syn)
            results.append({"Method": name, "Spearman_rho": rho})
            print(f"{name}: Spearman rho = {rho:.4f}")
        else:
            # Fallback: check results/tables/biomarkers if files aren't in sync_data
            # But we need raw files to run run_deg_analysis correctly for pathway comparison
            print(f"Synthetic data for {name} not found at {path}")

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv("results/tables/r1c5_pathway_concordance.csv", index=False)
        
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Method'], df_res['Spearman_rho'], color='skyblue')
        plt.ylabel("Spearman Rank Concordance (rho)")
        plt.title("Pathway Enrichment Concordance: GANomics vs Baselines")
        plt.ylim(0, 1.0)
        plt.savefig("results/figures/r1c5_pathway_comparison.png")
        print("Saved pathway comparison plot.")

if __name__ == "__main__":
    analyze_pathways()
