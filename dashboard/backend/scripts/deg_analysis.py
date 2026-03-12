import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis
from src.core.pathway import jaccard_threshold_curve

def main():
    parser = argparse.ArgumentParser(description="Focused DEG Concordance Analysis")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--ext_id", type=str)
    parser.add_argument("--labels", type=str)
    args = parser.parse_args()

    # Paths Setup
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sync_root = os.path.join(backend_dir, "results", "2_SyncData", args.run_id)
    project_id = args.run_id.split('_')[0]
    
    if args.ext_id:
        test_dir = os.path.join(sync_root, args.ext_id)
        out_root = test_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        out_root = os.path.join(backend_dir, "results", "4_Biomarkers")

    # Load All 4 Profiles
    try:
        ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    except FileNotFoundError as e:
        print(f"Error: Required data files not found in {test_dir}. Ensure Step 2 (Sync) completed. {e}")
        return

    # Labels
    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    
    # Alignment (Must be perfectly paired)
    common_idx = ma_real.index.intersection(rs_real.index).intersection(df_labels.index)
    y = df_labels.loc[common_idx, 'label']
    
    print(f"Analyzing {len(common_idx)} samples across both directions.")

    # 1. Compute DEGs for all profiles
    print("Computing DEGs: Real Platforms...")
    deg_ma_real = run_deg_analysis(ma_real.loc[common_idx], y)
    deg_rs_real = run_deg_analysis(rs_real.loc[common_idx], y)
    
    print("Computing DEGs: GANomics Synthetic...")
    deg_ma_fake = run_deg_analysis(ma_fake.loc[common_idx], y)
    deg_rs_fake = run_deg_analysis(rs_fake.loc[common_idx], y)

    deg_dir = os.path.join(out_root, "DEG", args.run_id) if not args.ext_id else os.path.join(out_root, "DEG")
    os.makedirs(deg_dir, exist_ok=True)

    # 2. Calculate Jaccard Curves for the 3 Key Comparisons
    comparisons = [
        ('Baseline', deg_ma_real, deg_rs_real, "Native Cross-Platform"),
        ('GANomics_MA_to_RS', deg_rs_fake, deg_rs_real, "Microarray -> RNA-Seq"),
        ('GANomics_RS_to_MA', deg_ma_fake, deg_ma_real, "RNA-Seq -> Microarray")
    ]

    for algo_name, deg_test, deg_ref, desc in comparisons:
        print(f">>> Calculating overlap: {desc}")
        jac_curve = jaccard_threshold_curve(deg_ref, deg_test)
        jac_curve.to_csv(os.path.join(deg_dir, f"Jaccard_Curve_{algo_name}.csv"), index=False)

    # Save Significant DEG lists (p < 0.05) for download
    def save_deg_list(df, filename):
        if not df.empty and 'p_value' in df.columns:
            # Sort by p-value (lowest first)
            sigs = df[df['p_value'] < 0.05].sort_values('p_value')
            with open(os.path.join(deg_dir, filename), 'w') as f:
                f.write("\n".join(sigs.index.astype(str).tolist()))

    save_deg_list(deg_ma_real, "DEGs_Microarray_Real.txt")
    save_deg_list(deg_rs_real, "DEGs_RNAseq_Real.txt")
    save_deg_list(deg_ma_fake, "DEGs_Microarray_Fake.txt")
    save_deg_list(deg_rs_fake, "DEGs_RNAseq_Fake.txt")

    print("✅ Redesigned DEG analysis completed.")

if __name__ == "__main__":
    main()
