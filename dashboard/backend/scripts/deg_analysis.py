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
    parser = argparse.ArgumentParser(description="DEG Overlap Analysis")
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
        algo_dir = os.path.join(sync_root, f"algorithms_{args.ext_id}")
        out_root = test_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        algo_dir = os.path.join(sync_root, "algorithms")
        out_root = os.path.join(backend_dir, "results", "4_Biomarkers")

    real_ma_path = os.path.join(test_dir, "microarray_real.csv")
    if not os.path.exists(real_ma_path):
        real_ma_path = os.path.join(test_dir, "rnaseq_real.csv")
    
    if not os.path.exists(real_ma_path):
        print("Error: Synced real data not found.")
        return

    # Labels
    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    if not os.path.exists(label_path):
        print(f"Error: label.txt not found at {label_path}")
        return

    real_ma = pd.read_csv(real_ma_path, index_col=0)
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = real_ma.index.intersection(df_labels.index)
    real_ma = real_ma.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']

    # Profiles
    profiles = [('GANomics', os.path.join(test_dir, "microarray_fake.csv" if "microarray" in real_ma_path else "rnaseq_fake.csv"))]
    if os.path.exists(algo_dir):
        for f in os.listdir(algo_dir):
            if (f.startswith("microarray_fake_") or f.startswith("rnaseq_fake_")) and f.endswith(".csv"):
                algo_name = f.replace("microarray_fake_", "").replace("rnaseq_fake_", "").replace(".csv", "").upper()
                profiles.append((algo_name, os.path.join(algo_dir, f)))

    print("Running Reference DEG Analysis...")
    deg_real = run_deg_analysis(real_ma, y)

    deg_dir = os.path.join(out_root, "DEG", args.run_id) if not args.ext_id else os.path.join(out_root, "DEG")
    os.makedirs(deg_dir, exist_ok=True)

    for algo_name, syn_path in profiles:
        if not os.path.exists(syn_path): continue
        print(f"Processing DEG for {algo_name}...")
        syn_ma = pd.read_csv(syn_path, index_col=0).loc[common_idx]
        deg_syn = run_deg_analysis(syn_ma, y)
        jac_curve = jaccard_threshold_curve(deg_real, deg_syn)
        jac_curve.to_csv(os.path.join(deg_dir, f"Jaccard_Curve_{algo_name}.csv"), index=False)

    print("✅ DEG analysis completed.")

if __name__ == "__main__":
    main()
