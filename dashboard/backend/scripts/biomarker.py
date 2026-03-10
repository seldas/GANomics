import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis, train_eval_rf
from src.core.pathway import load_gmt, run_permutation_test, jaccard_threshold_curve

def main():
    parser = argparse.ArgumentParser(description="Biomarker and Cross-Platform Modeling for all algorithms")
    parser.add_argument("--run_id", type=str, required=True, help="Full run ID (e.g. NB_Ablation_Size_50_Run_0)")
    parser.add_argument("--labels", type=str, default=os.path.join("dataset", "NB", "clinical_info.tsv"), help="Path to clinical labels")
    parser.add_argument("--gmt", type=str, help="Path to MSigDB GMT file for pathway analysis")
    parser.add_argument("--no_adjust_path", action='store_true', help="Don't adjust PYTHONPATH")
    args = parser.parse_args()
    
    if not args.no_adjust_path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    # 1. Paths Setup
    sync_root = resolve_path(os.path.join("results", "2_SyncData", args.run_id))
    test_dir = os.path.join(sync_root, "test")
    algo_dir = os.path.join(sync_root, "algorithms")
    
    # 2. Load Real Data and Labels
    real_ma = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
    
    label_path = resolve_path(args.labels)
    sep = '\t' if label_path.endswith('.tsv') else ','
    df_labels = pd.read_csv(label_path, index_col=0, sep=sep)
    
    # Align labels with real_ma samples
    common_idx = real_ma.index.intersection(df_labels.index)
    if len(common_idx) == 0:
        print(f"Error: No matching samples between labels ({args.labels}) and data.")
        return
    
    real_ma = real_ma.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']
    print(f"[{args.run_id}] Aligned {len(common_idx)} samples for biomarker analysis.")

    # 3. Identify all synthetic profiles
    profiles = [
        ('GANomics', os.path.join(test_dir, "microarray_fake.csv"))
    ]
    
    if os.path.exists(algo_dir):
        for f in os.listdir(algo_dir):
            if f.startswith("microarray_fake_") and f.endswith(".csv"):
                algo_name = f.replace("microarray_fake_", "").replace(".csv", "").upper()
                profiles.append((algo_name, os.path.join(algo_dir, f)))

    # 4. Precompute Real DEGs (Reference)
    print("Running DEG analysis for Real Microarray (Reference)...")
    deg_real = run_deg_analysis(real_ma, y)

    # 5. Process each algorithm
    for algo_name, syn_path in profiles:
        print(f"\n>>> Processing Algorithm: {algo_name}")
        
        # Load synthetic data
        syn_ma = pd.read_csv(syn_path, index_col=0).loc[common_idx]
        
        # A. Prediction Model (Train on Syn -> Test on Real)
        # We split common_idx into train/test for this scenario
        n = len(common_idx) // 2
        train_idx, test_idx = common_idx[:n], common_idx[n:]
        
        print(f"[{algo_name}] Training Random Forest (Synthetic -> Real)...")
        metrics = train_eval_rf(syn_ma.loc[train_idx], y.loc[train_idx], real_ma.loc[test_idx], y.loc[test_idx])
        
        pred_dir = resolve_path(os.path.join("results", "4_Biomarkers", "Prediction", args.run_id))
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(os.path.join(pred_dir, f"Table_4_Classifier_Performance_{algo_name}.csv"), index=False)

        # B. DEG Analysis
        print(f"[{algo_name}] Running DEG Analysis...")
        deg_syn = run_deg_analysis(syn_ma, y)
        
        # Jaccard overlap curve
        jac_curve = jaccard_threshold_curve(deg_real, deg_syn)
        deg_dir = resolve_path(os.path.join("results", "4_Biomarkers", "DEG", args.run_id))
        os.makedirs(deg_dir, exist_ok=True)
        jac_curve.to_csv(os.path.join(deg_dir, f"Table_Fig9a_Jaccard_Curve_{algo_name}.csv"), index=False)

        # C. Pathway Enrichment (Optional)
        if args.gmt:
            print(f"[{algo_name}] Running Pathway Concordance (Permutation Test)...")
            gene_sets = load_gmt(resolve_path(args.gmt))
            obs_rho, null_dist, p_val = run_permutation_test(deg_real, deg_syn, gene_sets, B=100) # Reduced B for speed
            
            pathway_dir = resolve_path(os.path.join("results", "4_Biomarkers", "Pathway", args.run_id))
            os.makedirs(pathway_dir, exist_ok=True)
            
            pathway_results = {
                'Algorithm': algo_name,
                'Spearman_Rho': obs_rho,
                'P_Value': p_val
            }
            pd.DataFrame([pathway_results]).to_csv(os.path.join(pathway_results, f"Pathway_Concordance_{algo_name}.csv"), index=False)
            pd.Series(null_dist).to_csv(os.path.join(pathway_dir, f"Table_Fig9_Null_Dist_{algo_name}.csv"), index=False)

    print(f"\n✅ All biomarker analyses for {args.run_id} completed.")

if __name__ == "__main__":
    main()
