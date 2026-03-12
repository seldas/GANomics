import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import train_eval_rf

def main():
    parser = argparse.ArgumentParser(description="Cross-Platform Prediction Modeling")
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
    if not os.path.exists(real_ma_path): real_ma_path = os.path.join(test_dir, "rnaseq_real.csv")
    if not os.path.exists(real_ma_path): return

    # Labels
    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    real_ma = pd.read_csv(real_ma_path, index_col=0)
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = real_ma.index.intersection(df_labels.index)
    real_ma = real_ma.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']

    # Profiles
    is_ma = "microarray" in real_ma_path
    target_prefix = "microarray_fake_" if is_ma else "rnaseq_fake_"
    
    profiles = [('GANomics', os.path.join(test_dir, "microarray_fake.csv" if is_ma else "rnaseq_fake.csv"))]
    
    # Add Baseline: Microarray Real vs RNA-Seq Real
    other_real_path = os.path.join(test_dir, "rnaseq_real.csv" if is_ma else "microarray_real.csv")
    if os.path.exists(other_real_path):
        profiles.append(('Baseline', other_real_path))

    if os.path.exists(algo_dir):
        for f in os.listdir(algo_dir):
            if f.startswith(target_prefix) and f.endswith(".csv"):
                algo_name = f.replace(target_prefix, "").replace(".csv", "").upper()
                profiles.append((algo_name, os.path.join(algo_dir, f)))

    pred_dir = os.path.join(out_root, "Prediction", args.run_id) if not args.ext_id else os.path.join(out_root, "Prediction")
    os.makedirs(pred_dir, exist_ok=True)

    # Train/Test Split (Standard 50/50 for all scenarios)
    train_idx, test_idx = train_test_split(common_idx, test_size=0.5, random_state=42, stratify=y)

    for algo_name, syn_path in profiles:
        if not os.path.exists(syn_path): continue
        print(f"Processing Prediction Models for {algo_name}...")
        syn_ma = pd.read_csv(syn_path, index_col=0).loc[common_idx]
        
        all_metrics = []
        # 1. Real -> Real
        m_rr = train_eval_rf(real_ma.loc[train_idx], y.loc[train_idx], real_ma.loc[test_idx], y.loc[test_idx])
        m_rr['Scenario'] = 'Real->Real'; all_metrics.append(m_rr)
        
        # 2. Real -> Syn
        m_rs = train_eval_rf(real_ma.loc[train_idx], y.loc[train_idx], syn_ma.loc[test_idx], y.loc[test_idx])
        m_rs['Scenario'] = 'Real->Syn'; all_metrics.append(m_rs)
        
        # 3. Syn -> Real
        m_sr = train_eval_rf(syn_ma.loc[train_idx], y.loc[train_idx], real_ma.loc[test_idx], y.loc[test_idx])
        m_sr['Scenario'] = 'Syn->Real'; all_metrics.append(m_sr)
        
        # 4. Syn -> Syn
        m_ss = train_eval_rf(syn_ma.loc[train_idx], y.loc[train_idx], syn_ma.loc[test_idx], y.loc[test_idx])
        m_ss['Scenario'] = 'Syn->Syn'; all_metrics.append(m_ss)
        
        pd.DataFrame(all_metrics).to_csv(os.path.join(pred_dir, f"Classifier_Performance_{algo_name}.csv"), index=False)

    print("✅ Prediction modeling completed.")

if __name__ == "__main__":
    main()
