import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add the backend directory to sys.path to make 'src' importable
PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(PLAN_DIR, "..", "dashboard", "backend"))
sys.path.insert(0, BACKEND_DIR)

from src.core.analysis import run_deg_analysis, train_eval_rf
from src.core.pathway import jaccard_threshold_curve

def apply_mapping(df, mapping):
    if mapping is None: return df
    new_df = df.copy()
    # Map columns to uppercase symbols
    new_df.columns = [mapping.get(str(c), str(c)).upper() for c in new_df.columns]
    # Keep only valid symbols (filter out raw probes that didn't map)
    keep_cols = [c for c in new_df.columns if not c.startswith(('UKV4_', 'A_'))]
    if not keep_cols:
        return new_df
    # If duplicate symbols exist after mapping (many probes -> one gene), average them
    return new_df[keep_cols].T.groupby(level=0).mean().T

def run_biomarker_for_task(run_id, sync_root, biomarker_root):
    task_sync_dir = os.path.join(sync_root, run_id)
    test_dir = os.path.join(task_sync_dir, "test")
    project_id = run_id.split('_')[0]
    
    if not os.path.exists(test_dir):
        print(f"Skipping {run_id}: test directory not found.")
        return

    # 1. Load Data
    try:
        ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    except Exception as e:
        print(f"Error loading data for {run_id}: {e}")
        return

    # 2. Load Labels
    label_path = os.path.join(BACKEND_DIR, "dataset", project_id, "label.txt")
    if not os.path.exists(label_path):
        print(f"Skipping {run_id}: label.txt not found at {label_path}")
        return
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    
    common_idx = ma_real.index.intersection(rs_real.index).intersection(df_labels.index)
    if len(common_idx) < 10:
        print(f"Skipping {run_id}: too few samples ({len(common_idx)})")
        return
        
    y = df_labels.loc[common_idx, 'label']

    # --- DEG Analysis ---
    deg_out_dir = os.path.join(biomarker_root, "DEG", run_id)
    os.makedirs(deg_out_dir, exist_ok=True)
    
    # Load gene mapping if exists
    mapping_path = os.path.join(BACKEND_DIR, "dataset", project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, sep='\t').dropna(subset=['Agilent_Probe_Name', 'GeneSymbol'])
        gene_map = dict(zip(df_map['Agilent_Probe_Name'].astype(str), df_map['GeneSymbol'].astype(str)))

    # Apply mapping for DEG (matching symbols is better for cross-platform comparison)
    ma_real_mapped = apply_mapping(ma_real.loc[common_idx], gene_map)
    rs_real_mapped = apply_mapping(rs_real.loc[common_idx], gene_map)
    ma_fake_mapped = apply_mapping(ma_fake.loc[common_idx], gene_map)
    rs_fake_mapped = apply_mapping(rs_fake.loc[common_idx], gene_map)

    deg_ma_real = run_deg_analysis(ma_real_mapped, y)
    deg_rs_real = run_deg_analysis(rs_real_mapped, y)
    deg_ma_fake = run_deg_analysis(ma_fake_mapped, y)
    deg_rs_fake = run_deg_analysis(rs_fake_mapped, y)

    comparisons = [
        ('Baseline', deg_ma_real, deg_rs_real),
        ('GANomics_MA_to_RS', deg_ma_fake, deg_rs_real),
        ('GANomics_RS_to_MA', deg_rs_fake, deg_ma_real)
    ]

    for name, deg_test, deg_ref in comparisons:
        jac_curve = jaccard_threshold_curve(deg_ref, deg_test)
        jac_curve.to_csv(os.path.join(deg_out_dir, f"Jaccard_Curve_{name}.csv"), index=False)

    # --- Prediction Analysis ---
    pred_out_dir = os.path.join(biomarker_root, "Prediction", run_id)
    os.makedirs(pred_out_dir, exist_ok=True)

    train_idx, test_idx = train_test_split(common_idx, test_size=0.5, random_state=42, stratify=y)

    # Use original profiles (without symbol mapping as RF handles features fine)
    pred_profiles = [
        ('GANomics_MA', ma_fake.loc[common_idx], ma_real.loc[common_idx]),
        ('GANomics_RS', rs_fake.loc[common_idx], rs_real.loc[common_idx]),
        ('Baseline_MA_to_RS', rs_real.loc[common_idx], ma_real.loc[common_idx]),
        ('Baseline_RS_to_MA', ma_real.loc[common_idx], rs_real.loc[common_idx])
    ]

    for algo_name, syn_df, real_df in pred_profiles:
        all_metrics = []
        # Real -> Real
        m_rr = train_eval_rf(real_df.loc[train_idx], y.loc[train_idx], real_df.loc[test_idx], y.loc[test_idx])
        m_rr['Scenario'] = 'Real->Real'; all_metrics.append(m_rr)
        # Real -> Syn
        m_rs = train_eval_rf(real_df.loc[train_idx], y.loc[train_idx], syn_df.loc[test_idx], y.loc[test_idx])
        m_rs['Scenario'] = 'Real->Syn'; all_metrics.append(m_rs)
        # Syn -> Real
        m_sr = train_eval_rf(syn_df.loc[train_idx], y.loc[train_idx], real_df.loc[test_idx], y.loc[test_idx])
        m_sr['Scenario'] = 'Syn->Real'; all_metrics.append(m_sr)
        # Syn -> Syn
        m_ss = train_eval_rf(syn_df.loc[train_idx], y.loc[train_idx], syn_df.loc[test_idx], y.loc[test_idx])
        m_ss['Scenario'] = 'Syn->Syn'; all_metrics.append(m_ss)
        
        pd.DataFrame(all_metrics).to_csv(os.path.join(pred_out_dir, f"Classifier_Performance_{algo_name}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Batch Biomarker Analysis (DEG and Prediction)")
    parser.add_argument("--parent_dir", type=str, default="dashboard/backend/results", help="Parent results directory containing 2_SyncData")
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    sync_root = os.path.join(parent_dir, "2_SyncData")
    biomarker_root = os.path.join(parent_dir, "4_Biomarkers")

    if not os.path.exists(sync_root):
        print(f"Error: {sync_root} does not exist.")
        return

    task_ids = [d for d in os.listdir(sync_root) if os.path.isdir(os.path.join(sync_root, d))]
    print(f"Found {len(task_ids)} tasks in {sync_root}")

    for task_id in tqdm(task_ids, desc="Processing Biomarkers"):
        run_biomarker_for_task(task_id, sync_root, biomarker_root)

    print(f"✅ Batch biomarker analysis completed. Results saved to {biomarker_root}")

if __name__ == "__main__":
    main()
