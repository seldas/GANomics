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
    algo_dir = os.path.join(task_sync_dir, "algorithms")
    project_id = run_id.split('_')[0]
    
    if not os.path.exists(test_dir):
        print(f"Skipping {run_id}: test directory not found.")
        return

    # 1. Load Real and GANomics Data
    try:
        ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        ma_fake_gan = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        rs_fake_gan = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    except Exception as e:
        print(f"Error loading real/GANomics data for {run_id}: {e}")
        return

    # 2. Load Labels
    label_path = os.path.join(BACKEND_DIR, "dataset", project_id, "label.txt")
    if not os.path.exists(label_path):
        # Try finding label.txt in task directory as fallback
        label_path = os.path.join(BACKEND_DIR, "dataset", project_id, "labels.txt")
        if not os.path.exists(label_path):
            print(f"Skipping {run_id}: label.txt not found at {label_path}")
            return
            
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = ma_real.index.intersection(rs_real.index).intersection(df_labels.index)
    if len(common_idx) < 10:
        print(f"Skipping {run_id}: too few samples ({len(common_idx)})")
        return
        
    y = df_labels.loc[common_idx, 'label']

    # --- Setup Mapping ---
    mapping_path = os.path.join(BACKEND_DIR, "dataset", project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, sep='\t').dropna(subset=['Agilent_Probe_Name', 'GeneSymbol'])
        gene_map = dict(zip(df_map['Agilent_Probe_Name'].astype(str), df_map['GeneSymbol'].astype(str)))

    ma_real_mapped = apply_mapping(ma_real.loc[common_idx], gene_map)
    rs_real_mapped = apply_mapping(rs_real.loc[common_idx], gene_map)
    
    common_symbols = rs_real_mapped.columns.intersection(ma_real_mapped.columns)

    # --- 3. Gather all algorithms (GANomics + baselines) ---
    # format: list of (algo_name, ma_fake_df, rs_fake_df)
    algorithms = [('GANomics', ma_fake_gan, rs_fake_gan)]
    
    if os.path.exists(algo_dir):
        # find unique algorithm names from files like microarray_fake_combat.csv
        algo_names = set()
        for f in os.listdir(algo_dir):
            if f.endswith('.csv'):
                name = f.replace('microarray_fake_', '').replace('rnaseq_fake_', '').replace('.csv', '')
                algo_names.add(name.upper())
        
        for name in algo_names:
            ma_f = os.path.join(algo_dir, f"microarray_fake_{name.lower()}.csv")
            rs_f = os.path.join(algo_dir, f"rnaseq_fake_{name.lower()}.csv")
            if os.path.exists(ma_f) and os.path.exists(rs_f):
                try:
                    algorithms.append((name, pd.read_csv(ma_f, index_col=0), pd.read_csv(rs_f, index_col=0)))
                except:
                    pass

    # --- DEG Analysis ---
    deg_out_dir = os.path.join(biomarker_root, "DEG", run_id)
    os.makedirs(deg_out_dir, exist_ok=True)
    
    print(f"Processing DEG for {run_id}...")
    deg_ma_real = run_deg_analysis(ma_real_mapped, y)
    deg_rs_real = run_deg_analysis(rs_real_mapped, y)

    # Native Baseline
    jac_native = jaccard_threshold_curve(deg_rs_real, deg_ma_real)
    jac_native.to_csv(os.path.join(deg_out_dir, "Jaccard_Curve_Baseline.csv"), index=False)

    for algo_name, ma_f, rs_f in algorithms:
        ma_f_mapped = apply_mapping(ma_f.loc[common_idx], gene_map)
        rs_f_mapped = apply_mapping(rs_f.loc[common_idx], gene_map)
        
        deg_ma_f = run_deg_analysis(ma_f_mapped, y)
        deg_rs_f = run_deg_analysis(rs_f_mapped, y)
        
        # MA -> RS comparison
        jac_ma_rs = jaccard_threshold_curve(deg_rs_real, deg_ma_f)
        jac_ma_rs.to_csv(os.path.join(deg_out_dir, f"Jaccard_Curve_{algo_name}_MA_to_RS.csv"), index=False)
        
        # RS -> MA comparison
        jac_rs_ma = jaccard_threshold_curve(deg_ma_real, deg_rs_f)
        jac_rs_ma.to_csv(os.path.join(deg_out_dir, f"Jaccard_Curve_{algo_name}_RS_to_MA.csv"), index=False)

    # --- Prediction Analysis ---
    pred_out_dir = os.path.join(biomarker_root, "Prediction", run_id)
    os.makedirs(pred_out_dir, exist_ok=True)

    train_idx, test_idx = train_test_split(common_idx, test_size=0.5, random_state=42, stratify=y)
    
    print(f"Processing Prediction for {run_id}...")

    # Cross-platform baselines (Real vs Real)
    if len(common_symbols) > 100:
        rs_mapped_sub = rs_real_mapped.loc[common_idx, common_symbols]
        ma_mapped_sub = ma_real_mapped.loc[common_idx, common_symbols]
        
        for name, train_df, test_df in [('Baseline_MA_to_RS', rs_mapped_sub, ma_mapped_sub), 
                                        ('Baseline_RS_to_MA', ma_mapped_sub, rs_mapped_sub)]:
            try:
                all_m = []
                # Real -> Real
                m_rr = train_eval_rf(test_df.loc[train_idx], y.loc[train_idx], test_df.loc[test_idx], y.loc[test_idx])
                m_rr['Scenario'] = 'Real->Real'; all_m.append(m_rr)
                # Real -> Real (other)
                m_rs = train_eval_rf(test_df.loc[train_idx], y.loc[train_idx], train_df.loc[test_idx], y.loc[test_idx])
                m_rs['Scenario'] = 'Real->Syn'; all_m.append(m_rs)
                # Real (other) -> Real
                m_sr = train_eval_rf(train_df.loc[train_idx], y.loc[train_idx], test_df.loc[test_idx], y.loc[test_idx])
                m_sr['Scenario'] = 'Syn->Real'; all_m.append(m_sr)
                # Real (other) -> Real (other)
                m_ss = train_eval_rf(train_df.loc[train_idx], y.loc[train_idx], train_df.loc[test_idx], y.loc[test_idx])
                m_ss['Scenario'] = 'Syn->Syn'; all_m.append(m_ss)
                pd.DataFrame(all_m).to_csv(os.path.join(pred_out_dir, f"Classifier_Performance_{name}.csv"), index=False)
            except Exception as e:
                print(f"Error in {name}: {e}")

    for algo_name, ma_f, rs_f in algorithms:
        # Each algorithm gets two evaluation files: one for MA-space and one for RS-space
        for platform_name, syn_df, real_df in [('MA', ma_f, ma_real), ('RS', rs_f, rs_real)]:
            try:
                common_feats = real_df.columns.intersection(syn_df.columns)
                if len(common_feats) < 10: continue
                
                real_sub = real_df.loc[common_idx, common_feats]
                syn_sub = syn_df.loc[common_idx, common_feats]
                
                all_metrics = []
                # 1. Real -> Real
                m_rr = train_eval_rf(real_sub.loc[train_idx], y.loc[train_idx], real_sub.loc[test_idx], y.loc[test_idx])
                m_rr['Scenario'] = 'Real->Real'; all_metrics.append(m_rr)
                # 2. Real -> Syn
                m_rs = train_eval_rf(real_sub.loc[train_idx], y.loc[train_idx], syn_sub.loc[test_idx], y.loc[test_idx])
                m_rs['Scenario'] = 'Real->Syn'; all_metrics.append(m_rs)
                # 3. Syn -> Real
                m_sr = train_eval_rf(syn_sub.loc[train_idx], y.loc[train_idx], real_sub.loc[test_idx], y.loc[test_idx])
                m_sr['Scenario'] = 'Syn->Real'; all_metrics.append(m_sr)
                # 4. Syn -> Syn
                m_ss = train_eval_rf(syn_sub.loc[train_idx], y.loc[train_idx], syn_sub.loc[test_idx], y.loc[test_idx])
                m_ss['Scenario'] = 'Syn->Syn'; all_metrics.append(m_ss)
                
                filename = f"Classifier_Performance_{algo_name}_{platform_name}.csv"
                pd.DataFrame(all_metrics).to_csv(os.path.join(pred_out_dir, filename), index=False)
            except Exception as e:
                print(f"Error in {algo_name}_{platform_name}: {e}")

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
