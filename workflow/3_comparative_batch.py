import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add the backend directory to sys.path to make 'src' importable
# This script is located in ./workflow/
# Backend is in ./dashboard/backend/
PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(PLAN_DIR, "..", "dashboard", "backend"))
sys.path.insert(0, BACKEND_DIR)

from src.bio_utils import (
    combat_evaluate_paired, fit_cublock_translator, translate_cublock,
    quantile_normalize, tdm_normalize, yugene_evaluate_paired
)

def measure_perf_detailed(df_real, df_fake):
    """
    Compute per-sample metrics and return summary with CI (standard deviation).
    Derived from dashboard/backend/scripts/comparative_analysis.py
    """
    
    pearsons, spearmans, maes, rmses, l1s, l2s = [], [], [], [], [], []
    
    for i in range(len(df_real)):
        r = df_real.values[i]
        f = df_fake.values[i]
        
        # Pearson/Spearman
        p_val, _ = pearsonr(r, f)
        s_val, _ = spearmanr(r, f)
        pearsons.append(p_val)
        spearmans.append(s_val)
        
        # Errors
        diff = r - f
        maes.append(np.abs(diff).mean())
        rmses.append(np.sqrt((diff**2).mean()))
        l1s.append(np.abs(diff).sum())
        l2s.append(np.sqrt((diff**2).sum()))
    
    metrics = {
        'Pearson': pearsons,
        'Spearman': spearmans,
        'MAE': maes,
        'RMSE': rmses,
        'L1': l1s,
        'L2': l2s
    }
    
    summary = {}
    for k, vals in metrics.items():
        mean = np.mean(vals)
        std = np.std(vals)
        summary[k] = f"{mean:.3f} (±{std:.3f})"
    return summary

def run_comparative_for_task(run_id, sync_root, comp_root, algorithms):
    task_sync_dir = os.path.join(sync_root, run_id)
    train_dir = os.path.join(task_sync_dir, "train")
    test_dir = os.path.join(task_sync_dir, "test")
    
    if not os.path.exists(test_dir):
        return

    # Load data
    try:
        # Load training data (needed for baselines like ComBat, CuBlock, etc.)
        train_ag = pd.read_csv(os.path.join(train_dir, "microarray_real.csv"), index_col=0) if os.path.exists(os.path.join(train_dir, "microarray_real.csv")) else None
        train_ngs = pd.read_csv(os.path.join(train_dir, "rnaseq_real.csv"), index_col=0) if os.path.exists(os.path.join(train_dir, "rnaseq_real.csv")) else None
        
        test_ag = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        test_ngs = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        # harmonize gene id (columns)
        train_ngs.columns = train_ag.columns
        test_ngs.columns = test_ag.columns
        
        # GANomics Results (Fake) produced by previous sync step
        df_ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0) if os.path.exists(os.path.join(test_dir, "microarray_fake.csv")) else None
        df_rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0) if os.path.exists(os.path.join(test_dir, "rnaseq_fake.csv")) else None
    except Exception as e:
        print(f"Error loading data for {run_id}: {e}")
        return

    # Run Baselines (only if requested)
    if algorithms:
        alg_dir = os.path.join(task_sync_dir, 'algorithm')
        if not os.path.exists(alg_dir):
            os.system('mkdir -p '+alg_dir)

    df_ma_combat, df_rs_combat = None, None
    if 'combat' in algorithms and train_ag is not None and train_ngs is not None:
        res_combat = combat_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
        df_ma_combat, df_rs_combat = res_combat['rnaseq_to_microarray'], res_combat['microarray_to_rnaseq']
        df_ma_combat.to_csv(os.path.join(alg_dir, 'microarray_fake_combat.csv'))
        df_rs_combat.to_csv(os.path.join(alg_dir, 'rnaseq_fake_combat.csv'))
        

    df_ma_yugene, df_rs_yugene = None, None
    if 'yugene' in algorithms and train_ag is not None and train_ngs is not None:
        res_yugene = yugene_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
        df_ma_yugene, df_rs_yugene = res_yugene['rnaseq_to_microarray'], res_yugene['microarray_to_rnaseq']
        df_ma_yugene.to_csv(os.path.join(alg_dir, 'microarray_fake_yugene.csv'))
        df_rs_yugene.to_csv(os.path.join(alg_dir, 'rnaseq_fake_yugene.csv'))

        
    df_ma_cublock, df_rs_cublock = None, None
    if 'cublock' in algorithms and train_ag is not None and train_ngs is not None:
        trans_rs = fit_cublock_translator(train_ag, train_ngs)
        df_rs_cublock = translate_cublock(test_ag, trans_rs) if test_ag is not None else None
        trans_ma = fit_cublock_translator(train_ngs, train_ag)
        df_ma_cublock = translate_cublock(test_ngs, trans_ma) if test_ngs is not None else None
        df_ma_cublock.to_csv(os.path.join(alg_dir, 'microarray_fake_cublock.csv'))
        df_rs_cublock.to_csv(os.path.join(alg_dir, 'rnaseq_fake_cublock.csv'))
        
    
    df_ma_tdm, df_rs_tdm = None, None
    if 'tdm' in algorithms and train_ag is not None and train_ngs is not None:
        res_tdm = tdm_normalize(train_ag, test_ag, train_ngs, test_ngs)
        df_ma_tdm, df_rs_tdm = res_tdm['rnaseq_to_microarray'], res_tdm['microarray_to_rnaseq']
        df_ma_tdm.to_csv(os.path.join(alg_dir, 'microarray_fake_tdm.csv'))
        df_rs_tdm.to_csv(os.path.join(alg_dir, 'rnaseq_fake_tdm.csv'))
        
    
    df_ma_qn, df_rs_qn = None, None
    if 'qn' in algorithms and train_ag is not None and train_ngs is not None:
        res_qn = quantile_normalize(train_ag, test_ag, train_ngs, test_ngs)
        df_ma_qn, df_rs_qn = res_qn['rnaseq_to_microarray'], res_qn['microarray_to_rnaseq']
        df_ma_qn.to_csv(os.path.join(alg_dir, 'microarray_fake_qn.csv'))
        df_rs_qn.to_csv(os.path.join(alg_dir, 'rnaseq_fake_qn.csv'))
        
    
    # Aggregate Metrics for comparisons
    comparisons = []
    if test_ag is not None:
        if df_ma_fake is not None: comparisons.append(("GANomics (MA)", test_ag, df_ma_fake))
        if df_ma_combat is not None: comparisons.append(("ComBat (MA)",   test_ag, df_ma_combat))
        if df_ma_yugene is not None: comparisons.append(("YuGene (MA)",   test_ag, df_ma_yugene))
        if df_ma_cublock is not None: comparisons.append(("CuBlock (MA)",  test_ag, df_ma_cublock))
        if df_ma_tdm is not None: comparisons.append(("TDM (MA)",      test_ag, df_ma_tdm))
        if df_ma_qn is not None: comparisons.append(("Quantile (MA)", test_ag, df_ma_qn))
        
    if test_ngs is not None:
        if df_rs_fake is not None: comparisons.append(("GANomics (RS)", test_ngs, df_rs_fake))
        if df_rs_combat is not None: comparisons.append(("ComBat (RS)",   test_ngs, df_rs_combat))
        if df_rs_yugene is not None: comparisons.append(("YuGene (RS)",   test_ngs, df_rs_yugene))
        if df_rs_cublock is not None: comparisons.append(("CuBlock (RS)",  test_ngs, df_rs_cublock))
        if df_rs_tdm is not None: comparisons.append(("TDM (RS)",      test_ngs, df_rs_tdm))
        if df_rs_qn is not None: comparisons.append(("Quantile (RS)", test_ngs, df_rs_qn))
        
    if test_ag is not None and test_ngs is not None:
        comparisons.append(("Baseline (paired)", test_ag, test_ngs))

    final_results = []
    for name, real, fake in comparisons:
        print(name, real.shape, fake.shape)
        perf = measure_perf_detailed(real, fake)
        perf['Algorithm'] = name
        final_results.append(perf)
        
    df_final = pd.DataFrame(final_results)
    
    out_dir = os.path.join(comp_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    df_final.to_csv(os.path.join(out_dir, "Test_performance.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Batch Comparative Analysis for GANomics")
    parser.add_argument("--parent_dir", type=str, help="Parent results directory containing 2_testSync (e.g. dashboard/backend/results)")
    parser.add_argument("--algorithms", type=str, nargs="+", default='[combat,yugene,cublock,qn,tdm]', help="Baseline algorithms to run (default: none, only GANomics included)")
    args = parser.parse_args()

    # Determine absolute paths for input and output
    parent_dir = os.path.abspath(args.parent_dir)
    sync_root = os.path.join(parent_dir, "2_SyncData")
    comp_root = os.path.join(parent_dir, "3_ComparativeAnalysis")

    if not os.path.exists(sync_root):
        print(f"Error: {sync_root} does not exist. Please ensure data is present in 2_testSync.")
        return

    # Iterate through all task directories in 2_testSync
    task_ids = [d for d in os.listdir(sync_root) if os.path.isdir(os.path.join(sync_root, d))]
    task_ids = [d for d in task_ids if 'Ablation_Size_50' in d] # specific filters
    print(f"Found {len(task_ids)} tasks in {sync_root}")

    for task_id in tqdm(task_ids, desc="Processing tasks"):
        run_comparative_for_task(task_id, sync_root, comp_root, args.algorithms)

    print(f"✅ Batch comparative analysis completed. Results saved to {comp_root}")

if __name__ == "__main__":
    main()
