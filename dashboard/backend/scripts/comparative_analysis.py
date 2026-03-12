import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.evaluation import compute_metrics
from src.bio_utils import (
    combat_evaluate_paired, fit_cublock_translator, translate_cublock,
    quantile_normalize, tdm_normalize, yugene_evaluate_paired
)

def measure_perf_detailed(df_real, df_fake):
    """
    Compute per-sample metrics and return summary with CI (standard deviation).
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

def main():
    parser = argparse.ArgumentParser(description="Comparative Analysis with Baseline Methods")
    parser.add_argument("--run_id", type=str, required=True, help="Full run ID (e.g. NB_Ablation_Size_50_Run_0)")
    parser.add_argument("--ext_id", type=str, help="External Dataset ID (e.g. ext1)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=['combat', 'yugene', 'cublock', 'tdm', 'qn'], help="Baseline algorithms to run")
    parser.add_argument("--no_adjust_path", action='store_true', help="Don't adjust PYTHONPATH")
    args = parser.parse_args()
    
    if not args.no_adjust_path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    # 1. Load Sync Data
    sync_root = resolve_path(os.path.join("results", "2_SyncData", args.run_id))
    
    if args.ext_id:
        # For external datasets, we look in the ext_id folder
        ext_dir = os.path.join(sync_root, args.ext_id)
        
        # Load the Real data produced by Step 2 (test_sync.py)
        # This is better because Step 2 already handled gene list alignment
        path_ag_real = os.path.join(ext_dir, "microarray_real.csv")
        path_rs_real = os.path.join(ext_dir, "rnaseq_real.csv")
        
        # GANomics Results (Fake) produced by Step 2
        path_ag_fake = os.path.join(ext_dir, "microarray_fake.csv")
        path_rs_fake = os.path.join(ext_dir, "rnaseq_fake.csv")
        
        # Load whatever is available
        test_ag = pd.read_csv(path_ag_real, index_col=0) if os.path.exists(path_ag_real) else None
        test_ngs = pd.read_csv(path_rs_real, index_col=0) if os.path.exists(path_rs_real) else None
        df_ma_fake = pd.read_csv(path_ag_fake, index_col=0) if os.path.exists(path_ag_fake) else None
        df_rs_fake = pd.read_csv(path_rs_fake, index_col=0) if os.path.exists(path_rs_fake) else None
        
        # We still need training data for baselines (Combat, CuBlock, etc.)
        # Use the original training data of the project
        train_dir = os.path.join(sync_root, "train")
        train_ag = pd.read_csv(os.path.join(train_dir, "microarray_real.csv"), index_col=0)
        train_ngs = pd.read_csv(os.path.join(train_dir, "rnaseq_real.csv"), index_col=0)
        
    else:
        train_dir = os.path.join(sync_root, "train")
        test_dir = os.path.join(sync_root, "test")
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError(f"Sync data for {args.run_id} not found in {sync_root}. Run test_sync.py first.")

        train_ag = pd.read_csv(os.path.join(train_dir, "microarray_real.csv"), index_col=0)
        train_ngs = pd.read_csv(os.path.join(train_dir, "rnaseq_real.csv"), index_col=0)
        test_ag = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        test_ngs = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        
        # GANomics Results (Fake)
        df_ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        df_rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    
    # 2. Run Baselines
    print(f"[{args.run_id}] Running Baseline Comparisons: {args.algorithms}")
    
    # ComBat
    df_ma_combat, df_rs_combat = None, None
    if 'combat' in args.algorithms:
        res_combat = combat_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
        df_ma_combat, df_rs_combat = res_combat['rnaseq_to_microarray'], res_combat['microarray_to_rnaseq']
    
    # YuGene
    df_ma_yugene, df_rs_yugene = None, None
    if 'yugene' in args.algorithms:
        res_yugene = yugene_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
        df_ma_yugene, df_rs_yugene = res_yugene['rnaseq_to_microarray'], res_yugene['microarray_to_rnaseq']
    
    # CuBlock
    df_ma_cublock, df_rs_cublock = None, None
    if 'cublock' in args.algorithms:
        trans_rs = fit_cublock_translator(train_ag, train_ngs)
        df_rs_cublock = translate_cublock(test_ag, trans_rs) if test_ag is not None else None
        trans_ma = fit_cublock_translator(train_ngs, train_ag)
        df_ma_cublock = translate_cublock(test_ngs, trans_ma) if test_ngs is not None else None
    
    # TDM
    df_ma_tdm, df_rs_tdm = None, None
    if 'tdm' in args.algorithms:
        res_tdm = tdm_normalize(train_ag, test_ag, train_ngs, test_ngs)
        df_ma_tdm, df_rs_tdm = res_tdm['rnaseq_to_microarray'], res_tdm['microarray_to_rnaseq']
    
    # Quantile
    df_ma_qn, df_rs_qn = None, None
    if 'qn' in args.algorithms:
        res_qn = quantile_normalize(train_ag, test_ag, train_ngs, test_ngs)
        df_ma_qn, df_rs_qn = res_qn['rnaseq_to_microarray'], res_qn['microarray_to_rnaseq']

    # 3. Save Algorithm Profiles
    if args.ext_id:
        algo_dir = os.path.join(sync_root, f"algorithms_{args.ext_id}")
    else:
        algo_dir = os.path.join(sync_root, "algorithms")
    os.makedirs(algo_dir, exist_ok=True)

    algos = {
        'combat': (df_ma_combat, df_rs_combat),
        'yugene': (df_ma_yugene, df_rs_yugene),
        'cublock': (df_ma_cublock, df_rs_cublock),
        'tdm': (df_ma_tdm, df_rs_tdm),
        'qn': (df_ma_qn, df_rs_qn)
    }

    for name, data in algos.items():
        if data is None: continue
        ma, rs = data
        if ma is not None: ma.to_csv(os.path.join(algo_dir, f'microarray_fake_{name}.csv'))
        if rs is not None: rs.to_csv(os.path.join(algo_dir, f'rnaseq_fake_{name}.csv'))

    # 4. Aggregate Metrics
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
        perf = measure_perf_detailed(real, fake)
        perf['Algorithm'] = name
        final_results.append(perf)
        
    df_final = pd.DataFrame(final_results)
    
    if args.ext_id:
        out_dir = os.path.join(sync_root, args.ext_id) # Save in ext_id folder as per "subfolder of ext1"
    else:
        out_dir = resolve_path(os.path.join("results", "3_ComparativeAnalysis", args.run_id))
    
    os.makedirs(out_dir, exist_ok=True)
    df_final.to_csv(os.path.join(out_dir, "Test_performance.csv"), index=False)
    
    print(f"Results saved to {out_dir}/Test_performance.csv")
    print(df_final[['Algorithm', 'Pearson', 'Spearman', 'L1']])

if __name__ == "__main__":
    main()
