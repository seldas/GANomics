import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import pickle

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.genomics_dataset import GenomicsDataset
from src.models.ganomics_model import GANomicsModel
from src.core.evaluation import compute_metrics
from src.bio_utils import (
    combat_evaluate_paired, fit_cublock_translator, translate_cublock,
    quantile_normalize, tdm_normalize, yugene_evaluate_paired
)

def measure_perf_detailed(df_real, df_fake):
    """
    Compute per-sample metrics and return summary with CI.
    """
    metrics = []
    for i in range(len(df_real)):
        m = compute_metrics(df_real.values[i:i+1], df_fake.values[i:i+1])
        metrics.append(m)
    
    df_metrics = pd.DataFrame(metrics)
    summary = {}
    for col in df_metrics.columns:
        mean = df_metrics[col].mean()
        std = df_metrics[col].std()
        summary[col] = f"{mean:.3f} (±{std:.3f})"
    return summary

def main():
    parser = argparse.ArgumentParser(description="Comparative Analysis with Baseline Methods")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--project", type=str, default="NB", help="Project name")
    parser.add_argument("--sample_size", type=int, default=50, help="Training size")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--no_adjust_path", action='store_true', help="Don't adjust PYTHONPATH")
    parser.add_argument("--save_full", action='store_true', help="Save full results")
    args = parser.parse_args()
    
    if not args.no_adjust_path:
        # Add the parent directory to sys.path to make 'src' importable
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data and Split
    train_dir = os.path.join("results", "2_SyncData", f"{args.project}_{args.sample_size}_{args.run_id}", "train")
    train_ag = pd.read_csv(os.path.join(train_dir, "microarray_real.csv"), index_col=0)
    train_ngs = pd.read_csv(os.path.join(train_dir, "rnaseq_real.csv"), index_col=0)
    test_dir = os.path.join("results", "2_SyncData", f"{args.project}_{args.sample_size}_{args.run_id}", "test")
    test_ag = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
    test_ngs = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
    
    # 2. GANomics Results
    sync_dir = os.path.join("results", "2_SyncData", f"{args.project}_{args.sample_size}_{args.run_id}", "test")
    df_ma_fake = pd.read_csv(os.path.join(sync_dir, "microarray_fake.csv"), index_col=0)
    df_rs_fake = pd.read_csv(os.path.join(sync_dir, "rnaseq_fake.csv"), index_col=0)
    
    # 3. Run Baselines
    print("Running Baseline Comparisons...")
    
    # ComBat
    res_combat = combat_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
    df_ma_combat = res_combat['rnaseq_to_microarray']
    df_rs_combat = res_combat['microarray_to_rnaseq']
    
    # YuGene
    res_yugene = yugene_evaluate_paired(train_ag, train_ngs, test_ag, test_ngs)
    df_ma_yugene = res_yugene['rnaseq_to_microarray']
    df_rs_yugene = res_yugene['microarray_to_rnaseq']
    
    # CuBlock
    trans_rs = fit_cublock_translator(train_ag, train_ngs)
    df_rs_cublock = translate_cublock(test_ag, trans_rs)
    trans_ma = fit_cublock_translator(train_ngs, train_ag)
    df_ma_cublock = translate_cublock(test_ngs, trans_ma)
    
    # TDM
    res_tdm = tdm_normalize(train_ag, test_ag, train_ngs, test_ngs)
    df_ma_tdm = res_tdm['rnaseq_to_microarray']
    df_rs_tdm = res_tdm['microarray_to_rnaseq']
    
    # Quantile
    res_qn = quantile_normalize(train_ag, test_ag, train_ngs, test_ngs)
    df_ma_qn = res_qn['rnaseq_to_microarray']
    df_rs_qn = res_qn['microarray_to_rnaseq']

    # save fake data of all algorithm into the sync_data folder:
    sync_dir_fake = os.path.join("results", "2_SyncData", f"{args.project}_{args.sample_size}_{args.run_id}", "algorithms")
    if not os.path.exists(sync_dir_fake):
        os.makedirs(sync_dir_fake, exist_ok=True)

    df_ma_combat.to_csv(os.path.join(sync_dir_fake, 'microarray_fake_combat.csv'))
    df_ma_yugene.to_csv(os.path.join(sync_dir_fake, 'microarray_fake_yugene.csv'))
    df_ma_cublock.to_csv(os.path.join(sync_dir_fake, 'microarray_fake_cublock.csv'))
    df_ma_tdm.to_csv(os.path.join(sync_dir_fake, 'microarray_fake_tdm.csv'))
    df_ma_qn.to_csv(os.path.join(sync_dir_fake, 'microarray_fake_qn.csv'))

    df_ma_combat.to_csv(os.path.join(sync_dir_fake, 'rnaseq_fake_combat.csv'))
    df_ma_yugene.to_csv(os.path.join(sync_dir_fake, 'rnaseq_fake_yugene.csv'))
    df_ma_cublock.to_csv(os.path.join(sync_dir_fake, 'rnaseq_fake_cublock.csv'))
    df_ma_tdm.to_csv(os.path.join(sync_dir_fake, 'rnaseq_fake_tdm.csv'))
    df_ma_qn.to_csv(os.path.join(sync_dir_fake, 'rnaseq_fake_qn.csv'))

    # 4. Aggregate Metrics
    comparisons = [
        ("GANomics (MA)", test_ag, df_ma_fake),
        ("ComBat (MA)",   test_ag, df_ma_combat),
        ("YuGene (MA)",   test_ag, df_ma_yugene),
        ("CuBlock (MA)",  test_ag, df_ma_cublock),
        ("TDM (MA)",      test_ag, df_ma_tdm),
        ("Quantile (MA)", test_ag, df_ma_qn),
        ("GANomics (RS)", test_ngs, df_rs_fake),
        ("ComBat (RS)",   test_ngs, df_rs_combat),
        ("YuGene (RS)",   test_ngs, df_rs_yugene),
        ("CuBlock (RS)",  test_ngs, df_rs_cublock),
        ("TDM (RS)",      test_ngs, df_rs_tdm),
        ("Quantile (RS)", test_ngs, df_rs_qn),
        ("Baseline (paired)", test_ag, test_ngs)
    ]
    
    final_results = []
    for name, real, fake in comparisons:
        perf = measure_perf_detailed(real, fake)
        perf['Algorithm'] = name
        final_results.append(perf)
        
    df_final = pd.DataFrame(final_results)
    output_path = os.path.join("results", "3_ComparativeAnalysis", f"Table_2_Comparison_{args.project}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nComparative analysis saved to {output_path}")
    print(df_final[['Algorithm', 'Pearson', 'Spearman', 'L1']])

if __name__ == "__main__":
    main()
