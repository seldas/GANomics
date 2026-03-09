import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(real, synthetic):
    """
    Compute standard benchmarks for genomic translation.
    real, synthetic: (n_samples, n_genes) numpy arrays
    """
    # Per-sample Pearson and Spearman
    pearsons = []
    spearmans = []
    for i in range(real.shape[0]):
        r, _ = pearsonr(real[i], synthetic[i])
        s, _ = spearmanr(real[i], synthetic[i])
        pearsons.append(r)
        spearmans.append(s)
        
    avg_pearson = np.mean(pearsons)
    avg_spearman = np.mean(spearmans)
    
    # Global errors
    mae = mean_absolute_error(real, synthetic)
    rmse = np.sqrt(mean_squared_error(real, synthetic))
    
    # L1, L2 (as used in the paper)
    l1 = np.mean(np.abs(real - synthetic).sum(axis=1))
    l2 = np.mean(np.sqrt(((real - synthetic)**2).sum(axis=1)))
    
    return {
        'Pearson': avg_pearson,
        'Spearman': avg_spearman,
        'MAE': mae,
        'RMSE': rmse,
        'L1': l1,
        'L2': l2
    }

def benchmark_all_methods(real_A, real_B, syn_A, syn_B, baselines={}):
    """
    Generate comparison table against other algorithms.
    """
    results = []
    
    # 1. GANomics A -> B
    metrics_rs = compute_metrics(real_B, syn_B)
    metrics_rs['Algorithm'] = 'GANomics (RS)'
    results.append(metrics_rs)
    
    # 2. GANomics B -> A
    metrics_ma = compute_metrics(real_A, syn_A)
    metrics_ma['Algorithm'] = 'GANomics (MA)'
    results.append(metrics_ma)
    
    # 3. Baselines (TDM, QN, ComBat, YuGene, CuBlock)
    for name, b_data in baselines.items():
        # b_data: dict with 'ma_to_rs' and 'rs_to_ma' dataframes
        m_rs = compute_metrics(real_B, b_data['ma_to_rs'].values)
        m_rs['Algorithm'] = f'{name} (RS)'
        results.append(m_rs)
        
        m_ma = compute_metrics(real_A, b_data['rs_to_ma'].values)
        m_ma['Algorithm'] = f'{name} (MA)'
        results.append(m_ma)
        
    return pd.DataFrame(results)
