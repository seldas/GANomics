import numpy as np
import pandas as pd
from scipy import stats
import random

def load_gmt(path):
    """Load MSigDB GMT file into a dictionary {set_name: [genes]}."""
    gene_sets = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                name = parts[0]
                genes = parts[2:]
                gene_sets[name] = genes
    return gene_sets

def gene_set_enrichment(deg_df, gene_sets, how='abs_d'):
    """
    Simple enrichment scoring: mean of a statistic (e.g., abs Hedges' g)
    for genes in a set vs. a null distribution or ranking.
    """
    results = []
    # Statistic to rank by (e.g., cohen_d or abs(cohen_d))
    if how == 'abs_d':
        deg_df['stat'] = deg_df['cohen_d'].abs()
    else:
        deg_df['stat'] = deg_df['cohen_d']

    universe_genes = set(deg_df.index)
    
    for name, genes in gene_sets.items():
        set_genes = list(set(genes) & universe_genes)
        if len(set_genes) < 5:
            continue
            
        score = deg_df.loc[set_genes, 'stat'].mean()
        results.append({'set': name, 'score': score})
        
    df = pd.DataFrame(results)
    if not df.empty:
        df['rank'] = df['score'].rank(ascending=False)
    return df

def spearman_rank_concordance(df_real, df_syn):
    """
    Correlate pathway ranks between real and synthetic enrichment tables.
    """
    common = set(df_real['set']) & set(df_syn['set'])
    if len(common) < 3:
        return np.nan
        
    idx = sorted(list(common))
    a = df_real.set_index('set').loc[idx, 'rank']
    b = df_syn.set_index('set').loc[idx, 'rank']
    
    # Negative because lower rank = better enrichment
    rho, _ = stats.spearmanr(-a, -b)
    return rho

def run_permutation_test(deg_real, deg_syn, gene_sets, B=1000):
    """
    Permutation test: randomize gene-set membership to build a null distribution
    for the Spearman rank concordance (rho).
    """
    obs_rho = spearman_rank_concordance(
        gene_set_enrichment(deg_real, gene_sets),
        gene_set_enrichment(deg_syn, gene_sets)
    )
    
    universe = list(deg_syn.index)
    null = []
    
    for _ in range(B):
        # Permute gene sets by picking random genes from universe
        perm_sets = {}
        for name, genes in gene_sets.items():
            perm_sets[name] = random.sample(universe, len(genes))
            
        rho = spearman_rank_concordance(
            gene_set_enrichment(deg_real, gene_sets),
            gene_set_enrichment(deg_syn, perm_sets)
        )
        if np.isfinite(rho):
            null.append(rho)
            
    p_val = (np.sum(np.array(null) >= obs_rho) + 1.0) / (len(null) + 1.0)
    return obs_rho, np.array(null), p_val

def jaccard_threshold_curve(deg_real, deg_syn, thresholds=[1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]):
    """
    Compute Jaccard overlap of DEGs across a range of FDR thresholds (Fig 9a).
    """
    results = []
    for tau in thresholds:
        set_real = set(deg_real[deg_real['q'] <= tau].index)
        set_syn = set(deg_syn[deg_syn['q'] <= tau].index)
        
        inter = len(set_real & set_syn)
        union = len(set_real | set_syn)
        jac = inter / union if union > 0 else 0.0
        results.append({'threshold': tau, 'jaccard': jac, 'n_real': len(set_real)})
        
    return pd.DataFrame(results)
