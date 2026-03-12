import numpy as np
import pandas as pd
from scipy import stats
import random
import gseapy

def get_enrichr_gene_sets(library='KEGG_2021_Human'):
    """
    Fetch gene sets from Enrichr using gseapy.
    Common libraries: 'KEGG_2021_Human', 'GO_Biological_Process_2021', 'MSigDB_Hallmark_2020'
    """
    try:
        # Returns a dict {set_name: [genes]}
        gs = gseapy.get_library(library)
        return gs
    except Exception as e:
        print(f"Error fetching {library} from Enrichr: {e}")
        return {}

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
    # Make a copy and work with uppercase index for matching
    df = deg_df.copy()
    if 'cohen_d' not in df.columns:
        # Fallback if cohen_d is missing
        df['stat'] = 0
    elif how == 'abs_d':
        df['stat'] = df['cohen_d'].abs()
    else:
        df['stat'] = df['cohen_d']
    
    # Gene names in dataset (ensure uppercase for matching)
    df.index = df.index.astype(str).str.upper()
    universe_genes = set(df.index)
    
    for name, genes in gene_sets.items():
        # Match uppercase genes
        genes_upper = [str(g).upper() for g in genes]
        set_genes = list(set(genes_upper) & universe_genes)
        
        if len(set_genes) < 3: 
            continue
            
        score = df.loc[set_genes, 'stat'].mean()
        results.append({'set': name, 'score': score, 'count': len(set_genes)})
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['rank'] = res_df['score'].rank(ascending=False)
    else:
        return pd.DataFrame(columns=['set', 'score', 'rank', 'count'])
    return res_df

def ora_enrichment(deg_df, gene_sets, threshold=0.05):
    """
    Perform Over-Representation Analysis (ORA) using Fisher's Exact Test via gseapy.
    This mimics the DAVID functional annotation process.
    """
    # Identify DEGs (e.g., FDR < 0.05)
    degs = deg_df[deg_df['fdr'] < threshold].index.astype(str).unique().tolist()
    background = deg_df.index.astype(str).unique().tolist()
    
    if len(degs) < 5:
        # Not enough DEGs to run meaningful ORA
        return pd.DataFrame(columns=['set', 'p_value', 'fdr', 'overlap', 'rank', 'genes'])
        
    try:
        enr = gseapy.enrich(gene_list=degs,
                            gene_sets=gene_sets,
                            background=background,
                            outdir=None,
                            no_plot=True)
        
        res = enr.results
        if res.empty:
            return pd.DataFrame(columns=['set', 'p_value', 'fdr', 'overlap', 'rank', 'genes'])

        # Map gseapy columns to our expected format
        res_df = res[['Term', 'P-value', 'Adjusted P-value', 'Overlap', 'Genes']].rename(columns={
            'Term': 'set',
            'P-value': 'p_value',
            'Adjusted P-value': 'fdr',
            'Overlap': 'overlap',
            'Genes': 'genes'
        })
        # Rank by p-value (lowest p-value = rank 1)
        res_df['rank'] = res_df['p_value'].rank(ascending=True)
        return res_df
    except Exception as e:
        print(f"Error in ORA enrichment: {e}")
        return pd.DataFrame(columns=['set', 'p_value', 'fdr', 'overlap', 'rank', 'genes'])

def spearman_rank_concordance(df_real, df_syn):
    """
    Correlate pathway ranks between real and synthetic enrichment tables.
    """
    if df_real.empty or df_syn.empty or 'set' not in df_real.columns or 'set' not in df_syn.columns:
        return np.nan
        
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
            # Ensure we don't sample more than available
            sample_size = min(len(genes), len(universe))
            perm_sets[name] = random.sample(universe, sample_size)
            
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
    # Ensure indexes are strings for set operations
    dr = deg_real.copy()
    ds = deg_syn.copy()
    dr.index = dr.index.astype(str)
    ds.index = ds.index.astype(str)

    results = []
    for tau in thresholds:
        set_real = set(dr[dr['fdr'] <= tau].index)
        set_syn = set(ds[ds['fdr'] <= tau].index)
        
        inter = len(set_real & set_syn)
        union = len(set_real | set_syn)
        jac = inter / union if union > 0 else 0.0
        results.append({'threshold': tau, 'jaccard': jac, 'n_real': len(set_real)})
        
    return pd.DataFrame(results)
