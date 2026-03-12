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

def gene_set_enrichment(deg_df, gene_sets, how='abs_d', min_size=15, max_size=500):
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
        
        # Standard filtering: exclude too small or too large/generic sets
        if len(set_genes) < min_size or len(set_genes) > max_size:
            continue
            
        score = df.loc[set_genes, 'stat'].mean()
        results.append({'set': name, 'score': score, 'count': len(set_genes)})
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['rank'] = res_df['score'].rank(ascending=False)
    else:
        return pd.DataFrame(columns=['set', 'score', 'rank', 'count'])
    return res_df

def ora_enrichment(deg_df, gene_sets, threshold=0.05, min_size=15, max_size=500):
    """
    Perform Over-Representation Analysis (ORA) using Fisher's Exact Test via gseapy.
    This mimics the DAVID functional annotation process.
    """
    # Identify DEGs using raw p-value instead of FDR for more sensitivity
    # Ensure symbols are string and uppercase
    df = deg_df.copy()
    df.index = df.index.astype(str).str.upper()

    degs = df[df['p_value'] < threshold].index.unique().tolist()
    background = df.index.unique().tolist()

    if len(degs) < 5:
        return pd.DataFrame(columns=['set', 'p_value', 'fdr', 'overlap', 'rank', 'genes'])

    try:
        # Pre-filter gene sets to remove generic/large ones and tiny ones
        filtered_sets = {}
        bg_set = set(background)
        for name, genes in gene_sets.items():
            # Match gene set symbols to background (both uppercase)
            genes_upper = [str(g).upper() for g in genes]
            intersect_size = len(set(genes_upper) & bg_set)
            if min_size <= intersect_size <= max_size:
                filtered_sets[name] = genes_upper

        if not filtered_sets:
            return pd.DataFrame(columns=['set', 'p_value', 'fdr', 'overlap', 'rank', 'genes'])

        enr = gseapy.enrich(gene_list=degs,
                            gene_sets=filtered_sets,
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

def jaccard_topk_mc(M, K, B=5000):
    """
    Monte Carlo simulation for Jaccard overlap of two random Top-K sets.
    M: total number of pathways, K: size of top set, B: iterations.
    """
    if M <= 0 or K <= 0 or K > M: return np.zeros(B)
    rng = np.random.default_rng(42)
    # Draw two random K-sets from M items; compute Jaccard = |∩| / |∪|
    # Hypergeometric draws the size of intersection
    ints = rng.hypergeometric(ngood=K, nbad=M-K, nsample=K, size=B)
    j = ints / (2*K - ints)
    return j

def expected_jaccard_random(M, K):
    """Analytical baseline for random Jaccard overlap."""
    if M <= 0: return 0.0
    f = K / M
    return f / (2 - f)

def perm_pvalue(null, stat, side="greater"):
    null = np.asarray(null); null = null[null != np.nan]
    if null.size == 0 or not np.isfinite(stat): return np.nan
    if side == "greater":
        return (np.sum(null >= stat) + 1.0) / (null.size + 1.0)
    elif side == "less":
        return (np.sum(null <= stat) + 1.0) / (null.size + 1.0)
    return np.nan

def run_permutation_test(deg_real, deg_syn, gene_sets, B=100, min_size=15, max_size=500):
    """
    Permutation test: randomize gene-set membership to build a null distribution
    for the Spearman rank concordance (rho).
    """
    obs_rho = spearman_rank_concordance(
        gene_set_enrichment(deg_real, gene_sets, min_size=min_size, max_size=max_size),
        gene_set_enrichment(deg_syn, gene_sets, min_size=min_size, max_size=max_size)
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
            gene_set_enrichment(deg_real, gene_sets, min_size=min_size, max_size=max_size),
            gene_set_enrichment(deg_syn, perm_sets, min_size=min_size, max_size=max_size)
        )
        if np.isfinite(rho):
            null.append(rho)
            
    p_val = (np.sum(np.array(null) >= obs_rho) + 1.0) / (len(null) + 1.0)
    return obs_rho, np.array(null), p_val

def jaccard_threshold_curve(deg_real, deg_syn, thresholds=[1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]):
    """
    Compute Jaccard overlap of DEGs across a range of p-value thresholds.
    """
    # Ensure indexes are strings for set operations
    dr = deg_real.copy()
    ds = deg_syn.copy()
    dr.index = dr.index.astype(str)
    ds.index = ds.index.astype(str)

    results = []
    for tau in thresholds:
        set_real = set(dr[dr['p_value'] <= tau].index)
        set_syn = set(ds[ds['p_value'] <= tau].index)
        
        inter = len(set_real & set_syn)
        union = len(set_real | set_syn)
        jac = inter / union if union > 0 else 0.0
        results.append({
            'threshold': tau, 
            'jaccard': jac, 
            'n_real': len(set_real),
            'n_fake': len(set_syn),
            'n_overlap': inter
        })
        
    return pd.DataFrame(results)
