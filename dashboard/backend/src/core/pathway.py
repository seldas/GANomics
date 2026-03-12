import numpy as np
import pandas as pd
from scipy import stats
import random
import gseapy
from pathlib import Path

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
    if path is None:
        return gene_sets
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 2:
                    name = parts[0]
                    genes = parts[2:]
                    gene_sets[name] = set(genes)
    except Exception as e:
        print(f"Error loading GMT {path}: {e}")
    return gene_sets

def bh_fdr(pvals: np.ndarray):
    """Benjamini–Hochberg FDR for a 1D array; returns q-values in original order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0: return p
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    q = (p[order] * n) / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out

def ci95(x):
    """Compute bootstrap 95% confidence interval."""
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan, np.nan, np.nan)
    return (np.nanmean(x), np.nanstd(x, ddof=1), 
            np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))

def glass_delta(stat, null):
    """Compute Glass's Delta effect size."""
    null = np.asarray(null); null = null[np.isfinite(null)]
    if null.size < 2 or not np.isfinite(stat): return np.nan
    return (stat - np.nanmean(null)) / np.nanstd(null, ddof=1)

def gene_set_enrichment(effect_df: pd.DataFrame, gene_sets: dict, how='abs_d'):
    """
    Highly optimized set-level enrichment using vectorized Welch t-test.
    """
    if 'cohen_d' not in effect_df.columns:
        return pd.DataFrame(columns=['set', 'k', 't', 'p', 'q', 'mean_in', 'mean_out', 'rank'])

    gene_to_stat = effect_df['cohen_d'].astype(float)
    if how == 'abs_d':
        vec = np.abs(gene_to_stat)
    elif how == 'signed_d':
        vec = gene_to_stat
    else:
        raise ValueError("how must be 'abs_d' or 'signed_d'")

    vec = vec.dropna()
    all_genes = set(vec.index)
    
    # Pre-calculate global stats for background optimization
    global_mean = vec.mean()
    global_var = vec.var(ddof=1)
    global_n = len(vec)
    
    rows = []
    for name, members in gene_sets.items():
        set_members = set(members)
        g = list(all_genes & set_members)
        nk = len(g)
        if nk < 5: 
            continue
            
        # In-set stats
        xk = vec.loc[g].values
        mk = np.mean(xk)
        vk = np.var(xk, ddof=1)
        if vk == 0: vk = 1e-9 # Avoid division by zero
        
        # Background stats (everything NOT in the set)
        # Using formula for mean/var of complement set to avoid re-scanning all genes
        # m_bg = (sum_all - sum_k) / (n_all - n_k)
        # sum_sq_bg = sum_sq_all - sum_sq_k
        nbg = global_n - nk
        if nbg < 2: continue
        
        # Although complement formula is faster, for stability we'll just slice
        # (Slicing 20k genes is actually very fast in pandas/numpy)
        bg_genes = list(all_genes - set_members)
        yk = vec.loc[bg_genes].values
        mbg = np.mean(yk)
        vbg = np.var(yk, ddof=1)
        if vbg == 0: vbg = 1e-9

        # Welch's T
        se = np.sqrt(vk/nk + vbg/nbg)
        t = (mk - mbg) / se
        df_welch = (vk/nk + vbg/nbg)**2 / ((vk/nk)**2/(nk-1) + (vbg/nbg)**2/(nbg-1))
        p = 2 * stats.t.sf(np.abs(t), df_welch)
        
        rows.append((name, nk, t, p, mk, mbg))
        
    if not rows:
        return pd.DataFrame(columns=['set', 'k', 't', 'p', 'q', 'mean_in', 'mean_out', 'rank'])
        
    df = pd.DataFrame(rows, columns=['set', 'k', 't', 'p', 'mean_in', 'mean_out'])
    df['q'] = bh_fdr(df['p'].values)
    df.sort_values('p', inplace=True)
    df['rank'] = np.arange(1, len(df)+1)
    return df

def spearman_rank_concordance(df_real, df_syn):
    """
    Correlate pathway ranks between two enrichment tables.
    """
    if df_real is None or df_syn is None or len(df_real)==0 or len(df_syn)==0:
        return np.nan
        
    if not {"set","rank"}.issubset(df_real.columns) or not {"set","rank"}.issubset(df_syn.columns):
        return np.nan
        
    common = set(df_real["set"]) & set(df_syn["set"])
    if len(common) < 3:
        return np.nan
        
    idx = sorted(list(common))
    dr_idx = df_real.set_index("set")
    ds_idx = df_syn.set_index("set")
    
    a = dr_idx.loc[idx]["rank"]
    b = ds_idx.loc[idx]["rank"]
    
    # lower rank = more enriched → negate so larger = better for correlation
    rho, _ = stats.spearmanr(-a.to_numpy(), -b.to_numpy())
    return rho

def permute_gene_sets_once(gene_sets, universe):
    """Generate one randomized version of gene sets."""
    perm_sets = {}
    univ_list = list(universe)
    n_univ = len(univ_list)
    for name, members in gene_sets.items():
        size = min(len(members), n_univ)
        perm_sets[name] = random.sample(univ_list, size)
    return perm_sets

def gene_set_preservation_permutation(effect_real, effect_syn, gene_sets, B=50, how='abs_d', random_seed=3):
    """
    Build a null for rank concordance by permuting gene-set membership.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    df_r = gene_set_enrichment(effect_real, gene_sets, how=how)
    df_s = gene_set_enrichment(effect_syn,  gene_sets, how=how)
    obs_rho = spearman_rank_concordance(df_r, df_s)
    
    if not np.isfinite(obs_rho) or df_r.empty:
        return obs_rho, np.array([]), np.nan, df_r, df_s

    universe = set(effect_syn.index) | set(effect_real.index)
    
    null = []
    for _ in range(B):
        perm_syn_sets = permute_gene_sets_once(gene_sets, universe)
        df_perm = gene_set_enrichment(effect_syn, perm_syn_sets, how=how)
        rho = spearman_rank_concordance(df_r, df_perm)
        if np.isfinite(rho):
            null.append(rho)
            
    null = np.array(null)
    p = (np.sum(null >= obs_rho) + 1.0) / (len(null) + 1.0) if null.size > 0 else np.nan
    return obs_rho, null, p, df_r, df_s

def perm_pvalue(null, stat, side="greater"):
    """Permutation p-value."""
    null = np.asarray(null); null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(stat): return np.nan
    if side == "greater":
        return (np.sum(null >= stat) + 1.0) / (null.size + 1.0)
    elif side == "less":
        return (np.sum(null <= stat) + 1.0) / (null.size + 1.0)
    else:
        p1 = (np.sum(null >= stat) + 1.0) / (null.size + 1.0)
        p2 = (np.sum(null <= stat) + 1.0) / (null.size + 1.0)
        return 2*min(p1, p2)

def topK_overlap(a_df, b_df, K=20):
    """Jaccard overlap of top-K gene sets."""
    if a_df.empty or b_df.empty: return 0.0
    def get_top(df):
        sort_col = 'p' if 'p' in df.columns else 'rank'
        return set(df.sort_values(sort_col).head(K)['set'])
        
    A = get_top(a_df); B = get_top(b_df)
    union_len = len(A | B)
    return len(A & B) / union_len if union_len > 0 else 0.0

def jaccard_topk_mc(M, K, B=10000, rng=None):
    """Monte Carlo null for Jaccard overlap of Top-K sets."""
    if rng is None: rng = np.random.default_rng(7)
    if M <= 0 or K <= 0 or K > M: return np.zeros(B)
    ints = rng.hypergeometric(ngood=K, nbad=M-K, nsample=K, size=B)
    j = ints / (2*K - ints)
    return j

def expected_jaccard_random(M, K):
    """Expected Jaccard for two random Top-K sets."""
    if M <= 0: return 0.0
    f = K / M
    return f / (2 - f)

def bootstrap_pathway_rank_stability(data_real, data_syn, labels, gene_sets, B=50, frac=0.8, how='abs_d', K_list=(10,20,50)):
    """
    Bootstrap samples and recompute pathway ranks.
    """
    from .analysis import run_deg_analysis
    
    idx = data_real.index.values
    rhos = []
    topk = {K: [] for K in K_list}
    n = len(idx)

    for _ in range(B):
        boot_idx = np.random.choice(idx, size=int(frac*n), replace=True)
        # Use .loc with boot_idx
        lab_b = labels.loc[boot_idx]
        
        # Compute DEGs
        de_r = run_deg_analysis(data_real.loc[boot_idx], lab_b)
        de_s = run_deg_analysis(data_syn.loc[boot_idx], lab_b)
        
        # Compute Enrichment
        enr_r = gene_set_enrichment(de_r, gene_sets, how=how)
        enr_s = gene_set_enrichment(de_s, gene_sets, how=how)

        # Spearman Rho
        rho = spearman_rank_concordance(enr_r, enr_s)
        if np.isfinite(rho):
            rhos.append(rho)

        # Top-K Overlaps
        for K in K_list:
            tk = topK_overlap(enr_r, enr_s, K=K)
            topk[K].append(tk)

    return np.array(rhos), {K: np.array(v) for K, v in topk.items()}

def jaccard_threshold_curve(deg_real, deg_syn, thresholds=[1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0]):
    """Compute Jaccard overlap of DEGs across thresholds."""
    dr = deg_real.copy(); ds = deg_syn.copy()
    dr.index = dr.index.astype(str); ds.index = ds.index.astype(str)
    
    results = []
    for tau in thresholds:
        p_col_r = 'p_value' if 'p_value' in dr.columns else ('p' if 'p' in dr.columns else None)
        p_col_s = 'p_value' if 'p_value' in ds.columns else ('p' if 'p' in ds.columns else None)

        set_real = set(dr[dr[p_col_r] <= tau].index) if p_col_r else set()
        set_syn = set(ds[ds[p_col_s] <= tau].index) if p_col_s else set()
        
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
