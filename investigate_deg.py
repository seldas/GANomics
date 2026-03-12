import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

# Re-implement run_deg_analysis locally for speed and transparency
def run_deg(df, labels, group1=1, group2=0):
    g1 = df.loc[labels == group1]
    g2 = df.loc[labels == group2]
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(axis=0), g2.mean(axis=0)
    v1, v2 = g1.var(axis=0, ddof=1), g2.var(axis=0, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    se = np.where(se == 0, np.nan, se)
    t_stats = (m1 - m2) / se
    df_num = (v1/n1 + v2/n2)**2
    df_den = (v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)
    dof = df_num / df_den
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)
    return pd.DataFrame({'p_value': p_values}, index=df.columns)

def get_jaccard(deg1, deg2, tau=0.05):
    s1 = set(deg1[deg1['p_value'] <= tau].index)
    s2 = set(deg2[deg2['p_value'] <= tau].index)
    if not s1 or not s2: return 0.0, len(s1), len(s2), 0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union, len(s1), len(s2), inter

def main():
    invest_dir = "DEG_investigation"
    label_path = "dashboard/backend/dataset/NB/label.txt"
    ma_real_path = "dashboard/backend/results/2_SyncData/NB_Ablation_Size_50_Run_0/test/microarray_real.csv"

    # 1. Load Labels
    df_labels = pd.read_csv(label_path, index_col=0)
    
    # 2. Load Real RNA-Seq (Target)
    rs_real = pd.read_csv(os.path.join(invest_dir, "rnaseq_real.csv"), index_col=0)
    
    # 3. Load others
    rs_fake_gan = pd.read_csv(os.path.join(invest_dir, "rnaseq_fake.csv"), index_col=0)
    rs_fake_combat = pd.read_csv(os.path.join(invest_dir, "rnaseq_fake_combat.csv"), index_col=0)
    rs_fake_tdm = pd.read_csv(os.path.join(invest_dir, "rnaseq_fake_tdm.csv"), index_col=0)
    ma_real = pd.read_csv(ma_real_path, index_col=0)

    # Intersection of samples
    common_samples = rs_real.index.intersection(df_labels.index).intersection(ma_real.index)
    print(f"Common samples: {len(common_samples)}")
    
    y = df_labels.loc[common_samples, 'label']
    rs_real = rs_real.loc[common_samples]
    rs_fake_gan = rs_fake_gan.loc[common_samples]
    rs_fake_combat = rs_fake_combat.loc[common_samples]
    rs_fake_tdm = rs_fake_tdm.loc[common_samples]
    ma_real = ma_real.loc[common_samples]

    # Calculate DEGs
    print("Computing DEGs...")
    deg_rs_real = run_deg(rs_real, y)
    deg_rs_gan = run_deg(rs_fake_gan, y)
    deg_rs_combat = run_deg(rs_fake_combat, y)
    deg_rs_tdm = run_deg(rs_fake_tdm, y)
    deg_ma_real = run_deg(ma_real, y)

    # Intersection of genes (just in case)
    common_genes = deg_rs_real.index.intersection(deg_ma_real.index)
    print(f"Common genes: {len(common_genes)}")
    
    deg_rs_real = deg_rs_real.loc[common_genes]
    deg_rs_gan = deg_rs_gan.loc[common_genes]
    deg_rs_combat = deg_rs_combat.loc[common_genes]
    deg_rs_tdm = deg_rs_tdm.loc[common_genes]
    deg_ma_real = deg_ma_real.loc[common_genes]

    thresholds = [0.001, 0.01, 0.05, 0.1]
    
    results = []
    for tau in thresholds:
        # Baseline: MA Real vs RS Real
        jac_base, n1, n2, inter = get_jaccard(deg_ma_real, deg_rs_real, tau)
        results.append({'threshold': tau, 'algo': 'Baseline (MA vs RS)', 'jaccard': jac_base, 'n_real': n1, 'n_other': n2, 'overlap': inter})
        
        # GANomics: RS Fake vs RS Real
        jac_gan, n1, n2, inter = get_jaccard(deg_rs_gan, deg_rs_real, tau)
        results.append({'threshold': tau, 'algo': 'GANomics (Fake RS vs Real RS)', 'jaccard': jac_gan, 'n_real': n1, 'n_other': n2, 'overlap': inter})

        # ComBat: RS Fake vs RS Real
        jac_combat, n1, n2, inter = get_jaccard(deg_rs_combat, deg_rs_real, tau)
        results.append({'threshold': tau, 'algo': 'ComBat (Fake RS vs Real RS)', 'jaccard': jac_combat, 'n_real': n1, 'n_other': n2, 'overlap': inter})

        # TDM: RS Fake vs RS Real
        jac_tdm, n1, n2, inter = get_jaccard(deg_rs_tdm, deg_rs_real, tau)
        results.append({'threshold': tau, 'algo': 'TDM (Fake RS vs Real RS)', 'jaccard': jac_tdm, 'n_real': n1, 'n_other': n2, 'overlap': inter})

    df_res = pd.DataFrame(results)
    print(df_res.to_string())

    # Check if data looks normalized
    print("\nData Range Check (First 5 genes):")
    print(f"MA Real mean: {ma_real.iloc[:, :5].mean().mean():.3f}, std: {ma_real.iloc[:, :5].std().mean():.3f}")
    print(f"RS Real mean: {rs_real.iloc[:, :5].mean().mean():.3f}, std: {rs_real.iloc[:, :5].std().mean():.3f}")
    print(f"GANomics mean: {rs_fake_gan.iloc[:, :5].mean().mean():.3f}, std: {rs_fake_gan.iloc[:, :5].std().mean():.3f}")

if __name__ == "__main__":
    main()
