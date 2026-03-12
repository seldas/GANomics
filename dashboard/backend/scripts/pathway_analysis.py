import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis
from src.core.pathway import (
    load_gmt,
    gene_set_enrichment,
    spearman_rank_concordance, 
    gene_set_preservation_permutation,
    bootstrap_pathway_rank_stability,
    ci95,
    glass_delta,
    perm_pvalue,
    jaccard_topk_mc,
    expected_jaccard_random,
    topK_overlap
)

def main():
    parser = argparse.ArgumentParser(description="Focused Hallmark Pathway Analysis")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--ext_id", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--bootstrap_b", type=int, default=50)
    args = parser.parse_args()

    print(f"🚀 Starting Pathway Analysis for {args.run_id} (Hallmark ONLY, B={args.bootstrap_b})")

    # Paths Setup
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sync_root = os.path.join(backend_dir, "results", "2_SyncData", args.run_id)
    project_id = args.run_id.split('_')[0]
    
    # HARDCODED: ONLY use this specific GMT file
    gmt_file = os.path.join(backend_dir, "src", "db", "h.all.v2026.1.Hs.symbols.gmt")
    
    if args.ext_id:
        test_dir = os.path.join(sync_root, args.ext_id)
        out_root = test_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        out_root = os.path.join(backend_dir, "results", "4_Biomarkers")

    # Load Data
    try:
        print(f"📂 Loading data from {test_dir}...")
        ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    print(f"🏷️ Loading labels from {label_path}...")
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = ma_real.index.intersection(rs_real.index).intersection(df_labels.index)
    
    if len(common_idx) == 0:
        print("❌ Error: No common samples between data and labels.")
        return

    # Ensure samples are aligned
    ma_real = ma_real.loc[common_idx]
    rs_real = rs_real.loc[common_idx]
    ma_fake = ma_fake.loc[common_idx]
    rs_fake = rs_fake.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']

    # Mapping for symbols
    mapping_path = os.path.join(backend_dir, "dataset", project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        print(f"🗺️ Loading gene mapping from {mapping_path}...")
        df_map = pd.read_csv(mapping_path, sep='\t').dropna(subset=['Agilent_Probe_Name', 'GeneSymbol'])
        gene_map = dict(zip(df_map['Agilent_Probe_Name'].astype(str), df_map['GeneSymbol'].astype(str)))
    
    def apply_mapping(df, mapping):
        if mapping is None: return df
        new_df = df.copy()
        new_df.columns = [mapping.get(str(c), str(c)).upper() for c in new_df.columns]
        # Drop UKV/Probes that didn't map
        keep_cols = [c for c in new_df.columns if not (c.startswith(('UKV4_', 'A_')) and len(c) > 10)]
        return new_df[keep_cols]

    print("🧬 Mapping gene symbols...")
    ma_real = apply_mapping(ma_real, gene_map)
    rs_real = apply_mapping(rs_real, gene_map)
    ma_fake = apply_mapping(ma_fake, gene_map)
    rs_fake = apply_mapping(rs_fake, gene_map)

    # Libraries: MSigDB Hallmark ONLY
    if not os.path.exists(gmt_file):
        print(f"❌ CRITICAL ERROR: Hallmark GMT not found at {gmt_file}")
        return
        
    print(f"📖 Loading MSigDB Hallmark: {gmt_file}...")
    msigdb_sets = load_gmt(gmt_file)
    all_gene_sets = {'MSigDB_Hallmark': msigdb_sets}
    
    pathway_dir = os.path.join(out_root, "Pathway", args.run_id) if not args.ext_id else os.path.join(out_root, "Pathway")
    os.makedirs(pathway_dir, exist_ok=True)

    try:
        comparisons = [
            ('Baseline', ma_real, rs_real, "Native Cross-Platform"),
            ('GANomics_MA_to_RS', ma_fake, rs_real, "Microarray -> RNA-Seq"),
            ('GANomics_RS_to_MA', rs_fake, ma_real, "RNA-Seq -> Microarray")
        ]

        for algo_name, data_test, data_ref, desc in comparisons:
            print(f"\n💎 Processing Comparison: {algo_name} ({desc})")
            
            # Initial DEG
            print("  ⚡ Running initial DEG analysis...")
            deg_ref = run_deg_analysis(data_ref, y)
            deg_test = run_deg_analysis(data_test, y)
            
            for gs_name, gene_sets in all_gene_sets.items():
                if not gene_sets: continue
                
                print(f"  📂 Library: {gs_name} ({len(gene_sets)} sets)")
                
                # 1) Preservation Permutation
                print(f"    🎲 Running permutation test (B={args.bootstrap_b})...")
                obs_rho, null_dist, p_perm_obs, enr_real, enr_syn = gene_set_preservation_permutation(
                    deg_ref, deg_test, gene_sets, B=args.bootstrap_b, how='abs_d'
                )
                
                if not np.isfinite(obs_rho) or enr_real.empty:
                    print(f"    ⚠️ Warning: Insufficient data for {gs_name}. Skipping.")
                    continue

                # 2) Bootstrap Rank Stability
                print(f"    🔄 Running bootstrap stability (B={args.bootstrap_b})...")
                boot_rhos, topk_boot = bootstrap_pathway_rank_stability(
                    data_ref, data_test, y, gene_sets, B=args.bootstrap_b, frac=0.8, how='abs_d'
                )
                
                # 3) Statistics
                mu, sd, lo, hi = ci95(boot_rhos)
                p_perm_final = perm_pvalue(null_dist, mu, side="greater")
                g_delta = glass_delta(mu, null_dist)
                
                print(f"    📊 Results: obs_rho={obs_rho:.3f}, boot_mean={mu:.3f}, perm_p={p_perm_final:.4f}")

                # Save Details
                df_detail = pd.merge(
                    enr_real[['set', 'k', 't', 'p', 'q', 'mean_in', 'mean_out']].rename(columns=lambda x: 'Real_'+x.upper() if x!='set' else x),
                    enr_syn[['set', 'k', 't', 'p', 'q', 'mean_in', 'mean_out']].rename(columns=lambda x: 'Syn_'+x.upper() if x!='set' else x),
                    on='set', how='left'
                )
                df_detail = df_detail.sort_values('Real_P', ascending=True)
                df_detail.to_csv(os.path.join(pathway_dir, f"Pathway_Details___{algo_name}___{gs_name}.csv"), index=False)
                
                # Save Summary
                summary_res = {
                    'Algorithm': algo_name, 'Library': gs_name, 'Observed_Rho': obs_rho,
                    'Spearman_Rho': obs_rho, 'Bootstrap_Mean_Rho': mu, 'Bootstrap_SD': sd,
                    'CI_95_Low': lo, 'CI_95_High': hi, 'Permutation_P': p_perm_final, 'Glass_Delta': g_delta
                }
                summary_df = pd.DataFrame([summary_res])
                summary_df.to_csv(os.path.join(pathway_dir, f"Pathway_Summary___{algo_name}___{gs_name}.csv"), index=False)
                summary_df.to_csv(os.path.join(pathway_dir, f"Pathway_Concordance___{algo_name}___{gs_name}.csv"), index=False)
                
                # Save Top-K
                common_sets = set(enr_real['set']) & set(enr_syn['set'])
                M = len(common_sets)
                topk_stats = []
                for K in [10, 20, 50]:
                    if K > M: continue
                    emp_jac = np.nanmean(topk_boot[K]) if K in topk_boot else topK_overlap(enr_real, enr_syn, K=K)
                    null_jac_dist = jaccard_topk_mc(M, K, B=10000)
                    p_jac = perm_pvalue(null_jac_dist, emp_jac, side="greater")
                    exp_jac = expected_jaccard_random(M, K)
                    topk_stats.append({
                        'K': K, 'Observed_Jaccard': emp_jac, 'Expected_Jaccard': exp_jac, 
                        'P_Value': p_jac, 'Spearman_Rho': obs_rho
                    })
                
                if topk_stats:
                    tk_df = pd.DataFrame(topk_stats)
                    tk_df.to_csv(os.path.join(pathway_dir, f"Pathway_TopK___{algo_name}___{gs_name}.csv"), index=False)
                    tk_df.to_csv(os.path.join(pathway_dir, f"Pathway_Stats___{algo_name}___{gs_name}.csv"), index=False)

                # Save Distributions
                dist_df = pd.DataFrame({'Null_Rho': pd.Series(null_dist), 'Bootstrap_Rho': pd.Series(boot_rhos)})
                dist_df.to_csv(os.path.join(pathway_dir, f"Pathway_Distributions___{algo_name}___{gs_name}.csv"), index=False)

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Pathway analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ COMPLETED: Hallmark-only Pathway analysis finished successfully.")

if __name__ == "__main__":
    main()
