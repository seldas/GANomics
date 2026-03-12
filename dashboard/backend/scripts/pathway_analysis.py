import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis
from src.core.pathway import (
    get_enrichr_gene_sets, ora_enrichment, 
    spearman_rank_concordance, jaccard_topk_mc, 
    expected_jaccard_random, perm_pvalue
)

def main():
    parser = argparse.ArgumentParser(description="Focused Pathway Concordance Analysis")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--ext_id", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--libraries", type=str, nargs="+", default=['KEGG_2021_Human', 'GO_Biological_Process_2021'])
    parser.add_argument("--no_filter", action='store_true')
    args = parser.parse_args()

    # Paths Setup
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sync_root = os.path.join(backend_dir, "results", "2_SyncData", args.run_id)
    project_id = args.run_id.split('_')[0]
    apply_filter = not args.no_filter
    
    if args.ext_id:
        test_dir = os.path.join(sync_root, args.ext_id)
        out_root = test_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        out_root = os.path.join(backend_dir, "results", "4_Biomarkers")

    # Load Data
    try:
        ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
        ma_fake = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
        rs_fake = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)
    except: return

    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = ma_real.index.intersection(rs_real.index).intersection(df_labels.index)
    y = df_labels.loc[common_idx, 'label']

    # Mapping
    mapping_path = os.path.join(backend_dir, "dataset", project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, sep='\t').dropna(subset=['Agilent_Probe_Name', 'GeneSymbol'])
        gene_map = dict(zip(df_map['Agilent_Probe_Name'].astype(str), df_map['GeneSymbol'].astype(str)))
    
    def apply_mapping(deg_df, mapping):
        if mapping is None: return deg_df
        df = deg_df.copy(); df.index = df.index.astype(str)
        df.index = [mapping.get(x, x) for x in df.index]
        return df[~df.index.astype(str).str.startswith(('UKv4_', 'A_'))]

    # Run DEGs for all 4 profiles
    print("Computing DEGs for all profiles...")
    d_ma_real = apply_mapping(run_deg_analysis(ma_real.loc[common_idx], y), gene_map)
    d_rs_real = apply_mapping(run_deg_analysis(rs_real.loc[common_idx], y), gene_map)
    d_ma_fake = apply_mapping(run_deg_analysis(ma_fake.loc[common_idx], y), gene_map)
    d_rs_fake = apply_mapping(run_deg_analysis(rs_fake.loc[common_idx], y), gene_map)

    # Libraries
    all_gene_sets = {lib: get_enrichr_gene_sets(lib) for lib in args.libraries}
    
    pathway_dir = os.path.join(out_root, "Pathway", args.run_id) if not args.ext_id else os.path.join(out_root, "Pathway")
    os.makedirs(pathway_dir, exist_ok=True)

    # Comparison Definition
    comparisons = [
        ('Baseline', d_ma_real, d_rs_real, "Native Cross-Platform"),
        ('GANomics_MA_to_RS', d_rs_fake, d_rs_real, "Microarray -> RNA-Seq"),
        ('GANomics_RS_to_MA', d_ma_fake, d_ma_real, "RNA-Seq -> Microarray")
    ]

    for algo_name, deg_test, deg_ref, desc in comparisons:
        print(f"\n>>> Processing Pathways: {desc}")
        for gs_name, gene_sets in all_gene_sets.items():
            if not gene_sets: continue
            
            ora_ref = ora_enrichment(deg_ref, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            ora_test = ora_enrichment(deg_test, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            
            if ora_ref.empty: continue

            # Merge and Save Details
            df_detail = pd.merge(
                ora_ref[['set', 'p_value', 'overlap', 'genes']].rename(columns={'p_value': 'Real_P', 'overlap': 'Real_Overlap', 'genes': 'Real_Genes'}),
                ora_test[['set', 'p_value', 'overlap', 'genes']].rename(columns={'p_value': 'Syn_P', 'overlap': 'Syn_Overlap', 'genes': 'Syn_Genes'}),
                on='set', how='left'
            )
            def get_total_genes(ov):
                try: return int(str(ov).split('/')[-1])
                except: return 0
            df_detail['Genes'] = df_detail['Real_Overlap'].apply(get_total_genes)
            df_detail = df_detail.sort_values('Real_P', ascending=True)
            df_detail.to_csv(os.path.join(pathway_dir, f"Pathway_Details_{algo_name}_{gs_name}.csv"), index=False)
            
            # Save Concordance
            pd.DataFrame([{'Algorithm': algo_name, 'Library': gs_name, 'Significant_Real': len(ora_ref[ora_ref['p_value'] < 0.05])}]).to_csv(
                os.path.join(pathway_dir, f"Pathway_Concordance_{algo_name}_{gs_name}.csv"), index=False
            )

            # Statistical Validation
            common_sets = set(ora_ref['set']) & set(ora_test['set'])
            M = len(common_sets)
            if M > 0:
                stats_res = []
                rho = spearman_rank_concordance(ora_ref, ora_test)
                for K in [10, 20, 50]:
                    if K > M: continue
                    top_ref = set(ora_ref.nsmallest(K, 'p_value')['set'])
                    top_test = set(ora_test.nsmallest(K, 'p_value')['set'])
                    obs_jac = len(top_ref & top_test) / len(top_ref | top_test) if len(top_ref | top_test) > 0 else 0.0
                    null_dist = jaccard_topk_mc(M, K, B=2000)
                    p_val = perm_pvalue(null_dist, obs_jac)
                    exp_jac = expected_jaccard_random(M, K)
                    stats_res.append({'K': K, 'Observed_Jaccard': obs_jac, 'Expected_Jaccard': exp_jac, 'P_Value': p_val, 'Spearman_Rho': rho})
                
                if stats_res:
                    pd.DataFrame(stats_res).to_csv(os.path.join(pathway_dir, f"Pathway_Stats_{algo_name}_{gs_name}.csv"), index=False)

    print("✅ Redesigned Pathway analysis completed.")

if __name__ == "__main__":
    main()
