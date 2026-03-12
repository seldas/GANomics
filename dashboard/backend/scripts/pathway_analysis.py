import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis
from src.core.pathway import (
    get_enrichr_gene_sets, load_gmt, ora_enrichment, 
    gene_set_enrichment, run_permutation_test
)

def main():
    parser = argparse.ArgumentParser(description="Pathway Concordance Analysis")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--ext_id", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--libraries", type=str, nargs="+", default=['KEGG_2021_Human', 'GO_Biological_Process_2021'])
    parser.add_argument("--no_filter", action='store_true', help="Disable size filtering (15-500)")
    args = parser.parse_args()

    # Paths Setup
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sync_root = os.path.join(backend_dir, "results", "2_SyncData", args.run_id)
    project_id = args.run_id.split('_')[0]
    apply_filter = not args.no_filter
    
    if args.ext_id:
        test_dir = os.path.join(sync_root, args.ext_id)
        algo_dir = os.path.join(sync_root, f"algorithms_{args.ext_id}")
        out_root = test_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        algo_dir = os.path.join(sync_root, "algorithms")
        out_root = os.path.join(backend_dir, "results", "4_Biomarkers")

    real_ma_path = os.path.join(test_dir, "microarray_real.csv")
    if not os.path.exists(real_ma_path): real_ma_path = os.path.join(test_dir, "rnaseq_real.csv")
    if not os.path.exists(real_ma_path): return

    # Labels & Background
    label_path = args.labels if args.labels else os.path.join(backend_dir, "dataset", project_id, "label.txt")
    real_ma = pd.read_csv(real_ma_path, index_col=0)
    df_labels = pd.read_csv(label_path, index_col=0, sep=None, engine='python')
    common_idx = real_ma.index.intersection(df_labels.index)
    real_ma = real_ma.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']

    # Mapping
    mapping_path = os.path.join(backend_dir, "dataset", project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, sep='\t')
        gene_map = dict(zip(df_map['Agilent_Probe_Name'], df_map['GeneSymbol']))
    
    def apply_mapping(deg_df, mapping):
        if mapping is None: return deg_df
        df = deg_df.copy()
        df.index = [mapping.get(x, x) for x in df.index]
        df = df[~df.index.astype(str).str.startswith('UKv4_')]
        return df

    # Libraries
    all_gene_sets = {}
    for lib in args.libraries:
        gs = get_enrichr_gene_sets(lib)
        if gs: all_gene_sets[lib] = gs

    # Profiles
    profiles = [('GANomics', os.path.join(test_dir, "microarray_fake.csv" if "microarray" in real_ma_path else "rnaseq_fake.csv"))]
    if os.path.exists(algo_dir):
        for f in os.listdir(algo_dir):
            if (f.startswith("microarray_fake_") or f.startswith("rnaseq_fake_")) and f.endswith(".csv"):
                algo_name = f.replace("microarray_fake_", "").replace("rnaseq_fake_", "").replace(".csv", "").upper()
                profiles.append((algo_name, os.path.join(algo_dir, f)))

    print("Precomputing Reference Pathways...")
    deg_real = apply_mapping(run_deg_analysis(real_ma, y), gene_map)

    pathway_dir = os.path.join(out_root, "Pathway", args.run_id) if not args.ext_id else os.path.join(out_root, "Pathway")
    os.makedirs(pathway_dir, exist_ok=True)

    for algo_name, syn_path in profiles:
        if not os.path.exists(syn_path): continue
        print(f"\n>>> Processing Pathways: {algo_name}")
        syn_ma = pd.read_csv(syn_path, index_col=0).loc[common_idx]
        deg_syn = apply_mapping(run_deg_analysis(syn_ma, y), gene_map)

        for gs_name, gene_sets in all_gene_sets.items():
            enr_real = gene_set_enrichment(deg_real, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            enr_syn = gene_set_enrichment(deg_syn, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            ora_real = ora_enrichment(deg_real, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            ora_syn = ora_enrichment(deg_syn, gene_sets, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            
            obs_rho, null_dist, p_val = run_permutation_test(deg_real, deg_syn, gene_sets, B=100, min_size=15 if apply_filter else 0, max_size=500 if apply_filter else 10000)
            
            pd.DataFrame([{'Algorithm': algo_name, 'Library': gs_name, 'Spearman_Rho': obs_rho, 'P_Value': p_val}]).to_csv(os.path.join(pathway_dir, f"Pathway_Concordance_{algo_name}_{gs_name}.csv"), index=False)

            if not enr_real.empty and not enr_syn.empty:
                df_detail = pd.merge(
                    enr_real[['set', 'score', 'rank', 'count']].rename(columns={'score': 'Real_Score', 'rank': 'Real_Rank', 'count': 'Real_Count'}),
                    enr_syn[['set', 'score', 'rank', 'count']].rename(columns={'score': 'Syn_Score', 'rank': 'Syn_Rank', 'count': 'Syn_Count'}),
                    on='set', how='inner'
                )
                if not ora_real.empty:
                    df_detail = pd.merge(df_detail, ora_real[['set', 'p_value', 'fdr', 'overlap', 'genes']].rename(columns={'p_value': 'Real_P', 'fdr': 'Real_FDR', 'overlap': 'Real_Overlap', 'genes': 'Real_Genes'}), on='set', how='left')
                if not ora_syn.empty:
                    df_detail = pd.merge(df_detail, ora_syn[['set', 'p_value', 'fdr', 'overlap', 'genes']].rename(columns={'p_value': 'Syn_P', 'fdr': 'Syn_FDR', 'overlap': 'Syn_Overlap', 'genes': 'Syn_Genes'}), on='set', how='left')
                
                df_detail.sort_values('Real_Rank').to_csv(os.path.join(pathway_dir, f"Pathway_Details_{algo_name}_{gs_name}.csv"), index=False)

    print("✅ Pathway analysis completed.")

if __name__ == "__main__":
    main()
