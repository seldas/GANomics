import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis, train_eval_rf
from src.core.pathway import load_gmt, run_permutation_test, jaccard_threshold_curve, get_enrichr_gene_sets

def main():
    parser = argparse.ArgumentParser(description="Biomarker and Cross-Platform Modeling for all algorithms")
    parser.add_argument("--run_id", type=str, required=True, help="Full run ID (e.g. NB_Ablation_Size_50_Run_0)")
    parser.add_argument("--ext_id", type=str, help="External Dataset ID (e.g. ext1)")
    parser.add_argument("--labels", type=str, help="Path to clinical labels")
    parser.add_argument("--gmt", type=str, help="Path to MSigDB GMT file for pathway analysis")
    parser.add_argument("--libraries", type=str, nargs="+", default=['KEGG_2021_Human', 'GO_Biological_Process_2021'], help="Enrichr libraries to use")
    parser.add_argument("--no_adjust_path", action='store_true', help="Don't adjust PYTHONPATH")
    args = parser.parse_args()
    
    if not args.no_adjust_path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    # 1. Paths Setup
    sync_root = resolve_path(os.path.join("results", "2_SyncData", args.run_id))
    dataset_dir = resolve_path("dataset")
    
    if args.ext_id:
        ext_dir = os.path.join(sync_root, args.ext_id)
        algo_dir = os.path.join(sync_root, f"algorithms_{args.ext_id}")
        
        # Real data from synced results (Step 2)
        real_ma_path = os.path.join(ext_dir, "microarray_real.csv")
        if not os.path.exists(real_ma_path):
            real_ma_path = os.path.join(ext_dir, "rnaseq_real.csv") # Fallback to RS if MA not available
            
        real_ma = pd.read_csv(real_ma_path, index_col=0)
        
        # Metadata from dataset folder
        project_id = args.run_id.split('_')[0]
        ext_data_dir = os.path.join(dataset_dir, project_id, args.ext_id)
        
        # Labels for external test
        label_path = args.labels if args.labels else os.path.join(ext_data_dir, "label.txt")
        if not os.path.exists(label_path):
            print(f"Error: label.txt not found for external dataset {args.ext_id}. Skipping biomarker analysis.")
            return
            
        profiles = [
            ('GANomics', os.path.join(ext_dir, "microarray_fake.csv" if "microarray" in real_ma_path else "rnaseq_fake.csv"))
        ]
        
        # Output root for ext results
        out_root = ext_dir
    else:
        test_dir = os.path.join(sync_root, "test")
        algo_dir = os.path.join(sync_root, "algorithms")
        real_ma = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
        
        label_path = resolve_path(args.labels) if args.labels else None
        if not label_path:
            # Try default project label
            project_id = args.run_id.split('_')[0]
            label_path = os.path.join(dataset_dir, project_id, "label.txt")
            
        profiles = [
            ('GANomics', os.path.join(test_dir, "microarray_fake.csv"))
        ]
        out_root = resolve_path(os.path.join("results", "4_Biomarkers"))
        
    # 2. Load Labels
    sep = '\t' if label_path.endswith('.tsv') else ','
    df_labels = pd.read_csv(label_path, index_col=0, sep=sep)
    
    # Align labels with real_ma samples
    common_idx = real_ma.index.intersection(df_labels.index)
    if len(common_idx) == 0:
        print(f"Error: No matching samples between labels ({label_path}) and data.")
        return
    
    real_ma = real_ma.loc[common_idx]
    y = df_labels.loc[common_idx, 'label']
    print(f"[{args.run_id}] Aligned {len(common_idx)} samples for biomarker analysis.")

    # 3. Identify all synthetic profiles
    if os.path.exists(algo_dir):
        for f in os.listdir(algo_dir):
            if (f.startswith("microarray_fake_") or f.startswith("rnaseq_fake_")) and f.endswith(".csv"):
                algo_name = f.replace("microarray_fake_", "").replace("rnaseq_fake_", "").replace(".csv", "").upper()
                profiles.append((algo_name, os.path.join(algo_dir, f)))

    # 4. Precompute Real DEGs (Reference)
    print("Running DEG analysis for Real Data (Reference)...")
    deg_real = run_deg_analysis(real_ma, y)

    # 5. Load Mapping Table for Pathway Analysis
    mapping_path = os.path.join(dataset_dir, project_id, "gene_mapping.tsv")
    gene_map = None
    if os.path.exists(mapping_path):
        print(f"Loading gene mapping from {mapping_path}...")
        df_map = pd.read_csv(mapping_path, sep='\t')
        # Create a dict: Probe -> Symbol
        gene_map = dict(zip(df_map['Agilent_Probe_Name'], df_map['GeneSymbol']))
    
    def apply_mapping(deg_df, mapping):
        if mapping is None: return deg_df
        df = deg_df.copy()
        # Rename index from Probe to Symbol
        df.index = [mapping.get(x, x) for x in df.index]
        # Remove entries that didn't map to a symbol (heuristic: symbols are usually not UKv4...)
        df = df[~df.index.str.startswith('UKv4_')]
        # If multiple probes map to same gene, they will now have same index name.
        # pathway.py handles this via df.index = df.index.str.upper() and df.loc[set_genes]
        # but it's better to ensure unique index here if possible, or let pathway.py handle it.
        return df

    # Map real DEGs
    deg_real_mapped = apply_mapping(deg_real, gene_map)

    # 6. Load Gene Sets (GMT and Enrichr)
    all_gene_sets = {}
    if args.gmt:
        all_gene_sets['MSigDB_Local'] = load_gmt(resolve_path(args.gmt))
    
    for lib in args.libraries:
        print(f"Fetching {lib} from Enrichr...")
        gs = get_enrichr_gene_sets(lib)
        if gs:
            all_gene_sets[lib] = gs

    # 7. Process each algorithm
    from sklearn.model_selection import train_test_split
    
    for algo_name, syn_path in profiles:
        if not os.path.exists(syn_path): continue
        print(f"\n>>> Processing Algorithm: {algo_name}")
        
        # Load synthetic data
        syn_ma = pd.read_csv(syn_path, index_col=0).loc[common_idx]
        
        # A. Prediction Model (Four Scenarios)
        # Use stratified random split to ensure balanced classes in train/test
        train_idx, test_idx = train_test_split(
            common_idx, 
            test_size=0.5, 
            random_state=42, 
            stratify=y
        )
        
        print(f"[{algo_name}] Running Cross-Platform Prediction Modeling...")
        
        all_metrics = []
        
        # 1. Real -> Real
        m_rr = train_eval_rf(real_ma.loc[train_idx], y.loc[train_idx], real_ma.loc[test_idx], y.loc[test_idx])
        m_rr['Scenario'] = 'Real->Real'
        all_metrics.append(m_rr)
        
        # 2. Real -> Syn
        m_rs = train_eval_rf(real_ma.loc[train_idx], y.loc[train_idx], syn_ma.loc[test_idx], y.loc[test_idx])
        m_rs['Scenario'] = 'Real->Syn'
        all_metrics.append(m_rs)
        
        # 3. Syn -> Real
        m_sr = train_eval_rf(syn_ma.loc[train_idx], y.loc[train_idx], real_ma.loc[test_idx], y.loc[test_idx])
        m_sr['Scenario'] = 'Syn->Real'
        all_metrics.append(m_sr)
        
        # 4. Syn -> Syn
        m_ss = train_eval_rf(syn_ma.loc[train_idx], y.loc[train_idx], syn_ma.loc[test_idx], y.loc[test_idx])
        m_ss['Scenario'] = 'Syn->Syn'
        all_metrics.append(m_ss)
        
        pred_dir = os.path.join(out_root, "Prediction", args.run_id) if not args.ext_id else os.path.join(out_root, "Prediction")
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame(all_metrics).to_csv(os.path.join(pred_dir, f"Classifier_Performance_{algo_name}.csv"), index=False)

        # B. DEG Analysis
        print(f"[{algo_name}] Running DEG Analysis...")
        deg_syn = run_deg_analysis(syn_ma, y)
        
        # Jaccard overlap curve
        jac_curve = jaccard_threshold_curve(deg_real, deg_syn)
        deg_dir = os.path.join(out_root, "DEG", args.run_id) if not args.ext_id else os.path.join(out_root, "DEG")
        os.makedirs(deg_dir, exist_ok=True)
        jac_curve.to_csv(os.path.join(deg_dir, f"Jaccard_Curve_{algo_name}.csv"), index=False)

        # C. Pathway Enrichment
        # Map synthetic DEGs
        deg_syn_mapped = apply_mapping(deg_syn, gene_map)

        for gs_name, gene_sets in all_gene_sets.items():
            print(f"[{algo_name}] Running Pathway Concordance ({gs_name})...")
            obs_rho, null_dist, p_val = run_permutation_test(deg_real_mapped, deg_syn_mapped, gene_sets, B=100)
            
            pathway_dir = os.path.join(out_root, "Pathway", args.run_id) if not args.ext_id else os.path.join(out_root, "Pathway")
            os.makedirs(pathway_dir, exist_ok=True)
            
            pathway_results = {
                'Algorithm': algo_name,
                'Library': gs_name,
                'Spearman_Rho': obs_rho,
                'P_Value': p_val
            }
            # Save with library name in filename
            pd.DataFrame([pathway_results]).to_csv(os.path.join(pathway_dir, f"Pathway_Concordance_{algo_name}_{gs_name}.csv"), index=False)
            pd.Series(null_dist).to_csv(os.path.join(pathway_dir, f"Null_Dist_{algo_name}_{gs_name}.csv"), index=False)

    print(f"\n✅ All biomarker analyses for {args.run_id} completed.")

    print(f"\n✅ All biomarker analyses for {args.run_id} completed.")

if __name__ == "__main__":
    main()
