import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.analysis import run_deg_analysis, train_eval_rf
from src.core.pathway import load_gmt, run_permutation_test, jaccard_threshold_curve

def main():
    parser = argparse.ArgumentParser(description="Biomarker and Cross-Platform Modeling")
    parser.add_argument("--real_A", type=str, required=True, help="Path to Real Domain A")
    parser.add_argument("--real_B", type=str, required=True, help="Path to Real Domain B")
    parser.add_argument("--syn_A", type=str, required=True, help="Path to Synthetic Domain A (from B)")
    parser.add_argument("--syn_B", type=str, required=True, help="Path to Synthetic Domain B (from A)")
    parser.add_argument("--labels", type=str, default=os.path.join("plan", "SEQC_NB_249_ValidationSamples_ClinicalInfo_20121128.txt"), help="Path to clinical labels")
    parser.add_argument("--label_col", type=str, default="D_FAV_All", help="Column name for labels")
    parser.add_argument("--gmt", type=str, help="Path to MSigDB GMT file for pathway analysis")
    args = parser.parse_args()
    
    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    # 1. Load Data
    df_rA = pd.read_csv(args.real_A, index_col=0)
    df_rB = pd.read_csv(args.real_B, index_col=0)
    df_sA = pd.read_csv(args.syn_A, index_col=0)
    df_sB = pd.read_csv(args.syn_B, index_col=0)
    
    # Load labels
    if args.labels.endswith('.txt'):
        df_labels = pd.read_csv(args.labels, sep='\t')
        if 'SEQC_NB_SampleID' in df_labels.columns:
            df_labels.set_index('SEQC_NB_SampleID', inplace=True)
    else:
        df_labels = pd.read_csv(args.labels, index_col=0)

    # Determine label column
    if args.label_col in df_labels.columns:
        label_col = args.label_col
    elif "label" in df_labels.columns:
        label_col = "label"
    else:
        raise ValueError(f"Label column '{args.label_col}' or 'label' not found in {args.labels}")
    
    # Filter samples
    df_labels = df_labels.dropna(subset=[label_col])
    
    # Align samples
    idx = df_rA.index.intersection(df_labels.index)
    if len(idx) == 0:
        print(f"Warning: No matching samples found between data and labels.")
        return

    df_rA, df_rB = df_rA.loc[idx], df_rB.loc[idx]
    df_sA, df_sB = df_sA.loc[idx], df_sB.loc[idx]
    y = df_labels.loc[idx, label_col]
    
    if pd.api.types.is_numeric_dtype(y):
        y = y.map({0: 'Favorable', 1: 'Unfavorable'})
    
    print(f"Total aligned samples with labels: {len(idx)}")
    
    # Split for classification
    n = len(idx) // 2
    train_idx, test_idx = idx[:n], idx[n:]
    
    # 2. Modeling Scenarios (Table 4)
    scenarios = [
        ("Real->Real", df_rA.loc[train_idx], df_rA.loc[test_idx]),
        ("Real->Syn",  df_rA.loc[train_idx], df_sA.loc[test_idx]),
        ("Syn->Real",  df_sA.loc[train_idx], df_rA.loc[test_idx]),
        ("Syn->Syn",   df_sA.loc[train_idx], df_sA.loc[test_idx])
    ]
    
    results = []
    print("\nRunning Modeling Scenarios...")
    for name, train_x, test_x in scenarios:
        metrics = train_eval_rf(train_x, y.loc[train_idx], test_x, y.loc[test_idx])
        metrics['Scenario'] = name
        results.append(metrics)
        
    table_4 = pd.DataFrame(results)
    pred_dir = resolve_path("results/4_Biomarkers/Prediction")
    os.makedirs(pred_dir, exist_ok=True)
    table_4.to_csv(os.path.join(pred_dir, "Table_4_Classifier_Performance.csv"), index=False)
    print(table_4)

    # 3. Pathway & DEG Validation (Figure 9)
    print("\nRunning DEG and Pathway Validation...")
    deg_rA = run_deg_analysis(df_rA, y)
    deg_sA = run_deg_analysis(df_sA, y)
    
    # Jaccard overlap curve (Fig 9a)
    jac_curve = jaccard_threshold_curve(deg_rA, deg_sA)
    deg_dir = resolve_path("results/4_Biomarkers/DEG")
    os.makedirs(deg_dir, exist_ok=True)
    jac_curve.to_csv(os.path.join(deg_dir, "Table_Fig9a_Jaccard_Curve.csv"), index=False)
    
    if args.gmt:
        gene_sets = load_gmt(args.gmt)
        obs_rho, null_dist, p_val = run_permutation_test(deg_rA, deg_sA, gene_sets)
        print(f"Pathway Rank Concordance (Spearman rho): {obs_rho:.3f}, p-value: {p_val:.4f}")
        
        # Save null distribution for Fig 9b-e
        pathway_dir = resolve_path("results/4_Biomarkers/Pathway")
        os.makedirs(pathway_dir, exist_ok=True)
        pd.Series(null_dist).to_csv(os.path.join(pathway_dir, "Table_Fig9_Null_Dist.csv"), index=False)

if __name__ == "__main__":
    main()
