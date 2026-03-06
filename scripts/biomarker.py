import os
import argparse
import pandas as pd
from src.core.analysis import run_deg_analysis, train_eval_rf, cross_platform_evaluation

def main():
    parser = argparse.ArgumentParser(description="Biomarker and Predictive Modeling")
    parser.add_argument("--real_A", type=str, required=True, help="Path to Real Microarray data")
    parser.add_argument("--syn_A", type=str, required=True, help="Path to Synthetic Microarray data")
    parser.add_argument("--labels", type=str, required=True, help="Path to clinical labels")
    args = parser.parse_args()
    
    # 1. Load Data
    df_real = pd.read_csv(args.real_A, index_col=0)
    df_syn = pd.read_csv(args.syn_A, index_col=0)
    df_labels = pd.read_csv(args.labels, index_col=0)
    
    # Match samples
    common_idx = df_real.index.intersection(df_labels.index)
    df_real = df_real.loc[common_idx]
    df_syn = df_syn.loc[common_idx]
    labels = df_labels.loc[common_idx]['label'] # Assuming 'label' column exists
    
    # 2. DEG Analysis
    print("Running DEG Analysis...")
    deg_real = run_deg_analysis(df_real, labels)
    deg_syn = run_deg_analysis(df_syn, labels)
    
    # Save DEG overlap results
    deg_overlap = pd.DataFrame({
        'Gene': deg_real['gene'],
        'p_real': deg_real['p_value'],
        'p_syn': deg_syn['p_value']
    })
    
    os.makedirs("results/tables", exist_ok=True)
    deg_overlap.to_csv("results/tables/Table_DEGs_Overlap.csv", index=False)
    
    # 3. RF Modeling (Cross-platform)
    print("Evaluating Cross-platform Predictive Model...")
    # Train on real (subset) and evaluate on syn (remaining)
    # (In real scenario, use proper split)
    n = len(df_real) // 2
    metrics = cross_platform_evaluation(
        df_real.iloc[:n], labels.iloc[:n], 
        df_syn.iloc[n:], labels.iloc[n:]
    )
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("results/tables/Table_4_Classifier_Performance.csv", index=False)
    
    print("\nAnalysis complete. Results saved to results/tables/")
    print(metrics_df)

if __name__ == "__main__":
    main()
