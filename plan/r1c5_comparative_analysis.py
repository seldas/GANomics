import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def load_jaccard(method_path):
    df = pd.read_csv(os.path.join(method_path, "Table_Fig9a_Jaccard_Curve.csv"))
    return df

def plot_comparison():
    base_dir = "results/tables/biomarkers"
    methods = {
        "GANomics": os.path.join(base_dir, "GANomics_baseline"),
        "ComBat": os.path.join(base_dir, "ComBat"),
        "TDM": os.path.join(base_dir, "TDM"),
        "QN": os.path.join(base_dir, "QN"),
        "YuGene": os.path.join(base_dir, "YuGene")
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, path in methods.items():
        if os.path.exists(path):
            df = load_jaccard(path)
            plt.plot(df['threshold'], df['jaccard'], marker='o', label=name)
    
    plt.xscale('log')
    plt.xlabel('FDR Threshold')
    plt.ylabel('Jaccard Overlap')
    plt.title('DEG Overlap: GANomics vs Baselines (Figure 9a Baseline)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/r1c5_jaccard_comparison.png")
    print("Saved Jaccard comparison plot to results/figures/r1c5_jaccard_comparison.png")

def compare_classifier():
    base_dir = "results/tables/biomarkers"
    methods = ["GANomics_baseline", "ComBat", "TDM", "QN", "YuGene"]
    
    all_results = []
    for method in methods:
        path = os.path.join(base_dir, method, "Table_4_Classifier_Performance.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Method'] = method
            all_results.append(df)
            
    if all_results:
        summary = pd.concat(all_results)
        # Focus on Cross-Platform scenarios
        cp_summary = summary[summary['Scenario'].isin(['Real->Syn', 'Syn->Real'])]
        cp_pivot = cp_summary.pivot(index='Method', columns='Scenario', values='MCC')
        print("\nCross-Platform MCC Comparison:")
        print(cp_pivot)
        cp_pivot.to_csv("results/tables/r1c5_mcc_comparison.csv")

if __name__ == "__main__":
    plot_comparison()
    compare_classifier()
