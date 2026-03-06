import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(real, syn, labels, title, save_path):
    """
    Generate t-SNE clustering (Fig 7 & S2).
    """
    X = np.vstack([real, syn])
    # Create combined labels: Real-A, Real-B, Syn-A, Syn-B
    # Or just use domain labels from labels list
    
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1],
        hue=labels, style=labels, palette='viridis'
    )
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_bland_altman(real, syn, genes, title, save_path):
    """
    Generate Bland-Altman bias plots (Fig 8).
    """
    # Subset to genes (e.g., PAM50)
    real_sub = real[genes].values.flatten()
    syn_sub = syn[genes].values.flatten()
    
    mean = (real_sub + syn_sub) / 2
    diff = real_sub - syn_sub
    md = np.mean(diff)
    sd = np.std(diff)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='red', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean Expression')
    plt.ylabel('Difference (Real - Synthetic)')
    plt.title(f"{title}\nBias: {md:.2f}, LoA: [{md-1.96*sd:.2f}, {md+1.96*sd:.2f}]")
    plt.savefig(save_path)
    plt.close()

def plot_performance_bars(table_path, save_path):
    """
    Generate MCC Performance Bars (Fig 10).
    """
    df = pd.read_csv(table_path)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Scenario', y='MCC', palette='muted')
    plt.ylim(0, 1.0)
    plt.title('Cross-Platform Classifier Performance (MCC)')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Manuscript Figures")
    parser.add_argument("--type", choices=['tsne', 'bland_altman', 'performance'], required=True)
    parser.add_argument("--real", type=str, help="Path to real data")
    parser.add_argument("--syn", type=str, help="Path to synthetic data")
    parser.add_argument("--table", type=str, help="Path to metrics table (for performance bars)")
    parser.add_argument("--genes", type=str, help="Path to gene signature file (for Bland-Altman)")
    parser.add_argument("--out", type=str, required=True, help="Output path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    if args.type == 'tsne':
        df_r = pd.read_csv(args.real, index_col=0)
        df_s = pd.read_csv(args.syn, index_col=0)
        # Dummy labels for demo
        labels = ['Real']*len(df_r) + ['Synthetic']*len(df_s)
        plot_tsne(df_r.values, df_s.values, labels, "t-SNE: Real vs Synthetic Alignment", args.out)
        
    elif args.type == 'bland_altman':
        df_r = pd.read_csv(args.real, index_col=0)
        df_s = pd.read_csv(args.syn, index_col=0)
        if args.genes:
            with open(args.genes, 'r') as f:
                sig_genes = [l.strip() for l in f]
            plot_bland_altman(df_r, df_s, sig_genes, "Bland-Altman: PAM50 Signature Stability", args.out)
            
    elif args.type == 'performance':
        plot_performance_bars(args.table, args.out)

if __name__ == "__main__":
    main()
