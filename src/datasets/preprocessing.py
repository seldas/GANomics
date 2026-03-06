import pandas as pd
import numpy as np
import re
import os

def normalize_genomics(df, log2_transform=True, min_max=False):
    """
    Apply standard normalization to expression data.
    """
    if log2_transform:
        # Check if already log transformed (rough heuristic)
        if df.max().max() > 100: 
            df = np.log2(df + 1.0)
            
    if min_max:
        df = (df - df.min()) / (df.max() - df.min())
        
    return df

def align_platforms(df_A, df_B, mapping_df=None):
    """
    Align two platforms by common samples and genes.
    """
    # 1. Common Samples
    common_samples = sorted(list(set(df_A.index) & set(df_B.index)))
    df_A = df_A.loc[common_samples]
    df_B = df_B.loc[common_samples]
    
    # 2. Gene Mapping
    if mapping_df is not None:
        # Use provided mapping
        # mapping_df should have columns 'A_gene' and 'B_gene'
        mapping_dict = dict(zip(mapping_df['A_gene'], mapping_df['B_gene']))
        df_A = df_A.rename(columns=mapping_dict)
    
    # 3. Common Genes
    common_genes = sorted(list(set(df_A.columns) & set(df_B.columns)))
    df_A = df_A[common_genes]
    df_B = df_B[common_genes]
    
    # 4. Final Cleanup
    df_A = df_A.dropna(axis=1)
    df_B = df_B.dropna(axis=1)
    
    # Re-align genes after dropout
    final_genes = sorted(list(set(df_A.columns) & set(df_B.columns)))
    df_A = df_A[final_genes]
    df_B = df_B[final_genes]
    
    return df_A, df_B

def preprocess_dataset(path_A, path_B, output_dir, name, log2=True):
    """
    High-level preprocessing pipeline.
    """
    print(f"Preprocessing {name}...")
    
    # Load raw (handling both CSV and TSV)
    sep_A = '\t' if path_A.endswith(('.tsv', '.txt')) else ','
    sep_B = '\t' if path_B.endswith(('.tsv', '.txt')) else ','
    
    df_A = pd.read_csv(path_A, sep=sep_A, index_col=0)
    df_B = pd.read_csv(path_B, sep=sep_B, index_col=0)
    
    # Transpose if genes are in rows (heuristic: usually more genes than samples)
    if df_A.shape[0] > df_A.shape[1]:
        df_A = df_A.T
    if df_B.shape[0] > df_B.shape[1]:
        df_B = df_B.T
        
    # Align
    df_A, df_B = align_platforms(df_A, df_B)
    
    # Normalize
    df_A = normalize_genomics(df_A, log2_transform=log2)
    df_B = normalize_genomics(df_B, log2_transform=log2)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    df_A.to_csv(os.path.join(output_dir, f"{name}_A.csv"))
    df_B.to_csv(os.path.join(output_dir, f"{name}_B.csv"))
    
    print(f"Saved processed {name} to {output_dir}")
    print(f"Shape: {df_A.shape}")
    return df_A.shape
