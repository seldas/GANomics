import pandas as pd
import numpy as np
import re
import os

def load_raw_data(path):
    """
    Load data from CSV, TSV, or Excel.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path, index_col=0)
    elif path.endswith(('.tsv', '.txt')):
        return pd.read_csv(path, sep='\t', index_col=0)
    elif path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(path, index_col=0)
    else:
        raise ValueError(f"Unsupported format: {path}")

def map_probes_to_symbols(df, annotation_df, probe_col, symbol_col):
    """
    Map probe/ID columns to Gene Symbols.
    """
    mapping = dict(zip(annotation_df[probe_col], annotation_df[symbol_col]))
    df = df.copy()
    df.index = [mapping.get(x, x) for x in df.index]
    return df

def handle_duplicates(df, method='mean'):
    """
    Handle multiple probes mapping to the same gene.
    """
    if method == 'mean':
        return df.groupby(df.index).mean()
    elif method == 'max_var':
        return df.loc[df.var(axis=1).groupby(df.index).idxmax()]
    else:
        return df.groupby(df.index).first()

def normalize_genomics(df, log2_transform=True, min_max=False):
    """
    Apply standard normalization.
    """
    if log2_transform:
        # Avoid log(0)
        if df.max().max() > 100:
            df = np.log2(df + 1.0)
    if min_max:
        df = (df - df.min()) / (df.max() - df.min())
    return df

def align_platforms(df_A, df_B, force_index_mapping=True):
    """
    Align by common samples and common genes.
    """
    if force_index_mapping:
        if len(df_A) != len(df_B):
            raise ValueError(f"Forced index mapping requires same number of genes. "
                             f"A has {len(df_A)}, B has {len(df_B)}")
        # Assume same order, rename B to A for consistency
        df_B.index = df_A.index
    else:
        # Common Genes (columns if transposed, but usually genes are indices initially)
        common_genes = sorted(list(set(df_A.index) & set(df_B.index)))
        if len(common_genes) == 0:
            raise ValueError("No common genes found. If they are already aligned by index, set force_index_mapping=True.")
        df_A = df_A.loc[common_genes]
        df_B = df_B.loc[common_genes]
    
    # Transpose for GANomics (Samples x Genes)
    df_A, df_B = df_A.T, df_B.T
    
    # Common Samples
    common_samples = sorted(list(set(df_A.index) & set(df_B.index)))
    df_A = df_A.loc[common_samples]
    df_B = df_B.loc[common_samples]
    
    return df_A, df_B

def full_preprocess_pipeline(path_A, path_B, output_dir, name, config=None):
    """
    Complete pipeline starting from raw GEO/Synapse-like files.
    """
    print(f"--- Processing {name} ---")
    df_A = load_raw_data(path_A)
    df_B = load_raw_data(path_B)
    
    # Optional mapping if config provided
    if config and 'annotation_A' in config:
        ann_A = load_raw_data(config['annotation_A'])
        df_A = map_probes_to_symbols(df_A, ann_A, config['probe_col_A'], config['symbol_col_A'])
    
    if config and 'annotation_B' in config:
        ann_B = load_raw_data(config['annotation_B'])
        df_B = map_probes_to_symbols(df_B, ann_B, config['probe_col_B'], config['symbol_col_B'])

    # Handle duplicates (many probes -> one gene)
    df_A = handle_duplicates(df_A)
    df_B = handle_duplicates(df_B)
    
    # Align
    force_index_mapping = config.get('force_index_mapping', True) if config else True
    df_A, df_B = align_platforms(df_A, df_B, force_index_mapping=force_index_mapping)
    
    # Normalize
    df_A = normalize_genomics(df_A)
    df_B = normalize_genomics(df_B)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    df_A.to_csv(os.path.join(output_dir, f"{name}_A.csv"))
    df_B.to_csv(os.path.join(output_dir, f"{name}_B.csv"))
    
    return df_A.shape
