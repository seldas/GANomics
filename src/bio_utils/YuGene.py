import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import warnings

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric; non-convertible values become NaN.
    Keeps index/columns unchanged.
    """
    # Fast path on pandas >= 2.0
    try:
        # to_numpy with dtype below relies on columns being convertible; we still coerce first
        coerced = df.apply(pd.to_numeric, errors="coerce")
    except Exception:
        # Fallback if something odd happens
        coerced = df.copy()
        for c in coerced.columns:
            coerced[c] = pd.to_numeric(coerced[c], errors="coerce")
    return coerced

def yugene_transform_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    YuGene: per-sample ranks rescaled to [0,1], higher expr -> higher score.
    df shape: (n_samples, n_genes)
    """
    # 1) Coerce to numeric once
    df_num = _coerce_numeric(df)

    # 2) Get a float ndarray; ensure NaNs for missing/non-finite
    try:
        X = df_num.to_numpy(dtype=float, na_value=np.nan)  # pandas >= 2.0
    except TypeError:
        # pandas < 2.0 has no na_value=; values are already float/NaN after coercion
        X = df_num.to_numpy(dtype=float)

    n_samples, n_genes = X.shape
    out = np.empty_like(X, dtype=float)

    for i in range(n_samples):
        row = X[i, :]

        # Treat inf/-inf as missing
        finite = np.isfinite(row)
        if finite.sum() <= 1:
            out[i, :] = np.nan
            continue

        r = np.full(n_genes, np.nan, dtype=float)
        r[finite] = rankdata(row[finite], method="average")

        denom = finite.sum() - 1
        if denom > 0:
            out[i, finite] = (r[finite] - 1.0) / denom
        else:
            # all finite values equal
            out[i, finite] = 0.5

        # keep NaN where not finite
        out[i, ~finite] = np.nan

    return pd.DataFrame(out, index=df.index, columns=df.columns)

def yugene_train_paired(microarray_train_df, rnaseq_train_df):
    if microarray_train_df.shape != rnaseq_train_df.shape:
        raise ValueError(
            f"Training data shapes don't match: MA {microarray_train_df.shape} vs RNA {rnaseq_train_df.shape}"
        )

    n_samples, n_genes = microarray_train_df.shape
    print(f"Training YuGene with {n_samples} samples and {n_genes} genes")

    ma_transformed = yugene_transform_single(microarray_train_df)
    rna_transformed = yugene_transform_single(rnaseq_train_df)

    # Use nan-safe reducers
    ma_vals = ma_transformed.values
    rna_vals = rna_transformed.values
    ma_stats = {
        'mean': np.nanmean(ma_vals),
        'std': np.nanstd(ma_vals),
        'median': np.nanmedian(ma_vals),
        'min': np.nanmin(ma_vals),
        'max': np.nanmax(ma_vals),
    }
    rna_stats = {
        'mean': np.nanmean(rna_vals),
        'std': np.nanstd(rna_vals),
        'median': np.nanmedian(rna_vals),
        'min': np.nanmin(rna_vals),
        'max': np.nanmax(rna_vals),
    }

    yugene_model = {
        'ma_stats': ma_stats,
        'rna_stats': rna_stats,
        'sample_names': microarray_train_df.index.tolist(),
        'gene_names': microarray_train_df.columns.tolist(),
        'n_samples': n_samples,
        'n_genes': n_genes,
        'platform_mapping': {'microarray': 'ma', 'rnaseq': 'rna'},
        'method': 'YuGene (cumulative proportion)',
    }

    print("YuGene model trained successfully")
    print(f"MA stats - mean: {ma_stats['mean']:.4f}, std: {ma_stats['std']:.4f}")
    print(f"RNA stats - mean: {rna_stats['mean']:.4f}, std: {rna_stats['std']:.4f}")

    return yugene_model

def yugene_transform_paired(test_data_df, source_platform, target_platform, yugene_model):
    """
    Transform test data using YuGene.
    
    Since YuGene is applied per sample independently, we just apply the transformation.
    The source and target platform parameters are kept for consistency with other methods.
    
    Parameters:
    -----------
    test_data_df : pd.DataFrame, shape (n_samples, n_genes)
        Test data to transform (samples x genes)
    source_platform : str
        Source platform ('microarray' or 'rnaseq') - informational only for YuGene
    target_platform : str
        Target platform ('microarray' or 'rnaseq') - informational only for YuGene
    yugene_model : dict
        Trained YuGene model (mainly metadata)
    
    Returns:
    --------
    transformed_df : pd.DataFrame, shape (n_samples, n_genes)
        YuGene transformed data
    """
    
    if source_platform == target_platform:
        print(f"Source and target platforms are the same ({source_platform}), applying YuGene transformation")
    else:
        print(f"Transforming {test_data_df.shape[0]} samples from {source_platform} to {target_platform} using YuGene")
    
    # Apply YuGene transformation
    transformed_df = yugene_transform_single(test_data_df)
    
    print(f"YuGene transformation applied to {transformed_df.shape[0]} samples")
    
    return transformed_df

def yugene_evaluate_paired(microarray_train_df, rnaseq_train_df, 
                                  microarray_test_df, rnaseq_test_df):
    """
    Complete YuGene evaluation pipeline for perfectly paired data.
    
    Parameters:
    -----------
    microarray_train_df : pd.DataFrame
        Training microarray data (samples x genes)
    rnaseq_train_df : pd.DataFrame
        Training RNA-seq data (samples x genes)
    microarray_test_df : pd.DataFrame
        Test microarray data (samples x genes)
    rnaseq_test_df : pd.DataFrame
        Test RNA-seq data (samples x genes)
    
    Returns:
    --------
    results : dict
        Dictionary containing all results and metrics
    """
    
    print("=== YuGene Cross-Platform Evaluation (Corrected) ===")
    print(f"Training data: MA {microarray_train_df.shape}, RNA {rnaseq_train_df.shape}")
    print(f"Test data: MA {microarray_test_df.shape}, RNA {rnaseq_test_df.shape}")
    
    # 1. Train YuGene model (mainly for metadata)
    print("\n1. Training YuGene model...")
    yugene_model = yugene_train_paired(microarray_train_df, rnaseq_train_df)
    
    # 2. Transform test data
    print("\n2. Transforming test data...")
    
    # Apply YuGene to both test datasets
    print("   Applying YuGene to microarray test data...")
    ma_yugene = yugene_transform_paired(microarray_test_df, 'microarray', 'yugene', yugene_model)
    
    print("   Applying YuGene to RNA-seq test data...")
    rna_yugene = yugene_transform_paired(rnaseq_test_df, 'rnaseq', 'yugene', yugene_model)
    
    # Package results
    results = {
        'method_name': 'YuGene',
        'yugene_model': yugene_model,
        'microarray_yugene': ma_yugene,
        'rnaseq_yugene': rna_yugene,
        'microarray_to_rnaseq': ma_yugene,  # For consistency with other methods
        'rnaseq_to_microarray': rna_yugene,  # For consistency with other methods
    }
    
    print("\n✓ YuGene evaluation completed!")
    
    return results

# Example usage function
def example_usage_yugene(microarray_train_df, microarray_test_df, rnaseq_train_df, rnaseq_test_df):
    """Example with your actual data using corrected YuGene"""
    
    print("YuGene Corrected Implementation:")
    print(f"Training: MA {microarray_train_df.shape}, RNA {rnaseq_train_df.shape}")
    print(f"Testing: MA {microarray_test_df.shape}, RNA {rnaseq_test_df.shape}")
    
    # Run evaluation
    results = yugene_evaluate_paired(
        microarray_train_df, rnaseq_train_df,
        microarray_test_df, rnaseq_test_df
    )
    
    return results

if __name__ == "__main__":
    # Example with dummy data
    np.random.seed(42)
    
    # Create example data that matches your dimensions
    n_samples_train, n_samples_test, n_genes = 50, 448, 10042
    
    # Sample and gene names
    train_samples = [f"train_sample_{i}" for i in range(n_samples_train)]
    test_samples = [f"test_sample_{i}" for i in range(n_samples_test)]
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Training data
    microarray_train_df = pd.DataFrame(
        np.random.lognormal(2, 1, (n_samples_train, n_genes)),
        index=train_samples, columns=genes
    )
    rnaseq_train_df = pd.DataFrame(
        np.random.lognormal(4, 1.5, (n_samples_train, n_genes)),
        index=train_samples, columns=genes
    )
    
    # Test data
    microarray_test_df = pd.DataFrame(
        np.random.lognormal(2, 1, (n_samples_test, n_genes)),
        index=test_samples, columns=genes
    )
    rnaseq_test_df = pd.DataFrame(
        np.random.lognormal(4, 1.5, (n_samples_test, n_genes)),
        index=test_samples, columns=genes
    )
    
    # Run example
    results = example_usage_yugene(microarray_train_df, microarray_test_df, 
                                          rnaseq_train_df, rnaseq_test_df)
