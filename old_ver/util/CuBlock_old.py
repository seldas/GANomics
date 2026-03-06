import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import warnings

def cublock_transform_single(data_df, k_clusters=5, n_repetitions=30):
    """
    Apply CuBlock transformation to a single dataset.
    
    CuBlock uses k-means clustering to partition probes and applies
    cubic polynomial fitting to data blocks for normalization.
    
    Parameters:
    -----------
    data_df : pd.DataFrame, shape (n_samples, n_genes)
        Log2-transformed gene expression data (samples x genes)
    k_clusters : int, default=5
        Number of k-means clusters for probe partitioning
    n_repetitions : int, default=30
        Number of k-means repetitions
    
    Returns:
    --------
    transformed_df : pd.DataFrame, shape (n_samples, n_genes)
        CuBlock normalized data
    """
    
    # Convert to numpy array and transpose (genes x samples for clustering)
    data = data_df.values.T  # shape: (n_genes, n_samples)
    n_genes, n_samples = data.shape
    
    print(f"Applying CuBlock to {n_samples} samples and {n_genes} genes")
    
    # Initialize output array
    normalized_data = np.zeros_like(data)
    
    # Perform k-means clustering N times and average results
    all_normalized = np.zeros((n_repetitions, n_genes, n_samples))
    
    for rep in range(n_repetitions):
        # Perform k-means clustering on probes (genes x samples space)
        kmeans = KMeans(n_clusters=k_clusters, random_state=rep, n_init=10)
        probe_clusters = kmeans.fit_predict(data)
        
        # Initialize normalized data for this repetition
        rep_normalized = np.zeros_like(data)
        
        # Process each sample separately
        for sample_idx in range(n_samples):
            sample_data = data[:, sample_idx]
            
            # Process each cluster
            for cluster_id in range(k_clusters):
                # Get probes belonging to this cluster
                cluster_mask = probe_clusters == cluster_id
                if not np.any(cluster_mask):
                    continue
                
                # Extract block data (probe intensities for this cluster and sample)
                block_data = sample_data[cluster_mask]
                
                if len(block_data) < 2:  # Need at least 2 points for polynomial fitting
                    rep_normalized[cluster_mask, sample_idx] = block_data
                    continue
                
                # Apply CuBlock normalization to this block
                normalized_block = cublock_normalize_block(block_data)
                rep_normalized[cluster_mask, sample_idx] = normalized_block
        
        all_normalized[rep, :, :] = rep_normalized
    
    # Average across all repetitions
    normalized_data = np.mean(all_normalized, axis=0)
    
    # Convert back to DataFrame (samples x genes)
    transformed_df = pd.DataFrame(
        normalized_data.T,
        index=data_df.index,
        columns=data_df.columns
    )
    
    return transformed_df

def cublock_normalize_block(block_data):
    """
    Normalize a single data block using CuBlock cubic polynomial fitting.
    
    Parameters:
    -----------
    block_data : np.array
        Block of gene expression values
    
    Returns:
    --------
    normalized_block : np.array
        Normalized block data
    """
    
    if len(block_data) <= 1:
        return block_data
    
    # Step 1: Transform to z-scores (zero mean, unit variance)
    if np.std(block_data) == 0:
        return block_data  # All values are the same
    
    z_scores = (block_data - np.mean(block_data)) / np.std(block_data)
    
    # Step 2: Sort the z-scores
    sorted_indices = np.argsort(z_scores)
    sorted_z_scores = z_scores[sorted_indices]
    
    # Step 3: Create mapping function output values
    mapped_values = cublock_mapping_function(sorted_z_scores)
    
    # Step 4: Fit cubic polynomial
    if len(sorted_z_scores) >= 4:  # Need at least 4 points for cubic fit
        try:
            # Fit cubic polynomial: y = ax³ + bx² + cx + d
            coefficients = np.polyfit(sorted_z_scores, mapped_values, 3)
            
            # Evaluate polynomial on all z-scores
            polynomial_values = np.polyval(coefficients, z_scores)
            
            # Step 5: Handle decreasing values (monotonicity correction)
            polynomial_values = correct_decreasing_values(polynomial_values, z_scores)
            
            return polynomial_values
            
        except np.linalg.LinAlgError:
            # Fall back to linear scaling if polynomial fitting fails
            return cublock_linear_fallback(block_data)
    else:
        # Not enough points for cubic fit, use linear scaling
        return cublock_linear_fallback(block_data)

def cublock_mapping_function(sorted_values):
    """
    Create the mapping function as described in CuBlock algorithm.
    Maps sorted values to points between -1 and 1 with density increasing toward zero.
    
    Parameters:
    -----------
    sorted_values : np.array
        Sorted z-score values
    
    Returns:
    --------
    mapped_values : np.array
        Mapped output values for polynomial fitting
    """
    
    n = len(sorted_values)
    
    # Create equidistant points between -1 and 1
    equidistant_points = np.linspace(-1, 1, n)
    
    # Find appropriate uneven power (between 3 and 21)
    # The goal is to have values within standard deviation mapped to < 0.1
    best_power = 3
    
    for power in range(3, 22, 2):  # Try odd powers from 3 to 21
        # Apply power transformation
        powered_points = np.sign(equidistant_points) * np.abs(equidistant_points) ** power
        
        # Check if values within standard deviation are mapped to < 0.1
        within_std_mask = (sorted_values >= -1) & (sorted_values <= 1)
        if np.any(within_std_mask):
            within_std_values = powered_points[within_std_mask]
            if np.mean(np.abs(within_std_values)) < 0.1:
                best_power = power
                break
    
    # Apply the selected power transformation
    mapped_values = np.sign(equidistant_points) * np.abs(equidistant_points) ** best_power
    
    return mapped_values

def correct_decreasing_values(polynomial_values, z_scores):
    """
    Correct decreasing values in polynomial to preserve monotonicity.
    
    Parameters:
    -----------
    polynomial_values : np.array
        Values from polynomial evaluation
    z_scores : np.array
        Original z-score values
    
    Returns:
    --------
    corrected_values : np.array
        Monotonicity-corrected values
    """
    
    # Sort by z-scores to check monotonicity
    sorted_indices = np.argsort(z_scores)
    sorted_poly_values = polynomial_values[sorted_indices]
    
    # Check for decreasing segments
    corrected_sorted = sorted_poly_values.copy()
    
    for i in range(1, len(corrected_sorted)):
        if corrected_sorted[i] < corrected_sorted[i-1]:
            # Found decreasing value, correct it
            corrected_sorted[i] = corrected_sorted[i-1]
    
    # Map back to original order
    corrected_values = np.zeros_like(polynomial_values)
    corrected_values[sorted_indices] = corrected_sorted
    
    return corrected_values

def cublock_linear_fallback(block_data):
    """
    Linear fallback normalization when polynomial fitting is not possible.
    
    Parameters:
    -----------
    block_data : np.array
        Block of gene expression values
    
    Returns:
    --------
    normalized_block : np.array
        Linearly normalized block data
    """
    
    if np.std(block_data) == 0:
        return np.zeros_like(block_data)
    
    # Simple z-score normalization
    return (block_data - np.mean(block_data)) / np.std(block_data)

def cublock_train_paired(microarray_train_df, rnaseq_train_df, k_clusters=5, n_repetitions=1):
    """
    Train CuBlock cross-platform mappings.

    - Learn shared gene clusters on combined training data (genes x samples).
    - For each cluster (block) and each platform (MA/RNA), build two monotone
      spline mappings using pooled training values in that block:
        * f_p2ref : platform -> shared reference (quantile-averaged across platforms)
        * f_ref2p : shared reference -> platform
    - At inference, we map source -> ref -> target per block.

    Notes:
    - Uses PCHIP (monotone cubic) on percentile grids for robustness.
    - k-means run once (n_repetitions kept for signature compatibility).
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from scipy.interpolate import PchipInterpolator

    # ---------- Prep data ----------
    assert microarray_train_df.shape == rnaseq_train_df.shape, \
        f"Training shapes must match, got MA {microarray_train_df.shape} vs RNA {rnaseq_train_df.shape}"

    # Data shape: (n_samples, n_genes), convert to (n_genes, n_samples) for clustering
    ma = microarray_train_df.values.T
    rna = rnaseq_train_df.values.T
    n_genes, n_samples = ma.shape

    # ---------- Shared clustering on genes ----------
    # Cluster gene profiles using BOTH platforms (concatenate along samples)
    gene_features = np.concatenate([ma, rna], axis=1)  # (n_genes, 2*n_samples)
    km = KMeans(n_clusters=k_clusters, n_init=20, random_state=42)
    clusters = km.fit_predict(gene_features)  # length = n_genes

    # ---------- Helper: percentile grid + PCHIP fit ----------
    def _percentile_grid(x, q=101):
        # robust empirical quantiles; x is 1D
        qs = np.linspace(0, 100, q)
        return np.percentile(x, qs), qs

    def _fit_monotone_spline(x_vals, y_vals):
        # x_vals must be sorted ascending; ensure monotone by sorting+dedup
        x_sorted = np.array(x_vals, dtype=float)
        y_sorted = np.array(y_vals, dtype=float)
        order = np.argsort(x_sorted)
        x_sorted = x_sorted[order]
        y_sorted = y_sorted[order]
        # de-duplicate x for spline stability
        mask = np.diff(x_sorted, prepend=x_sorted[0]-1e-12) > 0
        x_unique = x_sorted[mask]
        y_unique = y_sorted[mask]
        if len(x_unique) < 2:
            # fallback: identity
            x_unique = np.array([-3.0, 3.0])
            y_unique = np.array([-3.0, 3.0])
        return PchipInterpolator(x_unique, y_unique, extrapolate=True)

    # ---------- Build per-block mappings ----------
    block_maps = []  # list of dicts per block: {'genes': idx, 'ma2ref', 'ref2ma', 'rna2ref', 'ref2rna'}
    percentile_count = 101  # 0..100

    for b in range(k_clusters):
        gene_idx = np.where(clusters == b)[0]
        if len(gene_idx) == 0:
            # empty block (rare), create identity maps
            empty_map = dict(
                genes=np.array([], dtype=int),
                ma2ref=_fit_monotone_spline([-3, 3], [-3, 3]),
                ref2ma=_fit_monotone_spline([-3, 3], [-3, 3]),
                rna2ref=_fit_monotone_spline([-3, 3], [-3, 3]),
                ref2rna=_fit_monotone_spline([-3, 3], [-3, 3]),
            )
            block_maps.append(empty_map)
            continue

        # Pool raw values across all samples and genes IN THIS BLOCK, per platform
        # (You can switch to z-scores per-gene if you prefer; pooling raw log2 is common in CuBlock)
        ma_block_vals = ma[gene_idx, :].ravel()
        rna_block_vals = rna[gene_idx, :].ravel()

        # Compute platform quantile curves
        ma_q, _ = _percentile_grid(ma_block_vals, q=percentile_count)
        rna_q, _ = _percentile_grid(rna_block_vals, q=percentile_count)

        # Define shared reference as the average of platform quantiles (balanced reference)
        ref_q = 0.5 * (ma_q + rna_q)

        # Fit monotone splines:
        # platform -> ref  (x=platform quantiles, y=ref quantiles)
        ma2ref = _fit_monotone_spline(ma_q, ref_q)
        rna2ref = _fit_monotone_spline(rna_q, ref_q)

        # ref -> platform (x=ref quantiles, y=platform quantiles)
        ref2ma = _fit_monotone_spline(ref_q, ma_q)
        ref2rna = _fit_monotone_spline(ref_q, rna_q)

        block_maps.append(dict(
            genes=gene_idx,
            ma2ref=ma2ref, ref2ma=ref2ma,
            rna2ref=rna2ref, ref2rna=ref2rna
        ))

    # Store model
    cublock_model = {
        'k_clusters': k_clusters,
        'n_repetitions': n_repetitions,
        'clusters': clusters,                 # (n_genes,)
        'block_maps': block_maps,             # list of mapping dicts
        'sample_names': microarray_train_df.index.tolist(),
        'gene_names': microarray_train_df.columns.tolist(),
        'n_samples': microarray_train_df.shape[0],
        'n_genes': microarray_train_df.shape[1],
        'platform_mapping': {'microarray': 'ma', 'rnaseq': 'rna'},
        'method': 'CuBlock (shared blocks; monotone quantile-spline cross-map)',
    }

    # Some quick stats for logging (optional)
    ma_stats = {'mean': float(np.mean(ma)), 'std': float(np.std(ma))}
    rna_stats = {'mean': float(np.mean(rna)), 'std': float(np.std(rna))}
    cublock_model['ma_stats'] = ma_stats
    cublock_model['rna_stats'] = rna_stats

    print(f"[CuBlock] Trained with {k_clusters} blocks on {len(cublock_model['gene_names'])} genes.")
    return cublock_model

def cublock_transform_paired(test_data_df, source_platform, target_platform, cublock_model):
    """
    Transform test data using learned shared-block CuBlock mappings.

    - If target_platform is 'microarray' or 'rnaseq' and different from source,
      perform true cross-platform mapping: source -> ref -> target (per block).
    - If target_platform is 'cublock' (legacy), just normalize to the shared
      reference (source -> ref) and return in the original scale of 'ref'.

    Parameters
    ----------
    test_data_df : (n_samples, n_genes) DataFrame
    source_platform : 'microarray' or 'rnaseq'
    target_platform : 'microarray', 'rnaseq', or 'cublock'
    cublock_model : dict from cublock_train_paired
    """
    import numpy as np
    import pandas as pd

    X = test_data_df.values  # (n_samples, n_genes)
    n_samples, n_genes = X.shape
    assert n_genes == len(cublock_model['gene_names']), "Gene dimension mismatch with model."

    clusters = cublock_model['clusters']
    block_maps = cublock_model['block_maps']

    # Select mapping accessors based on platform labels
    def _get_maps_for_block(b):
        m = block_maps[b]
        return m['ma2ref'], m['ref2ma'], m['rna2ref'], m['ref2rna']

    def _map_array(arr, f):
        # Evaluate monotone spline with clipping at ends handled by PCHIP's extrapolate
        return f(arr)

    # Determine mapping functions to apply
    src = {'microarray': 'ma', 'rnaseq': 'rna'}.get(source_platform, None)
    tgt = {'microarray': 'ma', 'rnaseq': 'rna', 'cublock': 'ref'}.get(target_platform, None)
    if src is None or tgt is None:
        raise ValueError(f"Unknown platform(s): source={source_platform}, target={target_platform}")

    Y = np.empty_like(X, dtype=float)

    # Process each block separately for every sample
    for b in range(cublock_model['k_clusters']):
        gene_idx = block_maps[b]['genes']
        if gene_idx.size == 0:
            continue

        ma2ref, ref2ma, rna2ref, ref2rna = _get_maps_for_block(b)

        # pick forward mapping from source to ref
        if src == 'ma':
            f_src2ref = ma2ref
        else:
            f_src2ref = rna2ref

        # pick mapping from ref to target
        if tgt == 'ma':
            f_ref2tgt = ref2ma
        elif tgt == 'rna':
            f_ref2tgt = ref2rna
        else:  # 'ref' normalization only
            f_ref2tgt = None

        block_vals = X[:, gene_idx]  # (n_samples, n_block_genes)

        # source -> ref
        block_ref = _map_array(block_vals, f_src2ref)

        # ref -> target (optional)
        if f_ref2tgt is not None:
            block_out = _map_array(block_ref, f_ref2tgt)
        else:
            block_out = block_ref  # normalized to shared reference only

        Y[:, gene_idx] = block_out

    transformed_df = pd.DataFrame(Y, index=test_data_df.index, columns=test_data_df.columns)

    # Friendly logs
    action = f"{source_platform}→{target_platform}" if target_platform in ('microarray', 'rnaseq') else f"{source_platform}→ref"
    print(f"[CuBlock] Transformed {n_samples} samples ({action}) using shared blocks.")
    return transformed_df


def cublock_evaluate_paired(microarray_train_df, rnaseq_train_df, 
                           microarray_test_df, rnaseq_test_df,
                           k_clusters=5, n_repetitions=30):
    """
    Complete CuBlock evaluation pipeline for perfectly paired data.
    
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
    k_clusters : int, default=5
        Number of k-means clusters
    n_repetitions : int, default=30
        Number of k-means repetitions
    
    Returns:
    --------
    results : dict
        Dictionary containing all results and metrics
    """
    
    print("=== CuBlock Cross-Platform Evaluation ===")
    print(f"Training data: MA {microarray_train_df.shape}, RNA {rnaseq_train_df.shape}")
    print(f"Test data: MA {microarray_test_df.shape}, RNA {rnaseq_test_df.shape}")
    print(f"Parameters: k_clusters={k_clusters}, n_repetitions={n_repetitions}")
    
    # 1. Train CuBlock model
    print("\n1. Training CuBlock model...")
    cublock_model = cublock_train_paired(microarray_train_df, rnaseq_train_df, 
                                        k_clusters, n_repetitions)
    
    # 2. Transform test data
    print("\n2. Transforming test data...")
    
    # Apply CuBlock to both test datasets
    print("   Applying CuBlock to microarray test data...")
    ma_cublock = cublock_transform_paired(microarray_test_df, 'microarray', 'rnaseq', cublock_model)
    
    print("   Applying CuBlock to RNA-seq test data...")
    rna_cublock = cublock_transform_paired(rnaseq_test_df, 'rnaseq', 'microarray', cublock_model)
        
    # Package results
    results = {
        'method_name': 'CuBlock',
        'cublock_model': cublock_model,
        'microarray_cublock': ma_cublock,
        'rnaseq_cublock': rna_cublock,
        'microarray_to_rnaseq': ma_cublock,  # For consistency with other methods
        'rnaseq_to_microarray': rna_cublock,  # For consistency with other methods
    }
    
    return results

# Example usage function
def example_usage_cublock(microarray_train_df, microarray_test_df, rnaseq_train_df, rnaseq_test_df):
    """Example with your actual data using CuBlock"""
    
    print("CuBlock Implementation:")
    print(f"Training: MA {microarray_train_df.shape}, RNA {rnaseq_train_df.shape}")
    print(f"Testing: MA {microarray_test_df.shape}, RNA {rnaseq_test_df.shape}")
    
    # Run evaluation
    results = cublock_evaluate_paired(
        microarray_train_df, rnaseq_train_df,
        microarray_test_df, rnaseq_test_df,
        k_clusters=5, n_repetitions=30
    )
    
    return results
