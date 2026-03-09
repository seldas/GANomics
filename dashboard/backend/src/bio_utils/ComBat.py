import numpy as np
import pandas as pd
from scipy import stats
import warnings

def combat_train_paired(microarray_train_df, rnaseq_train_df, parametric=True, mean_only=False):
    """
    Train ComBat model for cross-platform normalization using perfectly paired data.
    
    Parameters:
    -----------
    microarray_train_df : pd.DataFrame, shape (n_samples, n_genes)
        Training microarray data (samples x genes)
    rnaseq_train_df : pd.DataFrame, shape (n_samples, n_genes)
        Training RNA-seq data (samples x genes) - same samples and genes as microarray
    parametric : bool, default=True
        Whether to use parametric adjustments
    mean_only : bool, default=False
        Whether to only adjust mean (not variance)
    
    Returns:
    --------
    combat_model : dict
        Trained ComBat model parameters
    """
    
    print(f"Training ComBat with {microarray_train_df.shape[0]} samples and {microarray_train_df.shape[1]} genes")
    
    # Convert to numpy arrays and transpose to genes x samples
    ma_data = microarray_train_df.values.T  # shape: (n_genes, n_samples)
    rna_data = rnaseq_train_df.values.T     # shape: (n_genes, n_samples)
    
    # Combine data from both platforms
    combined_data = np.hstack([ma_data, rna_data])  # shape: (n_genes, 2*n_samples)
    
    # Create batch labels (0 for microarray, 1 for RNA-seq)
    n_samples = microarray_train_df.shape[0]
    batch_labels = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    # Train ComBat model
    combat_model = combat_train(combined_data, batch_labels, parametric=parametric, mean_only=mean_only)
    
    # Add metadata
    combat_model['platform_mapping'] = {'microarray':0, 'rnaseq':1}
    combat_model['sample_names'] = microarray_train_df.index.tolist()
    combat_model['gene_names'] = microarray_train_df.columns.tolist()
    combat_model['n_samples'] = n_samples
    combat_model['n_genes'] = microarray_train_df.shape[1]
    
    return combat_model

def combat_transform_paired(test_data_df, source_platform, target_platform, combat_model):
    """
    Transform test data from source platform to target platform.
    
    Parameters:
    -----------
    test_data_df : pd.DataFrame, shape (n_samples, n_genes)
        Test data to transform (samples x genes)
    source_platform : str
        Source platform ('microarray' or 'rnaseq')
    target_platform : str
        Target platform ('microarray' or 'rnaseq')
    combat_model : dict
        Trained ComBat model
    
    Returns:
    --------
    transformed_df : pd.DataFrame, shape (n_samples, n_genes)
        Transformed data (samples x genes)
    """
    
    if source_platform == target_platform:
        print(f"Source and target platforms are the same ({source_platform}), returning original data")
        return test_data_df.copy()
    
    # Get platform indices
    platform_mapping = combat_model['platform_mapping']
    try:
        source_idx = platform_mapping[source_platform]
        target_idx = platform_mapping[target_platform]
    except KeyError as e:
        raise ValueError(f"Unknown platform label {e.args[0]!r}. Known: {list(platform_mapping.keys())}")
    
    # --- enforce gene alignment and dtype ---
    train_genes = combat_model.get('gene_names', None)
    if train_genes is None:
        # fall back: assume current columns are already in the training order
        aligned = test_data_df.copy()
    else:
        # ensure all required genes exist and in the same order
        missing = [g for g in train_genes if g not in test_data_df.columns]
        if missing:
            raise ValueError(f"Test data missing {len(missing)} genes from training: e.g., {missing[:5]} ...")
        aligned = test_data_df.loc[:, train_genes]

    X = aligned.astype(float).to_numpy().T  # genes x samples
    n_genes, n_samples = X.shape
    print(f"Transforming {n_samples} samples from {source_platform} to {target_platform}")

    # --- remove source batch (to common space around alpha) ---
    batch_labels = np.full(n_samples, source_idx, dtype=int)
    corrected_data = combat_test(X, batch_labels, combat_model)  # genes x samples

    # --- synthesize target batch ---
    beta_target   = np.asarray(combat_model['beta_hat'][:, target_idx])    # alpha_j
    gamma_target  = np.asarray(combat_model['gamma_hat'][:, target_idx])   # mean shift in raw space
    delta_target  = np.asarray(combat_model['delta_hat'][:, target_idx])   # variance (std^2) factor on stdzd scale
    mean_only     = bool(combat_model.get('mean_only', False))

    if mean_only:
        # y_t = corrected + gamma_target
        transformed_data = corrected_data + gamma_target[:, None]
    else:
        # y_t = ((corrected - alpha) * sqrt(delta_target)) + alpha + gamma_target
        sqrt_delta = np.sqrt(np.maximum(delta_target, 1e-8))
        centered   = corrected_data - beta_target[:, None]
        transformed_data = centered * sqrt_delta[:, None] + beta_target[:, None] + gamma_target[:, None]

    # back to samples x genes, keep original row index and training gene order
    transformed_df = pd.DataFrame(
        transformed_data.T,
        index=test_data_df.index,
        columns=(train_genes if train_genes is not None else test_data_df.columns)
    )
    
    return transformed_df

def combat_evaluate_paired(microarray_train_df, rnaseq_train_df, 
                          microarray_test_df, rnaseq_test_df,
                          parametric=True, mean_only=False):
    """
    Complete ComBat evaluation pipeline for perfectly paired data.
    
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
    parametric : bool, default=True
        Whether to use parametric adjustments
    mean_only : bool, default=False
        Whether to only adjust mean (not variance)
    
    Returns:
    --------
    results : dict
        Dictionary containing all results and metrics
    """
    
    print("=== ComBat Cross-Platform Evaluation ===")
    
    # 1. Train ComBat model
    print("\n1. Training ComBat model...")
    combat_model = combat_train_paired(microarray_train_df, rnaseq_train_df, 
                                      parametric=parametric, mean_only=mean_only)
    
    # 2. Transform test data
    print("\n2. Transforming test data...")
    
    # Microarray -> RNA-seq
    print("   Microarray -> RNA-seq...")
    ma_to_rnaseq = combat_transform_paired(microarray_test_df, 'microarray', 'rnaseq', combat_model)
    
    # RNA-seq -> Microarray
    print("   RNA-seq -> Microarray...")
    rnaseq_to_ma = combat_transform_paired(rnaseq_test_df, 'rnaseq', 'microarray', combat_model)
    
    # Package results
    results = {
        'method_name': 'ComBat',
        'combat_model': combat_model,
        'microarray_to_rnaseq': ma_to_rnaseq,
        'rnaseq_to_microarray': rnaseq_to_ma,
    }
    
    return results

# Keep the original core functions unchanged
def combat_train(data, batch, covariates=None, parametric=True, mean_only=False):
    """
    Parametric EB ComBat training.
    Inputs
    ------
    data : array-like, shape (n_genes, n_samples)
    batch : array-like, shape (n_samples,)
    covariates : None or array-like (n_samples, p) -- *not* including batch dummies
    parametric : bool (only parametric implemented here)
    mean_only : bool (if True, only shift means; no variance adjustment)

    Returns
    -------
    combat_model : dict
        Contains posterior batch effect means/variances and pooled location/scale
        arranged so your existing wrappers work as intended.
    """
    import numpy as np

    data = np.array(data, dtype=float)
    batch = np.asarray(batch)
    n_features, n_samples = data.shape

    # Batches & sizes
    batches = np.unique(batch)
    n_batches = len(batches)
    nb = np.array([np.sum(batch == b) for b in batches])

    # Design for covariates (intercept-only if None)
    if covariates is not None:
        X = np.asarray(covariates)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # prepend intercept
        X = np.column_stack([np.ones(n_samples), X])
    else:
        X = np.ones((n_samples, 1))

    # Fit per-gene covariate model y = X beta + eps  (no batch dummies here)
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = (XtX_inv @ X.T @ data.T).T  # (n_genes, p)
    fitted = (X @ betas.T).T            # (n_genes, n_samples)
    resid = data - fitted               # remove covariates only

    # Raw batch means (on covariate-residual space)
    gamma_hat_raw = np.zeros((n_features, n_batches))
    for bi, b in enumerate(batches):
        mask = (batch == b)
        if nb[bi] > 0:
            gamma_hat_raw[:, bi] = resid[:, mask].mean(axis=1)
        else:
            gamma_hat_raw[:, bi] = 0.0

    # Standardize by pooled SD (within-batch residual spread)
    # Compute pooled sigma_j from residuals after removing batch means
    sigma = np.zeros(n_features)
    z = np.zeros_like(resid)
    for j in range(n_features):
        # remove per-batch means from residuals
        r = resid[j].copy()
        for bi, b in enumerate(batches):
            r[batch == b] -= gamma_hat_raw[j, bi]
        # pooled SD with ddof= (#batches + #covars)
        # use ddof>=1 safeguard
        dd = max(1, n_samples - X.shape[1] - n_batches)
        sj = np.sqrt(np.sum(r**2) / dd) if np.sum(r**2) > 0 else 1.0
        sigma[j] = sj if sj > 0 else 1.0
        z[j] = (resid[j] - np.take(gamma_hat_raw[j], [np.where(batches == bb)[0][0] for bb in batch])) / sigma[j]

    # Batch-wise stats on standardized scale
    # z_{js} ~ N(gamma_jb, delta_jb); estimate per-gene per-batch means/vars
    gamma_hat = np.zeros_like(gamma_hat_raw)  # standardized means m_jb
    delta_hat = np.ones_like(gamma_hat_raw)   # standardized variances s2_jb
    for bi, b in enumerate(batches):
        mask = (batch == b)
        if nb[bi] > 0:
            zj = z[:, mask]
            gamma_hat[:, bi] = zj.mean(axis=1)
            if not mean_only:
                # unbiased var; guard for nb==1
                v = np.var(zj, axis=1, ddof=1) if nb[bi] > 1 else np.ones(n_features)
                v[v <= 1e-8] = 1e-8
                delta_hat[:, bi] = v
            else:
                delta_hat[:, bi] = 1.0

    # Priors across genes (per batch)
    gamma_bar = gamma_hat.mean(axis=0)
    tau2 = np.var(gamma_hat, axis=0, ddof=1) + 1e-8  # stabilize

    if not mean_only:
        # Inverse-gamma priors for delta via method of moments
        a_prior = np.zeros(n_batches)
        b_prior = np.zeros(n_batches)
        for bi in range(n_batches):
            d = delta_hat[:, bi]
            m, v = d.mean(), d.var(ddof=1)
            # Handle degenerate cases
            if v <= 1e-8 or m <= 1e-8:
                a_prior[bi], b_prior[bi] = 1.0, 1.0
            else:
                a_prior[bi] = (2 * v + m**2) / v
                b_prior[bi] = (m * v + m**3) / v
    else:
        a_prior = np.ones(n_batches)
        b_prior = np.ones(n_batches)

    # Empirical-Bayes posteriors on standardized scale
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.copy(delta_hat)
    for bi, b in enumerate(batches):
        mask = (batch == b)
        nb_b = nb[bi]
        # sample mean m_jb and sample var s2_jb already computed: gamma_hat, delta_hat
        m = gamma_hat[:, bi]
        s2 = delta_hat[:, bi]

        if not mean_only:
            # posterior for delta (inverse-gamma): a_post, b_post
            # use within-batch SSE around the mean
            zj = z[:, mask]
            sse = np.sum((zj - m[:, None])**2, axis=1)
            a_post = a_prior[bi] + nb_b / 2.0
            b_post = b_prior[bi] + 0.5 * sse
            # posterior expectation of delta
            delta_star[:, bi] = b_post / (a_post - 1.0)
            delta_star[:, bi] = np.maximum(delta_star[:, bi], 1e-8)
        else:
            delta_star[:, bi] = 1.0

        # posterior mean for gamma with prior N(gamma_bar, tau2) and lik var delta/n
        w = nb_b / delta_star[:, bi]
        gamma_star[:, bi] = (w * m + (gamma_bar[bi] / tau2[bi])) / (w + (1.0 / tau2[bi]))

    # We want your existing wrappers to implement:
    #   remove: standardize, subtract gamma*, divide by sqrt(delta*), then unstandardize to a common space
    #   add:    multiply by sqrt(delta*_target), add gamma*_target, then unstandardize
    # To make your current formulas do exactly that, we store:
    #   beta_hat[:, b] = alpha_j  (grand mean per gene; same for all b)
    #   gamma_hat[:, b] = sigma_j * gamma*_jb
    #   delta_hat[:, b] = delta*_jb
    # where alpha_j = mean fitted value from covariates-only model; sigma_j = pooled SD above.
    alpha = fitted.mean(axis=1)  # grand mean per gene under covariate model
    beta_hat_store = np.tile(alpha[:, None], (1, n_batches))

    gamma_store = (sigma[:, None] * gamma_star)
    delta_store = delta_star

    combat_model = {
        'beta_hat': beta_hat_store,   # shape (n_genes, n_batches) all columns = alpha_j
        'var_pooled': sigma**2,       # not used by wrappers, kept for reference
        'gamma_hat': gamma_store,     # raw-space surrogate = sigma * gamma_star
        'delta_hat': delta_store,     # posterior variances on standardized scale
        'gamma_bar': gamma_bar,
        'tau2': tau2,
        'a_prior': a_prior,
        'b_prior': b_prior,
        'batches': batches,
        'design_matrix': X,           # covariate design used
        'n_batches': n_batches,
        'parametric': bool(parametric),
        'mean_only': bool(mean_only),
        'n_features': n_features,
        # keep for completeness
        'alpha': alpha,
        'sigma': sigma,
    }
    return combat_model

def combat_test(data, batch, combat_model, covariates=None):
    """
    Apply ComBat removal of current batch effects to map data into the
    common (batch-free) space consistent with the training above.

    Inputs
    ------
    data : array-like, shape (n_genes, n_samples)
    batch : array-like, shape (n_samples,)
    covariates : optional (ignored for mapping; covariates were already handled in training)

    Returns
    -------
    corrected_data : np.ndarray, shape (n_genes, n_samples)
        Data with the source batch removed (ready for your wrapper to add target batch).
    """
    import numpy as np
    import warnings

    data = np.array(data, dtype=float)
    batch = np.array(batch)

    n_features, n_samples = data.shape

    beta_hat = combat_model['beta_hat']   # stores alpha_j in every column
    gamma_hat = combat_model['gamma_hat'] # = sigma * gamma_star
    delta_hat = combat_model['delta_hat'] # = delta_star (standardized scale)
    batches = combat_model['batches']
    mean_only = combat_model['mean_only']
    n_batches = combat_model['n_batches']

    # Build one-hot for batch
    batch_design = np.zeros((n_samples, n_batches))
    for i, b in enumerate(batches):
        mask = (batch == b)
        if np.any(mask):
            batch_design[mask, i] = 1

    corrected_data = np.zeros_like(data)

    # Per sample -> identify its batch, then remove that batch effect
    for i in range(n_samples):
        b_idx = np.where(batch_design[i] == 1)[0]
        if len(b_idx) == 0:
            corrected_data[:, i] = data[:, i]
            warnings.warn(f"Unknown batch for sample {i}, no correction applied")
            continue
        b_idx = b_idx[0]

        # Common reference mean per gene (alpha_j)
        batch_mean = beta_hat[:, b_idx]  # same alpha_j across columns by construction

        # Remove batch: ((y - gamma) - alpha) / sqrt(delta) + alpha
        y = data[:, i]
        y_corr = y - gamma_hat[:, b_idx]
        if not mean_only:
            y_corr = ((y_corr - batch_mean) / np.sqrt(delta_hat[:, b_idx])) + batch_mean
        corrected_data[:, i] = y_corr

    return corrected_data

# Example usage
def example_usage_combat(microarray_train_df, microarray_test_df, rnaseq_train_df, rnaseq_test_df):
    """Example with perfectly paired data"""
    print(f"Training: {microarray_train_df.shape}")
    print(f"Testing: {microarray_test_df.shape}")
    
    # Run evaluation
    results = combat_evaluate_paired(
        microarray_train_df, rnaseq_train_df,
        microarray_test_df, rnaseq_test_df
    )
    
    return results