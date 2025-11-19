import pandas as pd
import numpy as np

def example_usage_quantile(train_ag, test_ag, train_ngs, test_ngs):
    """
    Bidirectional quantile normalization using the target platform's training distribution.
    Returns:
        {
          'rnaseq_to_microarray': DataFrame  # test_ngs -> microarray-like (QN to AG reference)
          'microarray_to_rnaseq': DataFrame  # test_ag  -> RNA-seq-like   (QN to NGS reference)
        }
    Reference: Thompson et al., PeerJ 2016; see 'normalize.quantiles.use.target' concept. 
    """

    def _reference_sorted_vector(ref_df: pd.DataFrame) -> np.ndarray:
        # For each sample (row), sort values; then mean across samples at each rank
        X = ref_df.to_numpy(dtype=float)  # shape (n_samples, n_genes)
        sorted_by_row = np.sort(X, axis=1)
        ref_sorted = sorted_by_row.mean(axis=0)  # length n_genes
        return ref_sorted

    def _quantile_map_to_reference(test_df: pd.DataFrame, ref_sorted: np.ndarray) -> pd.DataFrame:
        X = test_df.to_numpy(dtype=float)
        n_samples, n_genes = X.shape
        if ref_sorted.shape[0] != n_genes:
            raise ValueError("Reference and test must have the same number of genes (columns).")

        Y = np.empty_like(X, dtype=float)
        for i in range(n_samples):
            x = X[i, :]
            # argsort to get ranks; stable sort to handle ties consistently
            order = np.argsort(x, kind='mergesort')
            # place ref_sorted values into the sorted positions, then invert the permutation
            y_sorted = ref_sorted
            y = np.empty_like(x, dtype=float)
            y[order] = y_sorted
            Y[i, :] = y
        return pd.DataFrame(Y, index=test_df.index, columns=test_df.columns)

    # Build reference distributions from training data of the *target* platforms
    ref_ag_sorted  = _reference_sorted_vector(train_ag)   # microarray ref
    ref_ngs_sorted = _reference_sorted_vector(train_ngs)  # RNA-seq ref

    # Apply to test sets from the *source* platforms
    df_ma_qn = _quantile_map_to_reference(test_ngs, ref_ag_sorted)   # RNA-seq -> Microarray-like
    df_rs_qn = _quantile_map_to_reference(test_ag,  ref_ngs_sorted)  # Microarray -> RNA-seq-like

    return {'rnaseq_to_microarray': df_ma_qn,
            'microarray_to_rnaseq': df_rs_qn}
