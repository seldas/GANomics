import numpy as np
import pandas as pd

def example_usage_tdm(train_ag, test_ag, train_ngs, test_ngs, data_is_log2=True, eps=1e-12):
    """
    Bidirectional TDM translation between microarray (AG) and RNA-seq (NGS).
    Returns:
        {
          'rnaseq_to_microarray': DataFrame  # test_ngs -> microarray-like
          'microarray_to_rnaseq': DataFrame  # test_ag  -> RNA-seq-like
        }
    Assumptions:
      - Rows are samples; columns are matched genes (already aligned in your code).
      - Values are approximately log2-scaled if data_is_log2=True (your pipeline uses log2 for both).
    References: Thompson et al., PeerJ 2016 (TDM).  DOI:10.7717/peerj.1621
    """
    def _tdm_match(source_test_df, source_train_df, target_train_df, data_is_log2=True):
        # helper to compute robust quantiles on flattened arrays
        def _five_number_stats(x):
            x = x[np.isfinite(x)]
            q1, q3 = np.quantile(x, [0.25, 0.75])
            iqr = max(q3 - q1, eps)
            return x.min(), q1, q3, x.max(), iqr

        # training (target) distribution statistics
        tgt_vals = target_train_df.to_numpy(dtype=float).ravel()
        t_min, t_q1, t_q3, t_max, t_iqr = _five_number_stats(tgt_vals)
        top_ratio = (t_max - t_q3) / t_iqr
        bot_ratio = max(0.0, (t_q1 - t_min) / t_iqr)

        # test (source) distribution statistics
        test_vals = source_test_df.to_numpy(dtype=float)
        flat = test_vals.ravel()
        _, s_q1, s_q3, _, s_iqr = _five_number_stats(flat)

        # winsorize test to bounds implied by target tail/IQR ratios
        lower = s_q1 - bot_ratio * s_iqr
        upper = s_q3 + top_ratio * s_iqr
        flat_w = np.clip(flat, lower, upper)

        # rescale to training range (optionally in inverse-log space per TDM)
        if data_is_log2:
            # map in linear space, then return to log2
            lower_lin = np.power(2.0, lower)
            upper_lin = np.power(2.0, upper)
            flat_lin  = np.power(2.0, flat_w)
            t_min_lin = np.power(2.0, t_min)
            t_max_lin = np.power(2.0, t_max)

            denom = max(upper_lin - lower_lin, eps)
            scaled = (flat_lin - lower_lin) / denom
            scaled = np.clip(scaled, 0.0, 1.0)
            mapped_lin = scaled * (t_max_lin - t_min_lin) + t_min_lin
            mapped = np.log2(mapped_lin + eps)
        else:
            denom = max(upper - lower, eps)
            scaled = (flat_w - lower) / denom
            scaled = np.clip(scaled, 0.0, 1.0)
            mapped = scaled * (t_max - t_min) + t_min

        out = mapped.reshape(test_vals.shape)
        return pd.DataFrame(out, index=source_test_df.index, columns=source_test_df.columns)

    # RNA-seq -> Microarray
    df_ma_tdm = _tdm_match(source_test_df=test_ngs,
                           source_train_df=train_ngs,
                           target_train_df=train_ag,
                           data_is_log2=data_is_log2)

    # Microarray -> RNA-seq  (less common; provided for symmetry)
    df_rs_tdm = _tdm_match(source_test_df=test_ag,
                           source_train_df=train_ag,
                           target_train_df=train_ngs,
                           data_is_log2=data_is_log2)

    return {'rnaseq_to_microarray': df_ma_tdm,
            'microarray_to_rnaseq': df_rs_tdm}
