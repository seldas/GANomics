import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.interpolate import PchipInterpolator

# ---------- CuBlock helpers ----------
def _choose_odd_power(z_sorted, within_sd_mask,
                      candidate_p=(3,5,7,9,11,13,15,17,19,21),
                      target_abs_mean=0.10):
    n = z_sorted.size
    if n == 0 or not np.any(within_sd_mask):
        return candidate_p[0]
    u = np.linspace(-1.0, 1.0, n, dtype=float)
    for p in candidate_p:
        y = np.sign(u) * (np.abs(u) ** p)
        if np.mean(np.abs(y[within_sd_mask])) < target_abs_mean:
            return p
    return candidate_p[-1]

def _enforce_monotone(y_sorted):
    # isotonic-like: cumulative max enforces non-decreasing
    return np.maximum.accumulate(y_sorted)


def _cublock_normalize_block(vals):
    x = np.asarray(vals, dtype=float)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s <= 1e-12:
        return np.zeros_like(x)

    z = (x - m) / s
    order = np.argsort(z)
    z_sorted = z[order]
    within_sd = (np.abs(z_sorted) <= 1.0)

    p = _choose_odd_power(z_sorted, within_sd)
    u = np.linspace(-1.0, 1.0, z_sorted.size, dtype=float)
    y_target = np.sign(u) * (np.abs(u) ** p)

    # cubic fit on sorted z → target y
    try:
        coef = np.polyfit(z_sorted, y_target, deg=3)
    except Exception:
        coef = np.polyfit(z_sorted, y_target, deg=1)

    # evaluate cubic on the *unsorted* z
    y_hat = np.polyval(coef, z)

    # optional: gentle monotone safeguard by projecting the *sorted* (z,y_hat) to monotone
    # and re-interpolating only if strong inversions are detected
    if np.any(np.diff(np.polyval(coef, z_sorted)) < 0):
        y_fit_sorted = np.polyval(coef, z_sorted)
        y_mono_sorted = np.maximum.accumulate(y_fit_sorted)
        z_unique, idx = np.unique(z_sorted, return_index=True)
        y_unique = y_mono_sorted[idx]
        y_hat = np.interp(z, z_unique, y_unique)

    return y_hat

''' old version, not original
def _cublock_normalize_block(vals):
    x = np.asarray(vals, dtype=float)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s <= 1e-12:
        return np.zeros_like(x)

    z = (x - m) / s
    order = np.argsort(z)
    z_sorted = z[order]
    within_sd = (np.abs(z_sorted) <= 1.0)

    p = _choose_odd_power(z_sorted, within_sd)
    u = np.linspace(-1.0, 1.0, z_sorted.size, dtype=float)
    y_target = np.sign(u) * (np.abs(u) ** p)

    # cubic (fallback to linear)
    try:
        coef = np.polyfit(z_sorted, y_target, deg=3)
        y_fit_sorted = np.polyval(coef, z_sorted)
    except Exception:
        coef = np.polyfit(z_sorted, y_target, deg=1)
        y_fit_sorted = np.polyval(coef, z_sorted)

    y_mono_sorted = _enforce_monotone(y_fit_sorted)

    # unique z for safe interpolation
    z_unique, idx = np.unique(z_sorted, return_index=True)
    y_unique = y_mono_sorted[idx]
    if z_unique.size < 2:
        return np.zeros_like(x)

    return np.interp(z, z_unique, y_unique)
'''

def fit_cublock_clusters_source(df_source_train: pd.DataFrame,
                                k: int = 5,
                                n_repetitions: int = 30,
                                random_state: int = 0):
    """
    Learn CuBlock gene clusters for the SOURCE platform and freeze them.
    Returns a dict with k-means labels (one array per repetition) and seeds.
    """
    X = df_source_train.astype(float).to_numpy()   # (n_samples, n_genes)
    G = X.T                                        # (n_genes, n_samples)
    n_genes = G.shape[0]

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31-1, size=n_repetitions)

    labels_list = []
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init="auto", random_state=int(seed))
        labels = km.fit_predict(G)                 # len n_genes
        labels_list.append(labels.astype(np.int32))

    return {
        "k": k,
        "n_repetitions": n_repetitions,
        "random_state": int(random_state),
        "seeds": seeds.tolist(),
        "labels_list": labels_list,                # list of length n_repetitions
        "genes": df_source_train.columns.tolist(), # to enforce column order
    }

def apply_cublock_with_frozen_clusters(df_source: pd.DataFrame,
                                       cluster_model: dict,
                                       verbose: bool = False) -> pd.DataFrame:
    """
    CuBlock-normalize SOURCE data using frozen gene clusters learned on source training.
    Averages across repetitions (as in the paper).
    """
    # enforce gene order
    genes = cluster_model["genes"]
    missing = [g for g in genes if g not in df_source.columns]
    if missing:
        raise ValueError(f"Input missing {len(missing)} genes seen in training; e.g., {missing[:5]}")

    X = df_source.loc[:, genes].astype(float).to_numpy()   # (n_samples, n_genes)
    n_samples, n_genes = X.shape

    k = int(cluster_model["k"])
    labels_list = cluster_model["labels_list"]

    accum = np.zeros_like(X, dtype=float)

    for rep, labels in enumerate(labels_list, start=1):
        if verbose:
            print(f"[CuBlock] Apply rep {rep}/{len(labels_list)}")

        # precompute blocks
        blocks = [np.where(labels == c)[0] for c in range(k)]
        Y = np.empty_like(X, dtype=float)

        for s in range(n_samples):
            row = X[s]
            for c_idx in range(k):
                idx = blocks[c_idx]
                if idx.size == 0:
                    continue
                block_vals = row[idx]
                if idx.size < 4 or not np.isfinite(block_vals).all():
                    m = np.nanmean(block_vals)
                    sd = np.nanstd(block_vals, ddof=1)
                    Y[s, idx] = 0.0 if (not np.isfinite(sd) or sd <= 1e-12) else (block_vals - m)/sd
                else:
                    Y[s, idx] = _cublock_normalize_block(block_vals)

        accum += Y

    out = accum / float(len(labels_list))
    return pd.DataFrame(out, index=df_source.index, columns=genes)

def fit_cublock_translator(source_train_df: pd.DataFrame,
                           target_train_df: pd.DataFrame,
                           k: int = 5,
                           n_repetitions: int = 30,
                           random_state: int = 0,
                           q_grid: np.ndarray | None = None):
    """
    Build a CuBlock-inspired translator:
      1) Learn & freeze source CuBlock clusters on source training data.
      2) CuBlock-normalize source training data with the frozen clusters.
      3) For each gene, fit a monotone PCHIP f_j: source_CuBlock -> target_raw using quantiles.

    Returns a model dict you can use at inference.
    """
    if q_grid is None:
        q_grid = np.linspace(0.01, 0.99, 99)

    # 1) Fit source CuBlock clusters
    cluster_model = fit_cublock_clusters_source(
        source_train_df, k=k, n_repetitions=n_repetitions, random_state=random_state
    )

    # 2) Normalize source training set with frozen clusters
    src_norm = apply_cublock_with_frozen_clusters(source_train_df, cluster_model)

    # 3) Per-gene monotone maps to target raw (requires identical gene set/order)
    genes = cluster_model["genes"]
    missing_t = [g for g in genes if g not in target_train_df.columns]
    if missing_t:
        raise ValueError(f"Target training is missing {len(missing_t)} genes; e.g., {missing_t[:5]}")

    maps = {}  # gene -> dict(xq, yq)
    for g in genes:
        x = src_norm[g].to_numpy(dtype=float)               # source CuBlock-normalized values
        y = target_train_df[g].to_numpy(dtype=float)        # target raw values

        xq = np.quantile(x, q_grid)
        yq = np.quantile(y, q_grid)

        # ensure strictly increasing xq for PCHIP safety
        xq_eps = np.maximum.accumulate(xq + np.linspace(0, 1e-8, xq.size))
        maps[g] = {"xq": xq_eps.astype(float), "yq": yq.astype(float)}

    return {
        "cluster_model": cluster_model,
        "q_grid": q_grid.tolist(),
        "gene_maps": maps,
        "genes": genes,
        "k": k,
        "n_repetitions": n_repetitions,
        "random_state": int(random_state),
        "note": "CuBlock-normalize(source) + per-gene PCHIP to target raw (translator)",
    }

def translate_cublock(source_df: pd.DataFrame,
                      translator_model: dict,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Given new SOURCE samples, produce synthetic TARGET profiles.
    Steps:
      - CuBlock-normalize with frozen source clusters
      - Per-gene apply monotone PCHIP mapping to target raw space
    """
    genes = translator_model["genes"]
    missing = [g for g in genes if g not in source_df.columns]
    if missing:
        raise ValueError(f"Source input missing {len(missing)} genes; e.g., {missing[:5]}")

    # CuBlock-normalize with frozen clusters
    src_norm = apply_cublock_with_frozen_clusters(
        source_df.loc[:, genes], translator_model["cluster_model"], verbose=verbose
    )

    out = np.empty_like(src_norm.to_numpy(), dtype=float)
    for j, g in enumerate(genes):
        gm = translator_model["gene_maps"][g]
        f = PchipInterpolator(gm["xq"], gm["yq"], extrapolate=True)
        out[:, j] = f(src_norm[g].to_numpy(dtype=float))

    return pd.DataFrame(out, index=source_df.index, columns=genes)
