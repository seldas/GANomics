import os
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add backend directory to sys.path
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", "dashboard", "backend"))
sys.path.insert(0, BACKEND_DIR)

from src.core.analysis import train_eval_rf


MIN_FEATURES = 10
TRAIN_SIZES = [10, 20, 50, 100, 150]
N_REPEATS = 10
TEST_SIZE = 0.5


def load_labels(project_id: str) -> pd.Series:
    """
    Load labels from dataset folder.
    """
    label_id = "NB" if project_id == "CycleGAN" else project_id
    candidates = [
        os.path.join(BACKEND_DIR, "dataset", label_id, "label.txt"),
        os.path.join(BACKEND_DIR, "dataset", label_id, "labels.txt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            df_labels = pd.read_csv(path, index_col=0, sep=None, engine="python")
            if "label" not in df_labels.columns:
                raise ValueError(f"'label' column missing in {path}")
            return df_labels["label"]

    raise FileNotFoundError(f"No label file found for project '{project_id}'")


def average_profiles(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average matched real and synthetic samples element-wise.
    """
    common_idx = real_df.index.intersection(syn_df.index)
    common_cols = real_df.columns.intersection(syn_df.columns)
    return (real_df.loc[common_idx, common_cols] + syn_df.loc[common_idx, common_cols]) / 2.0


def stack_profiles(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Stack real and synthetic samples to create an augmented training set.
    """
    common_idx = real_df.index.intersection(syn_df.index).intersection(y.index)
    common_cols = real_df.columns.intersection(syn_df.columns)

    real_sub = real_df.loc[common_idx, common_cols].copy()
    syn_sub = syn_df.loc[common_idx, common_cols].copy()

    real_sub.index = [f"{idx}__REAL" for idx in real_sub.index]
    syn_sub.index = [f"{idx}__SYN" for idx in syn_sub.index]

    x_aug = pd.concat([real_sub, syn_sub], axis=0)
    y_aug = pd.concat([
        pd.Series(y.loc[common_idx].values, index=real_sub.index),
        pd.Series(y.loc[common_idx].values, index=syn_sub.index),
    ])

    return x_aug, y_aug


def stratified_subsample(
    idx: pd.Index,
    y: pd.Series,
    train_size: int,
    random_state: int
) -> pd.Index:
    """
    Draw a balanced stratified subset of the training pool.
    """
    y_sub = y.loc[idx]

    if train_size >= len(idx):
        return idx

    sampled_idx, _ = train_test_split(
        idx,
        train_size=train_size,
        random_state=random_state,
        stratify=y_sub
    )
    return pd.Index(sampled_idx)


def evaluate_scenarios_multi_size(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    y: pd.Series,
    train_sizes: List[int],
    n_repeats: int
) -> pd.DataFrame:
    """
    Evaluate predictive scenarios across multiple training sizes and repeats.
    """
    common_idx = real_df.index.intersection(syn_df.index).intersection(y.index)
    common_cols = real_df.columns.intersection(syn_df.columns)

    if len(common_idx) < 20 or len(common_cols) < MIN_FEATURES:
        return pd.DataFrame()

    real_sub = real_df.loc[common_idx, common_cols].copy()
    syn_sub = syn_df.loc[common_idx, common_cols].copy()
    y_sub = y.loc[common_idx].copy()
    avg_sub = average_profiles(real_sub, syn_sub)

    results = []

    for repeat in range(n_repeats):
        random_state = 1000 + repeat

        # Fixed outer split for this repeat
        train_pool_idx, test_idx = train_test_split(
            common_idx,
            test_size=TEST_SIZE,
            random_state=random_state,
            stratify=y_sub.loc[common_idx]
        )

        y_train_pool = y_sub.loc[train_pool_idx]
        y_test = y_sub.loc[test_idx]

        # Check maximum feasible train size
        max_train_pool = len(train_pool_idx)
        valid_train_sizes = [n for n in train_sizes if n <= max_train_pool]

        for train_size in valid_train_sizes:
            try:
                train_idx = stratified_subsample(
                    train_pool_idx,
                    y_sub,
                    train_size=train_size,
                    random_state=random_state + train_size
                )
            except ValueError as exc:
                print(f"Skipping train_size={train_size}, repeat={repeat}: {exc}")
                continue

            X_real_train = real_sub.loc[train_idx]
            X_syn_train = syn_sub.loc[train_idx]
            X_avg_train = avg_sub.loc[train_idx]
            y_train = y_sub.loc[train_idx]

            X_real_test = real_sub.loc[test_idx]
            X_syn_test = syn_sub.loc[test_idx]

            # Augmented stacked training set
            X_stack_train, y_stack_train = stack_profiles(X_real_train, X_syn_train, y_train)

            scenarios = [
                ("Real->Real", X_real_train, y_train, X_real_test, y_test),
                ("Real+Syn(avg)->Real", X_avg_train, y_train, X_real_test, y_test),
                ("Syn->Real", X_syn_train, y_train, X_real_test, y_test),
                ("Syn->Syn", X_syn_train, y_train, X_syn_test, y_test),
                ("Real+Syn(stack)->Real", X_stack_train, y_stack_train, X_real_test, y_test),
                ("Real+Syn(stack)->Syn", X_stack_train, y_stack_train, X_syn_test, y_test),
            ]

            for scenario_name, x_train, y_train_use, x_test, y_test_use in scenarios:
                try:
                    out = train_eval_rf(x_train, y_train_use, x_test, y_test_use)
                    out["Scenario"] = scenario_name
                    out["Repeat"] = repeat + 1
                    out["Train_Size_Base"] = train_size
                    out["Train_N_Actual"] = len(x_train)
                    out["Test_N"] = len(x_test)
                    out["N_Features"] = x_train.shape[1]
                    results.append(out)
                except Exception as exc:
                    print(
                        f"Error in scenario={scenario_name}, "
                        f"train_size={train_size}, repeat={repeat + 1}: {exc}"
                    )

    return pd.DataFrame(results)


def load_ganomics_outputs(test_dir: str):
    """
    Load GANomics real/synthetic outputs only.
    """
    ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
    ma_syn = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
    rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
    rs_syn = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)

    # Align RS column names to MA space if dimensions match
    if rs_real.shape[1] == ma_real.shape[1]:
        rs_real.columns = ma_real.columns
    if rs_syn.shape[1] == ma_real.shape[1]:
        rs_syn.columns = ma_real.columns

    return ma_real, ma_syn, rs_real, rs_syn


def run_task(run_id: str, sync_root: str, out_root: str) -> None:
    """
    Run multi-size combined-model evaluation for one GANomics task.
    """
    task_dir = os.path.join(sync_root, run_id)
    test_dir = os.path.join(task_dir, "test")
    project_id = run_id.split("_")[0]

    if not os.path.exists(test_dir):
        print(f"Skipping {run_id}: missing test directory")
        return

    try:
        ma_real, ma_syn, rs_real, rs_syn = load_ganomics_outputs(test_dir)
        y_all = load_labels(project_id)
    except Exception as exc:
        print(f"Skipping {run_id}: {exc}")
        return

    # Intersect sample IDs with labels
    common_idx = ma_real.index.intersection(rs_real.index).intersection(y_all.index)
    if len(common_idx) < 30:
        print(f"Skipping {run_id}: too few labeled samples ({len(common_idx)})")
        return

    y = y_all.loc[common_idx]
    ma_real = ma_real.loc[common_idx]
    ma_syn = ma_syn.loc[common_idx]
    rs_real = rs_real.loc[common_idx]
    rs_syn = rs_syn.loc[common_idx]

    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {run_id} ...")

    for platform_name, real_df, syn_df in [
        ("MA", ma_real, ma_syn),
        ("RS", rs_real, rs_syn),
    ]:
        result_df = evaluate_scenarios_multi_size(
            real_df=real_df,
            syn_df=syn_df,
            y=y,
            train_sizes=TRAIN_SIZES,
            n_repeats=N_REPEATS
        )

        if result_df.empty:
            print(f"  No valid results for {platform_name}")
            continue

        out_path = os.path.join(
            out_dir,
            f"Classifier_Performance_Combined_GANomics_{platform_name}.csv"
        )
        result_df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined real/synthetic predictive modeling for GANomics only"
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="dashboard/backend/results",
        help="Parent results directory containing 2_SyncData"
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="NB_Ablation_Size_50_Run_",
        help="Only process task IDs containing this substring"
    )
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    sync_root = os.path.join(parent_dir, "2_SyncData")
    out_root = os.path.join(parent_dir, "5_CombinedModel")

    if not os.path.exists(sync_root):
        raise FileNotFoundError(f"Sync root not found: {sync_root}")

    task_ids = [
        task_id for task_id in os.listdir(sync_root)
        if os.path.isdir(os.path.join(sync_root, task_id))
    ]

    if args.task_filter:
        task_ids = [task_id for task_id in task_ids if args.task_filter in task_id]

    print(f"Found {len(task_ids)} tasks in {sync_root}")
    for task_id in tqdm(sorted(task_ids), desc="Processing combined models"):
        run_task(task_id, sync_root, out_root)

    print(f"Done. Results saved to {out_root}")


if __name__ == "__main__":
    main()