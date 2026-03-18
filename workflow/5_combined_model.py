import os
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add the backend directory to sys.path to make 'src' importable
PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(PLAN_DIR, "..", "dashboard", "backend"))
sys.path.insert(0, BACKEND_DIR)

from src.core.analysis import train_eval_rf


RANDOM_STATE = 42
TEST_SIZE = 0.5
MIN_FEATURES = 10


def load_labels(project_id: str) -> pd.Series:
    """Load sample labels for a given project ID."""
    label_id = 'NB' if project_id == 'CycleGAN' else project_id
    candidates = [
        os.path.join(BACKEND_DIR, "dataset", label_id, "label.txt"),
        os.path.join(BACKEND_DIR, "dataset", label_id, "labels.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df_labels = pd.read_csv(path, index_col=0, sep=None, engine='python')
            if 'label' not in df_labels.columns:
                raise ValueError(f"Label file found but 'label' column missing: {path}")
            return df_labels['label']
    raise FileNotFoundError(f"No label file found for project '{project_id}'. Checked: {candidates}")



def average_profiles(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """Average matched real and synthetic profiles sample-wise and feature-wise."""
    common_idx = real_df.index.intersection(syn_df.index)
    common_cols = real_df.columns.intersection(syn_df.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return pd.DataFrame(index=common_idx, columns=common_cols)
    return (real_df.loc[common_idx, common_cols] + syn_df.loc[common_idx, common_cols]) / 2.0



def stack_profiles(real_df: pd.DataFrame, syn_df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a simple augmentation set by stacking real and synthetic training samples."""
    common_idx = real_df.index.intersection(syn_df.index).intersection(y.index)
    common_cols = real_df.columns.intersection(syn_df.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return pd.DataFrame(), pd.Series(dtype=y.dtype)

    real_sub = real_df.loc[common_idx, common_cols].copy()
    syn_sub = syn_df.loc[common_idx, common_cols].copy()

    # Keep row names unique after stacking.
    real_sub.index = [f"{idx}__REAL" for idx in real_sub.index]
    syn_sub.index = [f"{idx}__SYN" for idx in syn_sub.index]

    x_aug = pd.concat([real_sub, syn_sub], axis=0)
    y_aug = pd.concat([
        pd.Series(y.loc[common_idx].values, index=real_sub.index),
        pd.Series(y.loc[common_idx].values, index=syn_sub.index),
    ])
    return x_aug, y_aug



def evaluate_scenarios(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    y: pd.Series,
    include_extra: bool = True,
) -> pd.DataFrame:
    """Evaluate real/synthetic/combined training scenarios for one platform."""
    common_idx = real_df.index.intersection(syn_df.index).intersection(y.index)
    common_cols = real_df.columns.intersection(syn_df.columns)
    if len(common_idx) < 10 or len(common_cols) < MIN_FEATURES:
        return pd.DataFrame()

    real_sub = real_df.loc[common_idx, common_cols]
    syn_sub = syn_df.loc[common_idx, common_cols]
    avg_sub = average_profiles(real_sub, syn_sub)

    train_idx, test_idx = train_test_split(
        common_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.loc[common_idx],
    )

    metrics: List[Dict] = []

    scenarios = [
        ("Real->Real", real_sub.loc[train_idx], y.loc[train_idx], real_sub.loc[test_idx], y.loc[test_idx]),
        ("Real+Syn(avg)->Real", avg_sub.loc[train_idx], y.loc[train_idx], real_sub.loc[test_idx], y.loc[test_idx]),
        ("Syn->Real", syn_sub.loc[train_idx], y.loc[train_idx], real_sub.loc[test_idx], y.loc[test_idx]),
        ("Syn->Syn", syn_sub.loc[train_idx], y.loc[train_idx], syn_sub.loc[test_idx], y.loc[test_idx]),
    ]

    if include_extra:
        # Helpful extra checks for augmentation and symmetry.
        x_aug, y_aug = stack_profiles(real_sub.loc[train_idx], syn_sub.loc[train_idx], y.loc[train_idx])
        if not x_aug.empty:
            scenarios.extend([
                ("Real+Syn(stack)->Real", x_aug, y_aug, real_sub.loc[test_idx], y.loc[test_idx]),
                ("Real+Syn(stack)->Syn", x_aug, y_aug, syn_sub.loc[test_idx], y.loc[test_idx]),
            ])

        scenarios.extend([
            ("Real->Syn", real_sub.loc[train_idx], y.loc[train_idx], syn_sub.loc[test_idx], y.loc[test_idx]),
            ("Real+Syn(avg)->Syn", avg_sub.loc[train_idx], y.loc[train_idx], syn_sub.loc[test_idx], y.loc[test_idx]),
        ])

    for scenario_name, x_train, y_train, x_test, y_test in scenarios:
        try:
            out = train_eval_rf(x_train, y_train, x_test, y_test)
            out['Scenario'] = scenario_name
            out['Train_N'] = len(x_train)
            out['Test_N'] = len(x_test)
            out['N_Features'] = x_train.shape[1]
            metrics.append(out)
        except Exception as exc:
            print(f"  Error in scenario '{scenario_name}': {exc}")

    return pd.DataFrame(metrics)



def load_algorithms(test_dir: str, algo_dir: str) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Load GANomics outputs and any baseline algorithm outputs."""
    ma_real = pd.read_csv(os.path.join(test_dir, "microarray_real.csv"), index_col=0)
    rs_real = pd.read_csv(os.path.join(test_dir, "rnaseq_real.csv"), index_col=0)
    ma_fake_gan = pd.read_csv(os.path.join(test_dir, "microarray_fake.csv"), index_col=0)
    rs_fake_gan = pd.read_csv(os.path.join(test_dir, "rnaseq_fake.csv"), index_col=0)

    # Match the gene-space handling used in the existing workflow.
    rs_real.columns = ma_real.columns
    rs_fake_gan.columns = ma_real.columns

    algorithms: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ('GANomics', ma_fake_gan, rs_fake_gan)
    ]

    if os.path.exists(algo_dir):
        algo_names = set()
        for file_name in os.listdir(algo_dir):
            if file_name.endswith('.csv'):
                name = file_name.replace('microarray_fake_', '').replace('rnaseq_fake_', '').replace('.csv', '')
                algo_names.add(name.upper())

        for algo_name in sorted(algo_names):
            ma_path = os.path.join(algo_dir, f"microarray_fake_{algo_name.lower()}.csv")
            rs_path = os.path.join(algo_dir, f"rnaseq_fake_{algo_name.lower()}.csv")
            if os.path.exists(ma_path) and os.path.exists(rs_path):
                try:
                    ma_syn = pd.read_csv(ma_path, index_col=0)
                    rs_syn = pd.read_csv(rs_path, index_col=0)
                    # Align gene names to MA space when needed.
                    if rs_syn.shape[1] == ma_real.shape[1]:
                        rs_syn.columns = ma_real.columns
                    algorithms.append((algo_name, ma_syn, rs_syn))
                except Exception as exc:
                    print(f"Skipping algorithm {algo_name}: {exc}")

    return ma_real, rs_real, algorithms



def run_task(run_id: str, sync_root: str, out_root: str, include_extra: bool) -> None:
    task_dir = os.path.join(sync_root, run_id)
    test_dir = os.path.join(task_dir, 'test')
    algo_dir = os.path.join(task_dir, 'algorithm')
    project_id = run_id.split('_')[0]

    if not os.path.exists(test_dir):
        print(f"Skipping {run_id}: missing test directory")
        return

    try:
        ma_real, rs_real, algorithms = load_algorithms(test_dir, algo_dir)
        y_all = load_labels(project_id)
    except Exception as exc:
        print(f"Skipping {run_id}: {exc}")
        return

    common_idx = ma_real.index.intersection(rs_real.index).intersection(y_all.index)
    if len(common_idx) < 10:
        print(f"Skipping {run_id}: too few labeled samples ({len(common_idx)})")
        return

    y = y_all.loc[common_idx]
    ma_real = ma_real.loc[common_idx]
    rs_real = rs_real.loc[common_idx]

    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {run_id} with {len(algorithms)} algorithms ...")
    for algo_name, ma_syn, rs_syn in algorithms:
        ma_syn = ma_syn.loc[common_idx]
        rs_syn = rs_syn.loc[common_idx]

        for platform_name, real_df, syn_df in [
            ('MA', ma_real, ma_syn),
            ('RS', rs_real, rs_syn),
        ]:
            result_df = evaluate_scenarios(real_df, syn_df, y, include_extra=include_extra)
            if result_df.empty:
                print(f"  No valid features for {algo_name}_{platform_name}")
                continue

            file_name = f"Classifier_Performance_Combined_{algo_name}_{platform_name}.csv"
            result_df.to_csv(os.path.join(out_dir, file_name), index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Combined real+synthetic predictive modeling workflow")
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='dashboard/backend/results',
        help='Parent results directory containing 2_SyncData',
    )
    parser.add_argument(
        '--task_filter',
        type=str,
        default='NB_Ablation_Size_50_Run_0',
        help='Only process task IDs containing this substring',
    )
    parser.add_argument(
        '--include_extra',
        action='store_true',
        help='Also evaluate extra symmetric/augmentation scenarios',
    )
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    sync_root = os.path.join(parent_dir, '2_SyncData')
    out_root = os.path.join(parent_dir, '5_CombinedModel')

    if not os.path.exists(sync_root):
        raise FileNotFoundError(f"Sync root not found: {sync_root}")

    task_ids = [
        task_id for task_id in os.listdir(sync_root)
        if os.path.isdir(os.path.join(sync_root, task_id))
    ]
    if args.task_filter:
        task_ids = [task_id for task_id in task_ids if args.task_filter in task_id]

    print(f"Found {len(task_ids)} tasks in {sync_root}")
    for task_id in tqdm(sorted(task_ids), desc='Processing combined models'):
        run_task(task_id, sync_root, out_root, include_extra=args.include_extra)

    print(f"✅ Combined-model analysis completed. Results saved to {out_root}")


if __name__ == '__main__':
    main()
