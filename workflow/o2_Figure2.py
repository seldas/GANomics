#!/usr/bin/env python3
"""
Plot GANomics results for two experiment families using all available runs.

Figure layout: 3 rows x 2 columns
    Left column  = family like NB_<size>_<repeat>
    Right column = family like NB_Size_<size>_Run_<repeat>

Panels:
    Row 1: Training feedback loss (Total_L) vs epoch, mean ± SD over repeats
    Row 2: Microarray test distribution, broken x-axis removing 50-200
    Row 3: RNA-seq test distribution, broken x-axis removing 50-200

Expected directories:
    dashboard/backend/results_ms/1_Training/logs/
    dashboard/backend/results_ms/2_SyncData/{exp_id}/test/

Example:
    python plot_all_ganomics_families.py \
        --root /compute001/lwu/projects/GANomics \
        --output ganomics_family_comparison.png
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# -----------------------------
# Experiment ID parsing
# -----------------------------
NB_PATTERN = re.compile(r"^NB_(\d+)_(\d+)$")
def parse_exp_id(exp_id: str):
    """
    Returns:
        (family_name, sample_size, repeat_id)

    Rules:
        size <= 50  -> right panel
        size >= 100 -> left panel
    """
    m = NB_PATTERN.match(exp_id)
    if not m:
        return None

    sample_size = int(m.group(1))
    repeat_id = int(m.group(2))

    if sample_size <= 50:
        family_name = "right"
    elif sample_size >= 100:
        family_name = "left"
    else:
        return None

    return family_name, sample_size, repeat_id


# -----------------------------
# Log parsing
# -----------------------------
def parse_loss_log(log_path: Path) -> pd.DataFrame:
    rows = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "====" in line:
                continue

            pieces = re.split(r"(?<!:)\s+", line)
            pieces = [re.sub(r"[\(\),]", "", p) for p in pieces if p]

            curr = {}
            for item in pieces:
                if ":" not in item:
                    continue
                try:
                    k, v = re.split(r":\s*", item, maxsplit=1)
                    curr[k] = float(v)
                except Exception:
                    continue

            if curr:
                for k in [
                    "D_A", "G_A", "cycle_A", "idt_A",
                    "D_B", "G_B", "cycle_B", "idt_B",
                    "median_A", "median_B"
                ]:
                    curr.setdefault(k, np.nan)

                curr["Total_G"] = np.nansum([curr["G_A"], curr["G_B"]])
                curr["Total_D"] = np.nansum([curr["D_A"], curr["D_B"]])
                curr["Total_C"] = np.nansum([curr["cycle_A"], curr["cycle_B"]])
                curr["Total_L"] = np.nansum([curr["median_A"], curr["median_B"]])

                rows.append(curr)

    if not rows:
        raise ValueError(f"No parseable rows found in log: {log_path}")

    df = pd.DataFrame(rows)

    if "epoch" not in df.columns:
        raise ValueError(f"Column 'epoch' not found in parsed log: {log_path}")

    sort_cols = ["epoch"] + (["iters"] if "iters" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # keep the last row per epoch
    df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df


def find_log_file(logs_dir: Path, exp_id: str) -> Path:
    candidates = [
        logs_dir / exp_id / "loss_log.txt",
        logs_dir / exp_id / "train.log",
        logs_dir / f"{exp_id}.txt",
        logs_dir / f"{exp_id}.log",
        logs_dir / f"{exp_id}_loss_log.txt",
        logs_dir / f"loss_log_{exp_id}.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = []
    for pat in [
        f"**/{exp_id}/loss_log.txt",
        f"**/*{exp_id}*loss*.txt",
        f"**/*{exp_id}*.txt",
        f"**/*{exp_id}*.log",
    ]:
        matches.extend(logs_dir.glob(pat))

    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find log file for exp_id={exp_id} under {logs_dir}")

    matches = sorted(matches, key=lambda p: (exp_id not in str(p), len(str(p))))
    return matches[0]


# -----------------------------
# Test CSV loading
# -----------------------------
def load_flat_values(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        num_df = df.apply(pd.to_numeric, errors="coerce")

    arr = num_df.to_numpy().ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def get_test_files(sync_root: Path, exp_id: str):
    test_dir = sync_root / exp_id / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    files = {
        "microarray_fake": test_dir / "microarray_fake.csv",
        "microarray_real": test_dir / "microarray_real.csv",
        "rnaseq_fake": test_dir / "rnaseq_fake.csv",
        "rnaseq_real": test_dir / "rnaseq_real.csv",
    }

    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing expected test CSVs:\n" + "\n".join(missing))

    return files


# -----------------------------
# Discovery
# -----------------------------
def discover_experiments(sync_root: Path):
    """
    Discovers experiments from subdirectories in 2_SyncData and groups them.

    Returns:
        {
            "left": {
                sample_size: [exp_id1, exp_id2, ...]
            },
            "right": {
                sample_size: [exp_id1, exp_id2, ...]
            }
        }
    """
    grouped = {"left": defaultdict(list), "right": defaultdict(list)}

    if not sync_root.exists():
        raise FileNotFoundError(f"Sync root does not exist: {sync_root}")

    for child in sorted(sync_root.iterdir()):
        if not child.is_dir():
            continue

        parsed = parse_exp_id(child.name)
        if parsed is None:
            continue

        family, sample_size, repeat_id = parsed
        grouped[family][sample_size].append(child.name)

    grouped["left"] = dict(sorted(grouped["left"].items()))
    grouped["right"] = dict(sorted(grouped["right"].items()))
    return grouped


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_losses(logs_dir: Path, exp_ids_by_size: dict):
    """
    Returns:
        {sample_size: DataFrame(epoch, mean, std, n)}
    """
    out = {}

    for sample_size, exp_ids in exp_ids_by_size.items():
        dfs = []
        for exp_id in exp_ids:
            log_path = find_log_file(logs_dir, exp_id)
            df = parse_loss_log(log_path)[["epoch", "Total_L"]].copy()
            df["exp_id"] = exp_id
            dfs.append(df)

        if not dfs:
            continue

        all_df = pd.concat(dfs, axis=0, ignore_index=True)
        agg = (
            all_df.groupby("epoch")["Total_L"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
        agg["std"] = agg["std"].fillna(0.0)
        out[sample_size] = agg

    return out


def aggregate_test_values(sync_root: Path, exp_ids_by_size: dict):
    """
    Returns:
        {
            sample_size: {
                "microarray_fake": np.array,
                "microarray_real": np.array,
                "rnaseq_fake": np.array,
                "rnaseq_real": np.array,
            }
        }
    """
    out = {}

    for sample_size, exp_ids in exp_ids_by_size.items():
        store = {
            "microarray_fake": [],
            "microarray_real": [],
            "rnaseq_fake": [],
            "rnaseq_real": [],
        }

        for exp_id in exp_ids:
            files = get_test_files(sync_root, exp_id)
            for key, path in files.items():
                vals = load_flat_values(path)
                if vals.size > 0:
                    store[key].append(vals)

        merged = {}
        for key, chunks in store.items():
            merged[key] = np.concatenate(chunks) if chunks else np.array([], dtype=float)

        out[sample_size] = merged

    return out


# -----------------------------
# Plot helpers
# -----------------------------
def make_broken_axes(fig, parent_spec, width_ratios=(1.0, 1.45), wspace=0.05):
    inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=parent_spec, width_ratios=width_ratios, wspace=wspace
    )
    ax_l = fig.add_subplot(inner[0, 0])
    ax_r = fig.add_subplot(inner[0, 1], sharey=ax_l)
    return ax_l, ax_r


def add_break_marks(ax_l, ax_r, d=0.012):
    kwargs = dict(transform=ax_l.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs["transform"] = ax_r.transAxes
    ax_r.plot((-d, +d), (-d, +d), **kwargs)
    ax_r.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def density_curve(values: np.ndarray, bins: np.ndarray):
    if values.size == 0:
        return np.full(len(bins) - 1, np.nan)

    hist, edges = np.histogram(values, bins=bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, hist


def get_family_label(family_name: str):
    return "NB_<size>_<repeat>" if family_name == "left" else "NB_Size_<size>_Run_<repeat>"


def get_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_loss_panel(ax, agg_losses: dict, family_name: str):
    sizes = sorted(agg_losses.keys())
    colors = get_colors(len(sizes))

    for color, size in zip(colors, sizes):
        df = agg_losses[size]
        x = df["epoch"].to_numpy()
        y = df["mean"].to_numpy()
        s = df["std"].to_numpy()

        ax.plot(x, y, linewidth=2, label=str(size), color=color)
        ax.fill_between(x, y - s, y + s, alpha=0.18, color=color)

    ax.set_title(f"{get_family_label(family_name)}\nTraining Feedback Loss (mean ± SD)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total_L")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(title="Train size", frameon=False, ncol=2, fontsize=9, title_fontsize=9)


def plot_distribution_panel(
    fig,
    parent_spec,
    agg_tests: dict,
    family_name: str,
    data_key_fake: str,
    data_key_real: str,
    title: str,
    xlabel: str = "Expression value",
    low_range=(0, 50),
    high_range=(200, None),
    bins_low=80,
    bins_high=80,
):
    ax_l, ax_r = make_broken_axes(fig, parent_spec)

    sizes = sorted(agg_tests.keys())
    colors = get_colors(len(sizes))

    pooled_real = []
    for size in sizes:
        vals = agg_tests[size][data_key_real]
        if vals.size > 0:
            pooled_real.append(vals)

    pooled_real = np.concatenate(pooled_real) if pooled_real else np.array([], dtype=float)

    fake_all = []
    for size in sizes:
        vals = agg_tests[size][data_key_fake]
        if vals.size > 0:
            fake_all.append(vals)

    fake_all = np.concatenate(fake_all) if fake_all else np.array([], dtype=float)
    vmax = 250.0
    if pooled_real.size or fake_all.size:
        vmax = float(np.nanmax(np.concatenate([pooled_real, fake_all])))
        vmax = max(vmax, high_range[0] + 10)

    high_range = (high_range[0], vmax)

    bins1 = np.linspace(low_range[0], low_range[1], bins_low)
    bins2 = np.linspace(high_range[0], high_range[1], bins_high)

    # pooled real as black reference
    if pooled_real.size > 0:
        real_low = pooled_real[(pooled_real >= low_range[0]) & (pooled_real <= low_range[1])]
        real_high = pooled_real[(pooled_real >= high_range[0]) & (pooled_real <= high_range[1])]

        x1, y1 = density_curve(real_low, bins1)
        x2, y2 = density_curve(real_high, bins2)

        ax_l.plot(x1, y1, linewidth=2.4, color="black", label="Real pooled")
        ax_r.plot(x2, y2, linewidth=2.4, color="black")

    # fake by size
    for color, size in zip(colors, sizes):
        fake_vals = agg_tests[size][data_key_fake]
        if fake_vals.size == 0:
            continue

        low_vals = fake_vals[(fake_vals >= low_range[0]) & (fake_vals <= low_range[1])]
        high_vals = fake_vals[(fake_vals >= high_range[0]) & (fake_vals <= high_range[1])]

        if low_vals.size > 0:
            x1, y1 = density_curve(low_vals, bins1)
            ax_l.plot(x1, y1, linewidth=1.6, color=color, label=f"Fake {size}")
        if high_vals.size > 0:
            x2, y2 = density_curve(high_vals, bins2)
            ax_r.plot(x2, y2, linewidth=1.6, color=color)

    ax_l.set_xlim(*low_range)
    ax_r.set_xlim(*high_range)

    ax_l.set_title(f"{get_family_label(family_name)}\n{title}")
    ax_l.set_ylabel("Density")
    ax_l.set_xlabel(xlabel)
    ax_r.set_xlabel(xlabel)

    ax_l.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax_r.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax_r.yaxis.set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    add_break_marks(ax_l, ax_r)

    handles, labels = ax_l.get_legend_handles_labels()
    ax_l.legend(handles, labels, frameon=False, fontsize=8, ncol=2)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot all GANomics runs for left and right experiment families.")
    parser.add_argument(
        "--root",
        type=Path,
        default = '/compute001/lwu/projects/GANomics/',
        help="Project root containing dashboard/backend/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ganomics_family_comparison.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    backend_dir = root / "dashboard" / "backend"
    logs_dir = backend_dir / "results_ms" / "1_Training" / "logs"
    sync_root = backend_dir / "results_ms" / "2_SyncData"

    if not backend_dir.exists():
        raise FileNotFoundError(f"backend directory not found: {backend_dir}")
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs directory not found: {logs_dir}")
    if not sync_root.exists():
        raise FileNotFoundError(f"sync data directory not found: {sync_root}")

    discovered = discover_experiments(sync_root)
    left_groups = discovered["left"]
    right_groups = discovered["right"]

    if not left_groups:
        raise RuntimeError(f"No left-family experiments found under {sync_root}")
    if not right_groups:
        raise RuntimeError(f"No right-family experiments found under {sync_root}")

    print("Discovered left-family sample sizes:")
    for k, v in left_groups.items():
        print(f"  {k}: {len(v)} run(s)")

    print("Discovered right-family sample sizes:")
    for k, v in right_groups.items():
        print(f"  {k}: {len(v)} run(s)")

    left_losses = aggregate_losses(logs_dir, left_groups)
    right_losses = aggregate_losses(logs_dir, right_groups)

    left_tests = aggregate_test_values(sync_root, left_groups)
    right_tests = aggregate_test_values(sync_root, right_groups)

    plt.style.use("default")
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)

    # Row 1: training feedback loss
    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    plot_loss_panel(ax11, left_losses, "left")
    plot_loss_panel(ax12, right_losses, "right")

    # Row 2: microarray
    plot_distribution_panel(
        fig, gs[1, 0], left_tests, "left",
        data_key_fake="microarray_fake",
        data_key_real="microarray_real",
        title="Microarray test distribution",
    )
    plot_distribution_panel(
        fig, gs[1, 1], right_tests, "right",
        data_key_fake="microarray_fake",
        data_key_real="microarray_real",
        title="Microarray test distribution",
    )

    # Row 3: RNA-seq
    plot_distribution_panel(
        fig, gs[2, 0], left_tests, "left",
        data_key_fake="rnaseq_fake",
        data_key_real="rnaseq_real",
        title="RNA-seq test distribution",
    )
    plot_distribution_panel(
        fig, gs[2, 1], right_tests, "right",
        data_key_fake="rnaseq_fake",
        data_key_real="rnaseq_real",
        title="RNA-seq test distribution",
    )

    fig.suptitle(
        "GANomics comparison across all runs\n"
        "Left: NB_<size>_<repeat>    Right: NB_Size_<size>_Run_<repeat>",
        fontsize=14,
        y=0.98,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to: {args.output.resolve()}")


if __name__ == "__main__":
    main()