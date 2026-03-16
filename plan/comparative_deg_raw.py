import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests
import glob
import os

sns.set(style="whitegrid", context="talk")

# -----------------------------
# Parameters
# -----------------------------
base_path = r"dashboard\backend\results_ms\4_Biomarkers\DEG"
N_REPEATS = 5

# -----------------------------
# Load and process data
# -----------------------------
data_list = []
baseline_list = []

ganomics_folders = glob.glob(os.path.join(base_path, "NB_Size_50_run_*"))
cyclegan_folders = glob.glob(os.path.join(base_path, "CycleGAN_50_*"))

def process_folder(folder, method_name):
    for direction in ["MA_to_RS", "RS_to_MA"]:
        file_name = f"Jaccard_Curve_GANomics_{direction}.csv"
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            tmp_df = pd.read_csv(file_path)
            tmp_df["recall"] = tmp_df["n_overlap"] / tmp_df["n_real"]
            tmp_df["Method"] = method_name
            tmp_df["Direction"] = direction
            tmp_df["Folder"] = os.path.basename(folder)
            data_list.append(tmp_df)
    
    # Process baseline (only once per folder or once per direction? 
    # Usually Jaccard_Curve_Baseline.csv is the same for both directions because it's real vs real)
    baseline_path = os.path.join(folder, "Jaccard_Curve_Baseline.csv")
    if os.path.exists(baseline_path):
        tmp_df = pd.read_csv(baseline_path)
        tmp_df["recall"] = tmp_df["n_overlap"] / tmp_df["n_real"]
        tmp_df["Folder"] = os.path.basename(folder)
        baseline_list.append(tmp_df)

for folder in ganomics_folders:
    process_folder(folder, "GANomics")

for folder in cyclegan_folders:
    process_folder(folder, "CycleGAN")

all_data = pd.concat(data_list, ignore_index=True)
all_baseline = pd.concat(baseline_list, ignore_index=True)

# Summarize baseline across all folders (or per folder type?)
# Let's just summarize all baselines since they should be similar
baseline_summary = all_baseline.groupby("threshold")["recall"].mean().reset_index()

# Summarize methods
summary_df = all_data.groupby(["Method", "Direction", "threshold"])["recall"].agg(["mean", "std"]).reset_index()
summary_df["threshold"] = summary_df["threshold"].astype(str)

# -----------------------------
# Statistical testing
# -----------------------------
pvals = []

for direction in summary_df["Direction"].unique():
    for thr in summary_df["threshold"].unique():
        sub = summary_df[(summary_df.Direction == direction) & (summary_df.threshold == str(thr))]
        
        if len(sub) < 2:
            continue
            
        g1 = sub[sub.Method == "CycleGAN"]
        g2 = sub[sub.Method == "GANomics"]
        
        if len(g1) == 0 or len(g2) == 0:
            continue
            
        g1 = g1.iloc[0]
        g2 = g2.iloc[0]
        
        t, p = ttest_ind_from_stats(
            mean1=g1["mean"],
            std1=g1["std"] if not pd.isna(g1["std"]) else 0,
            nobs1=N_REPEATS,
            mean2=g2["mean"],
            std2=g2["std"] if not pd.isna(g2["std"]) else 0,
            nobs2=N_REPEATS,
            equal_var=False
        )
        
        pvals.append({
            "Direction": direction,
            "threshold": thr,
            "pval": p
        })

pval_df = pd.DataFrame(pvals)

if not pval_df.empty:
    # FDR correction
    pval_df["p_adj"] = multipletests(pval_df["pval"], method="fdr_bh")[1]

    # significance labels
    def star(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    pval_df["sig"] = pval_df["p_adj"].apply(star)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

for ax, direction in zip(axes, summary_df.Direction.unique()):
    sub = summary_df[summary_df.Direction == direction]
    
    # Sort thresholds numerically for plot
    sub["threshold_num"] = sub["threshold"].astype(float)
    sub = sub.sort_values("threshold_num")
    unique_thrs = sorted(sub["threshold_num"].unique())
    sub["threshold"] = pd.Categorical(sub["threshold"], categories=[str(t) for t in unique_thrs], ordered=True)

    sns.barplot(
        data=sub,
        x="threshold",
        y="mean",
        hue="Method",
        ax=ax,
        palette="muted",
        alpha=0.8
    )

    # Add error bars
    for i, row in sub.iterrows():
        # Find index in Categorical
        x_idx = list(sub["threshold"].unique()).index(row["threshold"])
        # seaborn hue offset is roughly +/- 0.2
        offset = -0.2 if row["Method"] == "CycleGAN" else 0.2

        ax.errorbar(
            x_idx + offset,
            row["mean"],
            yerr=row["std"] if not pd.isna(row["std"]) else 0,
            fmt="none",
            c="black",
            capsize=5
        )

    # Add baseline dashed line
    # Map baseline threshold to x-axis
    baseline_summary_sorted = baseline_summary.copy()
    baseline_summary_sorted = baseline_summary_sorted[baseline_summary_sorted["threshold"].isin(unique_thrs)]
    baseline_summary_sorted = baseline_summary_sorted.sort_values("threshold")
    
    # We can plot the baseline as a line across the bars
    # Since thresholds are discrete on x-axis, we can plot a step line or just points connected
    x_vals = range(len(unique_thrs))
    y_vals = baseline_summary_sorted["recall"].values
    
    ax.plot(x_vals, y_vals, linestyle="--", color="red", label="Baseline (Real vs Real)", marker="o")

    # Significance annotation
    if not pval_df.empty:
        sig_sub = pval_df[pval_df.Direction == direction]
        for i, row in sig_sub.iterrows():
            try:
                x_idx = list(sub["threshold"].unique()).index(str(row["threshold"]))
                y_max = sub[sub.threshold == str(row["threshold"])]["mean"].max()
                y_pos = max(y_max, baseline_summary[baseline_summary.threshold == float(row["threshold"])]["recall"].values[0]) + 0.05
                ax.text(x_idx, y_pos, row["sig"], ha='center', fontsize=14, fontweight="bold")
            except:
                continue

    ax.set_title(f"Direction: {direction}", fontsize=16)
    ax.set_ylabel("DEG Recall (Overlap / Real)", fontsize=14)
    ax.set_xlabel("P-value Threshold", fontsize=14)
    ax.legend(title="Method", fontsize=12)
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("plan/comparative_deg_recall_plot.png")
print("Plot saved to plan/comparative_deg_recall_plot.png")
plt.show()
