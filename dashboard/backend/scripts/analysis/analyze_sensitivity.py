import glob
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_log_file(filepath, num_last_epochs=20):
    """
    Parses a GANomics training log file.
    Averages the metrics over the last `num_last_epochs` epochs to reflect final stable convergence.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Extract losses from each line
    data = []
    for line in lines:
        match = re.search(r'\(epoch: (\d+),.*G_A: ([\d\.]+) G_B: ([\d\.]+) D_A: ([\d\.]+) D_B: ([\d\.]+) cycle_A: ([\d\.]+) cycle_B: ([\d\.]+) feedback_A: ([\d\.]+) feedback_B: ([\d\.]+)', line)
        if match:
            ep = int(match.group(1))
            g_a = float(match.group(2))
            g_b = float(match.group(3))
            d_a = float(match.group(4))
            d_b = float(match.group(5))
            cycle = float(match.group(6)) + float(match.group(7))
            feedback = float(match.group(8)) + float(match.group(9))
            
            data.append({
                'epoch': ep,
                'G_A': g_a,
                'G_B': g_b,
                'D_A': d_a,
                'D_B': d_b,
                'Cycle': cycle,
                'Feedback': feedback
            })
            
    if not data:
        return None
        
    df = pd.DataFrame(data)
    
    # We want the average of the last `num_last_epochs` epochs.
    # The log records multiple iterations per epoch.
    # We first average within each epoch to avoid bias from iteration counts.
    epoch_means = df.groupby('epoch').mean()
    
    # Take the last `num_last_epochs` available epochs
    last_epochs = epoch_means.tail(num_last_epochs)
    
    # Return the mean of these last stable epochs for this single run
    return last_epochs.mean().to_dict()

def analyze_parameter(param_name, log_pattern):
    logs = glob.glob(log_pattern)
    if not logs:
        print(f"\nNo logs found for {param_name} sensitivity.")
        return None
        
    # Dictionary to hold all runs for a specific parameter value
    # results[param_value] = [run0_dict, run1_dict, ...]
    results = defaultdict(list)
    
    for log in logs:
        # Expected pattern: *_Sensitivity_Beta_10.0_Run_0_log.txt or *_Sensitivity_Lambda_5.0_Run_1_log.txt
        match = re.search(fr'{param_name}_([\d\.]+)_Run_(\d+)', log)
        if not match:
            continue
            
        val = float(match.group(1))
        run_id = int(match.group(2))
        
        run_stats = parse_log_file(log)
        if run_stats:
            results[val].append(run_stats)
            
    if not results:
        print(f"\nNo valid data parsed for {param_name} sensitivity.")
        return None

    # Aggregate across runs (mean and standard deviation)
    summary = []
    for val in sorted(results.keys()):
        runs = results[val]
        n_runs = len(runs)
        
        aggregated = {'Value': val, 'Runs': n_runs}
        
        for metric in ['G_A', 'G_B', 'Cycle', 'Feedback']:
            run_values = [r[metric] for r in runs if metric in r]
            if run_values:
                mean_val = np.mean(run_values)
                # Use sample standard deviation (ddof=1) if n_runs > 1, else 0
                std_val = np.std(run_values, ddof=1) if n_runs > 1 else 0.0
                
                aggregated[f'{metric}_mean'] = mean_val
                aggregated[f'{metric}_std'] = std_val
                aggregated[f'{metric}_str'] = f"{mean_val:.3f} ± {std_val:.3f}"
            else:
                aggregated[f'{metric}_str'] = "N/A"
                
        summary.append(aggregated)
        
    df_summary = pd.DataFrame(summary)
    
    # Print formatted Markdown table
    print(f"\n### Sensitivity Analysis for {param_name}")
    print(f"| {param_name} | Runs | G_A (Adversarial) | G_B (Adversarial) | Cycle-Consistency | Feedback Loss |")
    print("|---|" + "---|" * 5)
    for row in summary:
        print(f"| **{row['Value']}** | {row['Runs']} | {row['G_A_str']} | {row['G_B_str']} | {row['Cycle_str']} | {row['Feedback_str']} |")
        
    return df_summary

if __name__ == "__main__":
    print("Parsing logs and generating sensitivity reports...")
    
    # Determine backend directory
    # This script is in dashboard/backend/scripts/analysis/analyze_sensitivity.py
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    beta_pattern = resolve_path("results/1_Training/logs/*Sensitivity_Beta_*_log.txt")
    lambda_pattern = resolve_path("results/1_Training/logs/*Sensitivity_Lambda_*_log.txt")
    
    analyze_parameter("Beta", beta_pattern)
    analyze_parameter("Lambda", lambda_pattern)
