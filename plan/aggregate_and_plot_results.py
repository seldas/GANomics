import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Directories containing the results
directories = [
    'dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_0',
    'dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_1',
    'dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_2',
    'dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_3',
    'dashboard/backend/results/4_Biomarkers/Prediction/NB_Ablation_Size_50_Run_4'
]

# Algorithms to compare
algorithms = ['GANomics', 'COMBAT', 'CUBLOCK', 'QN', 'TDM', 'YUGENE']

# Scenarios to compare
scenarios = ['Real->Syn', 'Syn->Real']

# Initialize a dictionary to store the results
results = {algo: {scenario: [] for scenario in scenarios} for algo in algorithms}

# Aggregate results from the CSV files
for directory in directories:
    for algo in algorithms:
        file_path_ma = os.path.join(directory, f'Classifier_Performance_{algo}_MA.csv')
        file_path_rs = os.path.join(directory, f'Classifier_Performance_{algo}_RS.csv')
        
        if os.path.exists(file_path_ma) and os.path.exists(file_path_rs):
            df_ma = pd.read_csv(file_path_ma)
            df_rs = pd.read_csv(file_path_rs)
            
            for scenario in scenarios:
                accuracy_ma = df_ma.loc[df_ma['Scenario'] == scenario, 'Accuracy'].mean() if scenario in df_ma['Scenario'].values else np.nan
                accuracy_rs = df_rs.loc[df_rs['Scenario'] == scenario, 'Accuracy'].mean() if scenario in df_rs['Scenario'].values else np.nan
                
                # Store the average accuracy
                results[algo][scenario].append((accuracy_ma + accuracy_rs) / 2)

# Calculate mean and standard deviation for each algorithm and scenario
means = {algo: {scenario: np.mean(values) for scenario, values in results[algo].items()} for algo in algorithms}
stds = {algo: {scenario: np.std(values) for scenario, values in results[algo].items()} for algo in algorithms}

# Create a bar chart with error bars for each scenario
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, scenario in enumerate(scenarios):
    x = np.arange(len(algorithms))
    axes[i].bar(x, [means[algo][scenario] for algo in algorithms], yerr=[stds[algo][scenario] for algo in algorithms], capsize=5)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(algorithms)
    axes[i].set_ylabel('Accuracy')
    axes[i].set_title(scenario)
    axes[i].set_xlabel('Algorithm')

    # Ensure GANomics is on the left
    if 'GANomics' in algorithms:
        algo_order = ['GANomics'] + [algo for algo in algorithms if algo != 'GANomics']
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(algo_order)

plt.tight_layout()
plt.show()
