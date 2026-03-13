import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel('Alignment_result_ms.xlsx')

# Filter data for GANomics and CycleGAN
ganomics_ma = df[(df['Task'].str.contains('run')) & (df['Platform'] == 'Microarray')]
ganomics_rs = df[(df['Task'].str.contains('run')) & (df['Platform'] == 'RNAseq')]
cyclegan_ma = df[(df['Task'].str.contains('CycleGAN')) & (df['Platform'] == 'Microarray')]
cyclegan_rs = df[(df['Task'].str.contains('CycleGAN')) & (df['Platform'] == 'RNAseq')]

# Perform t-test
t_stat_pearson_ma, p_val_pearson_ma = ttest_ind(ganomics_ma['Pearson'], cyclegan_ma['Pearson'])
t_stat_spearman_ma, p_val_spearman_ma = ttest_ind(ganomics_ma['Spearman'], cyclegan_ma['Spearman'])
t_stat_l1_ma, p_val_l1_ma = ttest_ind(ganomics_ma['L1'], cyclegan_ma['L1'])

t_stat_pearson_rs, p_val_pearson_rs = ttest_ind(ganomics_rs['Pearson'], cyclegan_rs['Pearson'])
t_stat_spearman_rs, p_val_spearman_rs = ttest_ind(ganomics_rs['Spearman'], cyclegan_rs['Spearman'])
t_stat_l1_rs, p_val_l1_rs = ttest_ind(ganomics_rs['L1'], cyclegan_rs['L1'])

# Print t-test results
print("Microarray:")
print(f"Pearson: t-stat = {t_stat_pearson_ma:.4f}, p-val = {p_val_pearson_ma:.4f}")
print(f"Spearman: t-stat = {t_stat_spearman_ma:.4f}, p-val = {p_val_spearman_ma:.4f}")
print(f"L1: t-stat = {t_stat_l1_ma:.4f}, p-val = {p_val_l1_ma:.4f}")

print("\nRNA-seq:")
print(f"Pearson: t-stat = {t_stat_pearson_rs:.4f}, p-val = {p_val_pearson_rs:.4f}")
print(f"Spearman: t-stat = {t_stat_spearman_rs:.4f}, p-val = {p_val_spearman_rs:.4f}")
print(f"L1: t-stat = {t_stat_l1_rs:.4f}, p-val = {p_val_l1_rs:.4f}")

# Plot bar plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].bar(['GANomics', 'CycleGAN'], [ganomics_ma['Pearson'].mean(), cyclegan_ma['Pearson'].mean()])
ax[0].set_title('Microarray Pearson Correlation')
ax[0].set_xlabel('Method')
ax[0].set_ylabel('Pearson Correlation')

ax[1].bar(['GANomics', 'CycleGAN'], [ganomics_rs['Pearson'].mean(), cyclegan_rs['Pearson'].mean()])
ax[1].set_title('RNA-seq Pearson Correlation')
ax[1].set_xlabel('Method')
ax[1].set_ylabel('Pearson Correlation')

plt.tight_layout()
plt.show()