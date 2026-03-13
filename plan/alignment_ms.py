import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os

def calculate_l1_distance(real, fake):
    return np.mean(np.abs(real - fake))

def calculate_metrics(folder):
    # Load real datasets
    microarray_real = pd.read_csv(os.path.join(folder, 'microarray_real.csv'), index_col=0)
    rnaseq_real = pd.read_csv(os.path.join(folder, 'rnaseq_real.csv'), index_col=0)
    
    # Load fake datasets
    microarray_fake = pd.read_csv(os.path.join(folder, 'microarray_fake.csv'), index_col=0)
    rnaseq_fake = pd.read_csv(os.path.join(folder, 'rnaseq_fake.csv'), index_col=0)
    
    # Calculate metrics for microarray
    pearson_ma = pearsonr(microarray_real.values.flatten(), microarray_fake.values.flatten())[0]
    spearman_ma = spearmanr(microarray_real.values.flatten(), microarray_fake.values.flatten())[0]
    l1_ma = calculate_l1_distance(microarray_real.values, microarray_fake.values)
    
    # Calculate metrics for rnaseq
    pearson_rs = pearsonr(rnaseq_real.values.flatten(), rnaseq_fake.values.flatten())[0]
    spearman_rs = spearmanr(rnaseq_real.values.flatten(), rnaseq_fake.values.flatten())[0]
    l1_rs = calculate_l1_distance(rnaseq_real.values, rnaseq_fake.values)
    
    # Return results as two rows of array
    row1 = np.array([folder, 'Microarray', pearson_ma, spearman_ma, l1_ma])
    row2 = np.array([folder, 'RNAseq', pearson_rs, spearman_rs, l1_rs])
    
    return [row1, row2]

# Example usage
folder_used = ['NB_Size_50_run_0',
               'NB_Size_50_run_1',
               'NB_Size_50_run_2',
               'NB_Size_50_run_3',
               'NB_Size_50_run_4',
               'CycleGAN_50_0',
               'CycleGAN_50_1',
               'CycleGAN_50_2',
               'CycleGAN_50_3',
               'CycleGAN_50_4',]
res=[]
for folder in folder_used:
    curr_folder = f'dashboard/backend/results_ms/2_SyncData/{folder}/test'
    metrics = calculate_metrics(curr_folder)
    res = res + metrics
df_res = pd.DataFrame(res, columns=['Task','Platform', 'Pearson','Spearman','L1'])
df_res.to_excel('Alignment_result_ms.xlsx')

