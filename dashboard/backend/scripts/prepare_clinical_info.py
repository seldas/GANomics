import pandas as pd
import os

# Define paths
txt_path = r'plan\SEQC_NB_249_ValidationSamples_ClinicalInfo_20121128.txt'
out_path = r'dashboard\backend\dataset\NB\clinical_info.csv'

# Ensure directory exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Read TXT (tab-separated)
df = pd.read_csv(txt_path, sep='\t')

# Select relevant columns
# Based on the header: SEQC_NB_SampleID, D_FAV_All (Favorable/Unfavorable)
# The user hint mentioned D_FAV_All as the label column.
cols = ['SEQC_NB_SampleID', 'D_FAV_All']
df_subset = df[cols].copy()

# Rename for convenience
df_subset.rename(columns={'SEQC_NB_SampleID': 'sample_id', 'D_FAV_All': 'label'}, inplace=True)

# Map numeric labels to text if necessary (0: Favorable, 1: Unfavorable)
# Based on the first few rows: 0 is favorable, 1 is unfavorable
df_subset['label'] = df_subset['label'].map({0: 'Favorable', 1: 'Unfavorable'})

# Drop rows with missing labels
df_subset.dropna(subset=['label'], inplace=True)

# Save as CSV
df_subset.to_csv(out_path, index=False)
print(f"Clinical info saved to {out_path} ({len(df_subset)} samples)")
