import pandas as pd
import os

nb_dir = r'dashboard\backend\dataset\NB'

files_to_convert = [
    ('NB_AG.csv', 'NB_AG.tsv'),
    ('NB_NGS.csv', 'NB_NGS.tsv'),
    ('clinical_info.csv', 'clinical_info.tsv')
]

for old_name, new_name in files_to_convert:
    old_path = os.path.join(nb_dir, old_name)
    new_path = os.path.join(nb_dir, new_name)
    
    if os.path.exists(old_path):
        print(f"Converting {old_name} to {new_name}...")
        df = pd.read_csv(old_path, index_col=0)
        df.to_csv(new_path, sep='\t')
        os.remove(old_path)
        print(f"Successfully created {new_name} and removed {old_name}")
    else:
        print(f"Skipping {old_name} (not found)")
