import pandas as pd
import os

def main():
    ag_path = r'dashboard\backend\dataset\NB\df_ag.tsv'
    mapping_file = r'dashboard\backend\dataset\NB\GPL16876-29646.txt'
    output_path = r'dashboard\backend\dataset\NB\gene_mapping.tsv'

    print("Reading probe IDs from df_ag.tsv...")
    # Read only the header of df_ag.tsv
    with open(ag_path, 'r') as f:
        header = f.readline().strip().split('\t')
    
    # The first element might be empty if it's the index column
    probes_in_data = set(header)
    if '' in probes_in_data:
        probes_in_data.remove('')
    
    print(f"Found {len(probes_in_data)} probes in dataset.")

    print("Parsing mapping table...")
    # Read the mapping table, skipping metadata lines
    # Based on investigation, the header is at index 25
    df_map = pd.read_csv(mapping_file, sep='\t', skiprows=25, low_memory=False)
    
    # Columns of interest: Agilent_Probe_Name, GeneSymbol, GeneName, GO
    cols = ['Agilent_Probe_Name', 'GeneSymbol', 'GeneName', 'GO']
    
    # Check if columns exist
    missing_cols = [c for c in cols if c not in df_map.columns]
    if missing_cols:
        print(f"Error: Missing columns in mapping file: {missing_cols}")
        print(f"Available columns: {df_map.columns.tolist()}")
        return

    # Filter by probes present in df_ag.tsv
    df_filtered = df_map[df_map['Agilent_Probe_Name'].isin(probes_in_data)]
    
    # Keep only requested columns
    df_result = df_filtered[cols]
    
    # Remove duplicates if any (multiple probes might map to same info, but here we want probe-wise mapping)
    # Actually, we want one entry per probe.
    df_result = df_result.drop_duplicates(subset=['Agilent_Probe_Name'])

    print(f"Successfully mapped {len(df_result)} probes.")
    
    # Save to TSV
    df_result.to_csv(output_path, sep='\t', index=False)
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    main()
