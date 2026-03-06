import os
import argparse
import pandas as pd
from src.datasets.preprocessing import preprocess_dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess Genomic Datasets")
    parser.add_argument("--datasets", nargs='+', default=['NB'], help="List of datasets to process")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()
    
    # Placeholder for dataset paths (in a real scenario, these would be in a config)
    dataset_info = {
        'NB': {
            'A': 'dataset/NB/NB_AG_121.txt',
            'B': 'dataset/NB/NB_NGS_121.txt',
        },
        'METSIM': {
            'A': 'dataset/METSIM/df_ag.tsv',
            'B': 'dataset/METSIM/df_ngs.tsv',
        }
    }
    
    results = []
    for name in args.datasets:
        if name in dataset_info:
            info = dataset_info[name]
            shape = preprocess_dataset(info['A'], info['B'], args.output_dir, name)
            results.append({
                'Dataset': name,
                'Samples': shape[0],
                'Genes': shape[1]
            })
            
    # Generate Table 1 (Overview)
    if results:
        table_df = pd.DataFrame(results)
        table_path = "results/tables/Table_1_Dataset_Overview.csv"
        os.makedirs("results/tables", exist_ok=True)
        table_df.to_csv(table_path, index=False)
        print(f"\nTable 1 saved to {table_path}")
        print(table_df)

if __name__ == "__main__":
    main()
