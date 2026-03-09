import os
import argparse
import yaml
import pandas as pd
from src.datasets.preprocessing import full_preprocess_pipeline

def main():
    parser = argparse.ArgumentParser(description="Preprocess Genomic Datasets")
    parser.add_argument("--config", type=str, default="configs/preprocess_config.yaml", help="Path to preprocess config")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_root = config.get('output_root', 'dataset')
    results = []
    
    for ds_name, ds_config in config['datasets'].items():
        try:
            output_dir = os.path.join(output_root, ds_config.get('output_subfolder', ds_name))
            shape = full_preprocess_pipeline(
                path_A=ds_config['path_A'],
                path_B=ds_config['path_B'],
                output_dir=output_dir,
                name=ds_name,
                config=ds_config
            )
            results.append({
                'Dataset': ds_name,
                'Samples': shape[0],
                'Genes': shape[1]
            })
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            
    # Generate Table 1 (Overview)
    if results:
        table_df = pd.DataFrame(results)
        table_path = os.path.join("results/tables", "Table_1_Dataset_Overview.csv")
        os.makedirs("results/tables", exist_ok=True)
        table_df.to_csv(table_path, index=False)
        print(f"\nTable 1 saved to {table_path}")
        print(table_df)

if __name__ == "__main__":
    main()
