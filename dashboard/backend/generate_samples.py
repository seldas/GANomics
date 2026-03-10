import os
import yaml
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")

def generate_samples():
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_config.yaml"):
                config_path = os.path.join(root, file)
                print(f"Processing {config_path}...")
                try:
                    with open(config_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                    
                    # Path A is usually the primary microarray/AG file
                    path_a = cfg['dataset']['path_A']
                    if not os.path.isabs(path_a):
                        path_a = os.path.abspath(os.path.join(BACKEND_DIR, path_a))
                    
                    if os.path.exists(path_a):
                        # Read only the header to get sample names (columns)
                        df_headers = pd.read_csv(path_a, index_col=0, nrows=0, sep=None, engine='python')
                        samples = df_headers.columns.tolist()
                        
                        samples_path = os.path.join(root, "samples.tsv")
                        pd.DataFrame({'sample_id': samples}).to_csv(samples_path, sep='\t', index=False)
                        print(f"  Generated {samples_path} with {len(samples)} samples.")
                    else:
                        print(f"  File not found: {path_a}")
                except Exception as e:
                    print(f"  Error: {e}")

if __name__ == "__main__":
    generate_samples()
