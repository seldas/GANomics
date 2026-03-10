import os
import yaml
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")

def get_project_stats(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        path_a = cfg['dataset']['path_A']
        
        # Resolve path relative to backend
        full_path_a = os.path.abspath(os.path.join(BACKEND_DIR, path_a))
        
        if os.path.exists(full_path_a):
            # Strict Orientation: Columns are Genes, Rows are Samples
            df_headers = pd.read_csv(full_path_a, index_col=0, nrows=0, sep=None, engine='python')
            genes_count = len(df_headers.columns)
            
            with open(full_path_a, 'rb') as f:
                samples_count = sum(1 for _ in f) - 1 # Subtract header
            
            return genes_count, samples_count
    except Exception as e:
        print(f"Error getting stats for {config_path}: {e}")
    return 0, 0

def migrate():
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_config.yaml"):
                full_path = os.path.join(root, file)
                print(f"Updating metadata for {full_path} (Strict Orientation)...")
                
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                pid = os.path.basename(root)
                genes, samples = get_project_stats(full_path)
                
                if 'metadata' not in config:
                    config['metadata'] = {}
                
                config['metadata'].update({
                    'name': config['metadata'].get('name', pid),
                    'description': config['metadata'].get('description', f"Dataset for {pid} project."),
                    'genes': genes,
                    'samples': samples
                })
                
                # Update core config to match detected orientation
                config['model']['input_nc'] = genes
                config['model']['output_nc'] = genes
                config['dataset']['max_samples'] = samples
                
                with open(full_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"  Done. Set genes={genes}, samples={samples}")

if __name__ == "__main__":
    migrate()
