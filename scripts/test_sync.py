import os
import argparse
import yaml
import torch
import pandas as pd
from src.datasets.genomics_dataset import GenomicsDataset
from src.models.ganomics_model import GANomicsModel
from src.core.evaluation import benchmark_all_methods
from src.bio_utils import combat_evaluate_paired, quantile_normalize, tdm_normalize, yugene_evaluate_paired

def main():
    parser = argparse.ArgumentParser(description="Test GANomics Sync Data")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = GANomicsModel(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        device=device
    )
    model.load_networks(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")
    
    # 2. Load Test Data
    # Assuming test data is in separate files or split from training data
    test_dataset = GenomicsDataset(
        config['dataset']['path_A'], # In real usage, this should be path_test_A
        config['dataset']['path_B'], 
        is_train=False
    )
    
    # Generate Synthetic Data
    model.netG_A.eval()
    model.netG_B.eval()
    
    real_A = test_dataset.df_A.values
    real_B = test_dataset.df_B.values
    
    with torch.no_grad():
        tensor_A = torch.tensor(real_A, dtype=torch.float32).to(device)
        tensor_B = torch.tensor(real_B, dtype=torch.float32).to(device)
        
        # Generator outputs
        syn_B = model.netG_A(tensor_A).cpu().numpy().squeeze()
        syn_A = model.netG_B(tensor_B).cpu().numpy().squeeze()
        
    # 3. Evaluate Baselines (Dummy split for demonstration)
    # This would usually involve train/test splitting
    # ...
    
    # 4. Generate Tables 2 & 3
    results_df = benchmark_all_methods(real_A, real_B, syn_A, syn_B)
    
    os.makedirs("results/tables", exist_ok=True)
    results_df.to_csv("results/tables/Table_2_Benchmarks.csv", index=False)
    print("\nBenchmark results saved to results/tables/Table_2_Benchmarks.csv")
    print(results_df)

if __name__ == "__main__":
    main()
