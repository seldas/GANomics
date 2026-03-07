import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from src.datasets.genomics_dataset import GenomicsDataset
from src.models.ganomics_model import GANomicsModel
from src.core.evaluation import benchmark_all_methods

def main():
    parser = argparse.ArgumentParser(description="Test GANomics and Generate Sync Data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--project", type=str, default="NB", help="Project name (e.g. NB)")
    parser.add_argument("--sample_size", type=int, default=400, help="Training sample size used")
    parser.add_argument("--run_id", type=int, default=0, help="Run/Repetition ID")
    parser.add_argument("--epoch", type=str, default="latest", help="Epoch checkpoint to load")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Checkpoint Path
    # Convention: results/checkpoints/{Project}_GANomics_{sample_size}_{run_id}/net_epoch_{epoch}.pth
    exp_name = f"{args.project}_GANomics_{args.sample_size}_{args.run_id}"
    checkpoint_dir = os.path.join(config['output']['checkpoints_dir'], exp_name)
    
    if args.epoch == "latest":
        checkpoint_path = os.path.join(checkpoint_dir, "net_latest.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"net_epoch_{args.epoch}.pth")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2. Load Model
    model = GANomicsModel(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        device=device
    )
    model.load_networks(checkpoint_path)
    model.netG_A.eval()
    model.netG_B.eval()
    print(f"Loaded model from {checkpoint_path}")
    
    # 3. Load Dataset
    # We use the full dataset but the script will generate synthetic versions for all
    dataset = GenomicsDataset(
        config['dataset']['path_A'], 
        config['dataset']['path_B'], 
        is_train=False
    )
    
    genes = dataset.df_A.columns
    sample_names = dataset.df_A.index
    
    real_A = dataset.df_A.values
    real_B = dataset.df_B.values
    
    # 4. Generate Synthetic Data
    with torch.no_grad():
        tensor_A = torch.tensor(real_A, dtype=torch.float32).to(device)
        tensor_B = torch.tensor(real_B, dtype=torch.float32).to(device)
        
        # Generator outputs
        # Note: Model expects (B, C, 1, 1), model.netG_A handles 2D to 4D internally if implemented
        fake_B = model.netG_A(tensor_A).cpu().numpy().squeeze()
        fake_A = model.netG_B(tensor_B).cpu().numpy().squeeze()
        
    # 5. Save Sync Data
    # Convention: results/sync_data/{Project}_{sample_size}_{run_id}/
    sync_dir = os.path.join("results", "sync_data", f"{args.project}_{args.sample_size}_{args.run_id}")
    os.makedirs(sync_dir, exist_ok=True)
    
    df_real_A = pd.DataFrame(real_A, index=sample_names, columns=genes)
    df_fake_A = pd.DataFrame(fake_A, index=sample_names, columns=genes)
    df_real_B = pd.DataFrame(real_B, index=sample_names, columns=genes)
    df_fake_B = pd.DataFrame(fake_B, index=sample_names, columns=genes)
    
    df_real_A.to_csv(os.path.join(sync_dir, "microarray_real.csv"))
    df_fake_A.to_csv(os.path.join(sync_dir, "microarray_fake.csv"))
    df_real_B.to_csv(os.path.join(sync_dir, "rnaseq_real.csv"))
    df_fake_B.to_csv(os.path.join(sync_dir, "rnaseq_fake.csv"))
    
    print(f"Sync data saved to {sync_dir}")
    
    # 6. Optional: Quick benchmark print
    results_df = benchmark_all_methods(real_A, real_B, fake_A, fake_B)
    print("\nSample Benchmarks:")
    print(results_df)

if __name__ == "__main__":
    main()
