import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.genomics_dataset import GenomicsDataset
from src.models.ganomics_model import GANomicsModel
from src.core.evaluation import benchmark_all_methods

def main():
    parser = argparse.ArgumentParser(description="Test GANomics and Generate Sync Data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--epoch", type=str, default="latest", help="Epoch checkpoint to load")
    parser.add_argument("--no_adjust_path", action='store_true', help="Don't adjust PYTHONPATH")
    parser.add_argument("--exp", type=str, default="GANomics", help="Experiment type, specific to checkpoints.")
    args = parser.parse_args()

    if not args.no_adjust_path:
        # Add the parent directory to sys.path to make 'src' importable
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Checkpoint Path
    # Convention: results/checkpoints/{Project}_GANomics_{sample_size}_{run_id}/net_epoch_{epoch}.pth
    exp_name = f"{args.exp}"
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
        is_train=False,
        force_index_mapping=config['dataset'].get('force_index_mapping', False)
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
    # Convention: results/2_SyncData/{exp_name}/
    sync_dir = os.path.join("results", "2_SyncData", f"{args.exp}")
    
    # Automatic Train/Test Identification
    train_samples_path = os.path.join(checkpoint_dir, "train_samples.txt")
    if os.path.exists(train_samples_path):
        with open(train_samples_path, 'r') as f:
            train_ids = [line.strip() for line in f]
        test_ids = [s for s in sample_names if s not in train_ids]
        print(f"Audit Trail Found: {len(train_ids)} train samples, {len(test_ids)} unseen test samples.")
    else:
        # Fallback to simple split if audit trail missing
        train_ids = sample_names[:args.sample_size].tolist()
        test_ids = sample_names[args.sample_size:].tolist()
        print(f"Warning: No audit trail found. Using default split at index {args.sample_size}.")

    def save_subset(ids, subfolder):
        if not ids: return
        out_dir = os.path.join(sync_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        
        pd.DataFrame(real_A, index=sample_names, columns=genes).loc[ids].to_csv(os.path.join(out_dir, "microarray_real.csv"))
        pd.DataFrame(fake_A, index=sample_names, columns=genes).loc[ids].to_csv(os.path.join(out_dir, "microarray_fake.csv"))
        pd.DataFrame(real_B, index=sample_names, columns=genes).loc[ids].to_csv(os.path.join(out_dir, "rnaseq_real.csv"))
        pd.DataFrame(fake_B, index=sample_names, columns=genes).loc[ids].to_csv(os.path.join(out_dir, "rnaseq_fake.csv"))
        print(f"Saved {subfolder} sync data ({len(ids)} samples) to {out_dir}")

    save_subset(train_ids, "train")
    save_subset(test_ids, "test")
    
    # 6. Optional: Quick benchmark on UNSEEN samples
    if test_ids:
        test_idx = [list(sample_names).index(i) for i in test_ids]
        results_df = benchmark_all_methods(real_A[test_idx], real_B[test_idx], fake_A[test_idx], fake_B[test_idx])
        print("\nBenchmarks on UNSEEN samples:")
        print(results_df)

if __name__ == "__main__":
    main()
