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

def main():
    parser = argparse.ArgumentParser(description="Generate Sync Data for GANomics")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run_id", type=str, required=True, help="Full run ID (e.g. NB_Ablation_Size_50_Run_0)")
    parser.add_argument("--epoch", type=str, default="latest", help="Epoch checkpoint to load")
    args = parser.parse_args()

    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Checkpoint Path
    # Convention: results/1_Training/checkpoints/{run_id}/net_epoch_{epoch}.pth
    checkpoint_dir = resolve_path(os.path.join("results", "1_Training", "checkpoints", args.run_id))
    
    if args.epoch == "latest":
        checkpoint_path = os.path.join(checkpoint_dir, "net_latest.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"net_epoch_{args.epoch}.pth")
        
    if not os.path.exists(checkpoint_path):
        # Fallback to older convention if needed, but we should stick to the new one
        alt_path = resolve_path(os.path.join(config['output']['checkpoints_dir'], args.run_id, f"net_epoch_{args.epoch}.pth"))
        if os.path.exists(alt_path): checkpoint_path = alt_path
        else: raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        fake_B = model.netG_A(tensor_A).cpu().numpy().squeeze()
        fake_A = model.netG_B(tensor_B).cpu().numpy().squeeze()
        
    # 5. Save Sync Data
    # Convention: results/2_SyncData/{run_id}/
    sync_dir = resolve_path(os.path.join("results", "2_SyncData", args.run_id))
    
    # Automatic Train/Test Identification from audit trail
    train_samples_path = os.path.join(checkpoint_dir, "train_samples.txt")
    if os.path.exists(train_samples_path):
        with open(train_samples_path, 'r') as f:
            train_ids = [line.strip() for line in f]
        test_ids = [s for s in sample_names if s not in train_ids]
        print(f"Audit Trail Found: {len(train_ids)} train samples, {len(test_ids)} unseen test samples.")
    else:
        # If no audit trail, use default max_samples from config (dangerous but fallback)
        max_samples = config['dataset'].get('max_samples', 50)
        train_ids = sample_names[:max_samples].tolist()
        test_ids = sample_names[max_samples:].tolist()
        print(f"Warning: No audit trail found. Using first {max_samples} samples as train.")

    def save_subset(ids, subfolder):
        if not ids: return
        out_dir = os.path.join(sync_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        
        # Select rows using .loc and convert to float32 for consistency
        df_rA = pd.DataFrame(real_A, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_fA = pd.DataFrame(fake_A, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_rB = pd.DataFrame(real_B, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_fB = pd.DataFrame(fake_B, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        
        df_rA.to_csv(os.path.join(out_dir, "microarray_real.csv"))
        df_fA.to_csv(os.path.join(out_dir, "microarray_fake.csv"))
        df_rB.to_csv(os.path.join(out_dir, "rnaseq_real.csv"))
        df_fB.to_csv(os.path.join(out_dir, "rnaseq_fake.csv"))
        print(f"Saved {subfolder} sync data ({len(ids)} samples) to {out_dir}")

    save_subset(train_ids, "train")
    save_subset(test_ids, "test")

if __name__ == "__main__":
    main()
