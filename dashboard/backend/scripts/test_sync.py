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
from src.core.analysis import run_tsne_analysis

def main():
    parser = argparse.ArgumentParser(description="Generate Sync Data for GANomics")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run_id", type=str, required=True, help="Full run ID (e.g. NB_Ablation_Size_50_Run_0)")
    parser.add_argument("--epoch", type=str, default="latest", help="Epoch checkpoint to load")
    parser.add_argument("--ext_id", type=str, help="External Dataset ID (e.g. ext_1)")
    args = parser.parse_args()

    # Determine backend directory
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    def resolve_path(p):
        return os.path.join(backend_dir, p) if not os.path.isabs(p) else p

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Checkpoint Path
    checkpoint_dir = resolve_path(os.path.join("results", "1_Training", "checkpoints", args.run_id))
    checkpoint_path = os.path.join(checkpoint_dir, "net_latest.pth" if args.epoch == "latest" else f"net_epoch_{args.epoch}.pth")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2. Load Model
    model = GANomicsModel(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        device=device
    )
    model.load_networks(checkpoint_path)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    
    # 3. Load and Generate Data
    sync_root = resolve_path(os.path.join("results", "2_SyncData", args.run_id))
    
    if args.ext_id:
        project_id = args.run_id.split('_')[0]
        ext_dir = os.path.join(backend_dir, "dataset", project_id, args.ext_id)
        path_ag = os.path.join(ext_dir, "test_ag.tsv")
        path_rs = os.path.join(ext_dir, "test_rs.tsv")
        
        # Load whatever is available
        df_A = pd.read_csv(path_ag, sep='\t', index_col=0) if os.path.exists(path_ag) else None
        df_B = pd.read_csv(path_rs, sep='\t', index_col=0) if os.path.exists(path_rs) else None
        
        # Ensure they have same genes as training
        genes_path = os.path.join(checkpoint_dir, "genes.txt")
        if os.path.exists(genes_path):
            with open(genes_path, 'r') as f: expected_genes = [l.strip() for l in f]
            if df_A is not None: df_A = df_A.reindex(columns=expected_genes).fillna(0)
            if df_B is not None: df_B = df_B.reindex(columns=expected_genes).fillna(0)
        
        fake_A, fake_B = None, None
        if df_A is not None:
            tensor_A = torch.tensor(df_A.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                fake_B_tensor = model.netG_A(tensor_A).cpu().numpy()
                fake_B = fake_B_tensor.squeeze(-1).squeeze(-1) if fake_B_tensor.ndim == 4 else fake_B_tensor.squeeze()
                if fake_B.ndim == 1: fake_B = fake_B[np.newaxis, :]
            
        if df_B is not None:
            tensor_B = torch.tensor(df_B.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                fake_A_tensor = model.netG_B(tensor_B).cpu().numpy()
                fake_A = fake_A_tensor.squeeze(-1).squeeze(-1) if fake_A_tensor.ndim == 4 else fake_A_tensor.squeeze()
                if fake_A.ndim == 1: fake_A = fake_A[np.newaxis, :]
            
        out_dir = os.path.join(sync_root, args.ext_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save CSVs
        df_ma_real = df_A
        df_ma_fake = pd.DataFrame(fake_A, index=df_B.index, columns=df_B.columns) if fake_A is not None else None
        df_rs_real = df_B
        df_rs_fake = pd.DataFrame(fake_B, index=df_A.index, columns=df_A.columns) if fake_B is not None else None
        
        if df_ma_real is not None: df_ma_real.to_csv(os.path.join(out_dir, "microarray_real.csv"))
        if df_ma_fake is not None: df_ma_fake.to_csv(os.path.join(out_dir, "microarray_fake.csv"))
        if df_rs_real is not None: df_rs_real.to_csv(os.path.join(out_dir, "rnaseq_real.csv"))
        if df_rs_fake is not None: df_rs_fake.to_csv(os.path.join(out_dir, "rnaseq_fake.csv"))
        
        # Calculate t-SNE
        print("Generating t-SNE visualization coordinates...")
        df_tsne = run_tsne_analysis(df_ma_real, df_ma_fake, df_rs_real, df_rs_fake)
        df_tsne.to_csv(os.path.join(out_dir, "tsne_coords.csv"), index=False)
        
        print(f"Saved external sync data and t-SNE for ({args.ext_id}) to {out_dir}")
        return

    # Standard Internal Test Logic
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
    
    with torch.no_grad():
        tensor_A = torch.tensor(real_A, dtype=torch.float32).to(device)
        tensor_B = torch.tensor(real_B, dtype=torch.float32).to(device)
        fake_B_tensor = model.netG_A(tensor_A).cpu().numpy()
        fake_A_tensor = model.netG_B(tensor_B).cpu().numpy()
        fake_B = fake_B_tensor.squeeze(-1).squeeze(-1) if fake_B_tensor.ndim == 4 else fake_B_tensor.squeeze()
        fake_A = fake_A_tensor.squeeze(-1).squeeze(-1) if fake_A_tensor.ndim == 4 else fake_A_tensor.squeeze()
        
    train_samples_path = os.path.join(checkpoint_dir, "train_samples.txt")
    if os.path.exists(train_samples_path):
        with open(train_samples_path, 'r') as f: train_ids = [line.strip() for line in f]
        test_ids = [s for s in sample_names if s not in train_ids]
    else:
        max_samples = config['dataset'].get('max_samples', 50)
        train_ids = sample_names[:max_samples].tolist()
        test_ids = sample_names[max_samples:].tolist()

    def save_subset(ids, subfolder):
        if not ids: return
        out_dir = os.path.join(sync_root, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        df_rA = pd.DataFrame(real_A, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_fA = pd.DataFrame(fake_A, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_rB = pd.DataFrame(real_B, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_fB = pd.DataFrame(fake_B, index=sample_names, columns=genes).loc[ids].astype(np.float32)
        df_rA.to_csv(os.path.join(out_dir, "microarray_real.csv"))
        df_fA.to_csv(os.path.join(out_dir, "microarray_fake.csv"))
        df_rB.to_csv(os.path.join(out_dir, "rnaseq_real.csv"))
        df_fB.to_csv(os.path.join(out_dir, "rnaseq_fake.csv"))
        
        # Calculate t-SNE for test set
        if subfolder == "test":
            print(f"Generating t-SNE coordinates for {subfolder} set...")
            df_tsne = run_tsne_analysis(df_rA, df_fA, df_rB, df_fB)
            df_tsne.to_csv(os.path.join(out_dir, "tsne_coords.csv"), index=False)

    save_subset(train_ids, "train")
    save_subset(test_ids, "test")
    print("✅ Test data synchronization completed.")

if __name__ == "__main__":
    main()
