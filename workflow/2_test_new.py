import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add backend to path to import GANomics modules
# Current file: plan/test_sync.py
# Backend dir: dashboard/backend
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
backend_dir = os.path.join(root_dir, 'dashboard', 'backend')
sys.path.insert(0, backend_dir)
from src.models.ganomics_model import GANomicsModel
# from old_ver.model.test_model import TestModel as GANomicsModel

def run_ms_sync():
    ms_training_dir = os.path.join(backend_dir, "results_ms", "1_Training")
    checkpoint_root = os.path.join(ms_training_dir, "checkpoints")
    sync_root = os.path.join(backend_dir, "results_ms", "2_SyncData")
    dataset_root = os.path.join(backend_dir, "dataset")

    if not os.path.exists(checkpoint_root):
        print(f"Error: Manuscript checkpoint folder not found at {checkpoint_root}")
        return

    # Filter for directories that contain .pth files
    run_ids = []
    for d in os.listdir(checkpoint_root):
        d_path = os.path.join(checkpoint_root, d)
        if os.path.isdir(d_path) and os.path.exists(os.path.join(d_path, "net_latest.pth")):
            run_ids.append(d)
            
    print(f"Found {len(run_ids)} tasks to process.")

    for run_id in tqdm(run_ids, desc="Processing Tasks"):
        ckp_dir = os.path.join(checkpoint_root, run_id)
        exist_check = os.path.join(sync_root, run_id, 'test/microarray_fake.csv')
        if os.path.exists(exist_check): 
            print(run_id, 'has already been processed, skip ...')
            continue # skip existed 
        
        # 1. Determine Project and Load Dataset
        project_id = run_id.split('_')[0]
        # Handle special project naming
        if project_id == "CycleGAN": project_id = "NB"
        if "NB_Size" in run_id: project_id = "NB"
        
        proj_ds_dir = os.path.join(dataset_root, project_id)
        path_a = os.path.join(proj_ds_dir, "df_ag.tsv")
        path_b = os.path.join(proj_ds_dir, "df_rs.tsv")
        
        if not os.path.exists(path_a):
            print(f"  ⚠️ Skipping {run_id}: Project dataset {project_id} not found at {path_a}")
            continue
        
        try:
            df_a_full = pd.read_csv(path_a, sep='\t', index_col=0)
            df_b_full = pd.read_csv(path_b, sep='\t', index_col=0)
        except Exception as e:
            print(f"  ❌ Error reading dataset for {run_id}: {e}")
            continue

        # 2. Split into Train/Test based on archived samples
        train_samples_path = os.path.join(ckp_dir, "train_samples.txt")
        if not os.path.exists(train_samples_path):
            print(f"  ⚠️ Skipping {run_id}: train_samples.txt missing")
            continue
            
        with open(train_samples_path, 'r') as f:
            train_indices = [l.strip() for l in f if l.strip()]
        
        # Match index type (handles numerical IDs vs strings)
        if len(df_a_full.index) > 0 and not isinstance(df_a_full.index[0], str):
            try:
                train_indices = [type(df_a_full.index[0])(i) for i in train_indices]
            except: pass

        # 3. Setup Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_nc = len(df_a_full.columns)
        
        try:
            # Initialize model with appropriate configuration
            model = GANomicsModel(input_nc=input_nc, output_nc=input_nc, device=device)
            
            # Load checkpoint with all components
            checkpoint_path = os.path.join(ckp_dir, "net_latest.pth")
            if not os.path.exists(checkpoint_path):
                print(f"  ⚠️ Skipping {run_id}: net_latest.pth not found")
                continue
                
            model.load_networks(checkpoint_path)
            model.eval()
            
        except Exception as e:
            print(f"  ❌ Error loading model for {run_id}: {e}")
            continue

        # 4. Process both Train and Test splits
        for split_name in ['train', 'test']:
            if split_name == 'train':
                # Samples used during training
                idx = [i for i in train_indices if i in df_a_full.index]
            else:
                # Samples NOT in the training list
                idx = df_a_full.index.difference(train_indices)
            
            if len(idx) == 0: continue
            
            real_a = df_a_full.loc[idx]
            real_b = df_b_full.loc[idx]
            
            with torch.no_grad():
                tensor_a = torch.from_numpy(real_a.values).float().to(device)
                tensor_b = torch.from_numpy(real_b.values).float().to(device)
                
                # A -> B (Microarray -> RNAseq)
                fake_b_tensor = model.netG_A(tensor_a).detach()
                # B -> A (RNAseq -> Microarray)
                fake_a_tensor = model.netG_B(tensor_b).detach()
                
                # Squeeze handling for batch_size=1 vs batch_size>1
                fb_np = fake_b_tensor.cpu().numpy()
                if fb_np.ndim == 4: fb_np = fb_np.squeeze(-1).squeeze(-1)
                fa_np = fake_a_tensor.cpu().numpy()
                if fa_np.ndim == 4: fa_np = fa_np.squeeze(-1).squeeze(-1)
                
                fake_b = pd.DataFrame(fb_np, index=idx, columns=real_a.columns)
                fake_a = pd.DataFrame(fa_np, index=idx, columns=real_a.columns)

            # 5. Save Results
            out_dir = os.path.join(sync_root, run_id, split_name)
            os.makedirs(out_dir, exist_ok=True)
            real_a.to_csv(os.path.join(out_dir, "microarray_real.csv"))
            fake_a.to_csv(os.path.join(out_dir, "microarray_fake.csv"))
            real_b.to_csv(os.path.join(out_dir, "rnaseq_real.csv"))
            fake_b.to_csv(os.path.join(out_dir, "rnaseq_fake.csv"))

if __name__ == "__main__":
    run_ms_sync()
