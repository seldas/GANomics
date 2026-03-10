import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.ganomics_model import GANomicsModel

def main():
    parser = argparse.ArgumentParser(description="Run Inference on External Data")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to use for inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input TSV file")
    parser.add_argument("--direction", type=str, required=True, choices=['AtoB', 'BtoA'], help="Inference direction")
    parser.add_argument("--output", type=str, required=True, help="Path to save output TSV")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda:0)")
    args = parser.parse_args()

    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(backend_dir, "results", "1_Training", "checkpoints", args.run_id)
    checkpoint_path = os.path.join(results_dir, "net_latest.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # 1. Load Expected Gene List for Validation
    genes_path = os.path.join(results_dir, "genes.txt")
    if not os.path.exists(genes_path):
        print(f"Error: Gene list template (genes.txt) not found in {results_dir}. Cannot validate alignment.")
        sys.exit(1)
    
    with open(genes_path, 'r') as f:
        expected_genes = [line.strip() for line in f if line.strip()]

    # 2. Load Input Data
    try:
        df_input = pd.read_csv(args.input, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Validate orientation and genes
    input_genes = df_input.columns.tolist()
    
    if len(input_genes) != len(expected_genes):
        print(f"Error: Dimension mismatch. Input has {len(input_genes)} genes, model expects {len(expected_genes)}.")
        sys.exit(1)
    
    if input_genes != expected_genes:
        # Check if they are the same set but different order
        if set(input_genes) == set(expected_genes):
            print("Warning: Gene order differs. Reordering input features to match model expectations.")
            df_input = df_input[expected_genes]
        else:
            missing = set(expected_genes) - set(input_genes)
            extra = set(input_genes) - set(expected_genes)
            print(f"Error: Gene list mismatch.")
            if missing: print(f"Missing genes: {list(missing)[:10]}...")
            if extra: print(f"Extra genes: {list(extra)[:10]}...")
            sys.exit(1)

    # 3. Setup Model
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    input_nc = len(expected_genes)
    
    model = GANomicsModel(
        input_nc=input_nc,
        output_nc=input_nc,
        device=device
    )
    model.load_networks(checkpoint_path)
    model.eval()

    # 4. Run Inference
    input_tensor = torch.from_numpy(df_input.values).float().to(device)
    
    with torch.no_grad():
        if args.direction == 'AtoB':
            # Generator A converts Microarray to RNA-Seq
            output_tensor = model.netG_A(input_tensor)
        else:
            # Generator B converts RNA-Seq to Microarray
            output_tensor = model.netG_B(input_tensor)
            
    # 5. Save Results
    # Squeeze the 4D output (batch, genes, 1, 1) back to 2D (batch, genes)
    output_np = output_tensor.cpu().numpy()
    if output_np.ndim == 4:
        output_np = output_np.squeeze(-1).squeeze(-1)
        
    df_output = pd.DataFrame(
        output_np,
        index=df_input.index,
        columns=expected_genes
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_output.to_csv(args.output, sep='\t')
    print(f"Success: Inference completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
