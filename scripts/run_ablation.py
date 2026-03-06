import subprocess
import os
import argparse

def run_ablation():
    parser = argparse.ArgumentParser(description="Run GANomics Ablation Study (Sample Size)")
    parser.add_argument("--config", type=str, default="configs/nb_config.yaml", help="Base config file")
    parser.add_argument("--sizes", nargs='+', type=int, default=[10, 20, 30, 40, 50, 100, 200, 400], 
                        help="List of sample sizes to test")
    args = parser.parse_args()
    
    # We will use the existing scripts/train.py but override max_samples via a temporary config or command line
    # For simplicity, we'll modify train.py slightly to accept overrides or use separate configs.
    # Let's use a systematic naming for results.
    
    for size in args.sizes:
        name = f"NB_GANomics_{size}"
        print(f"\n{'='*20}")
        print(f"Starting Training: {name} (Size: {size})")
        print(f"{'='*20}")
        
        # We can pass these as arguments if we update scripts/train.py, 
        # or just use a shell command with a few flags.
        # Let's assume we update train.py to accept --max_samples and --name overrides.
        
        cmd = [
            "python", "scripts/train.py",
            "--config", args.config,
            "--max_samples", str(size),
            "--name", name
        ]
        
        # Set epochs based on sample size as per manuscript (fewer samples -> more epochs)
        if size <= 50:
            cmd += ["--n_epochs", "250", "--n_epochs_decay", "250"]
        elif size <= 100:
            cmd += ["--n_epochs", "100", "--n_epochs_decay", "100"]
        else:
            cmd += ["--n_epochs", "50", "--n_epochs_decay", "50"]
            
        subprocess.run(cmd)

if __name__ == "__main__":
    run_ablation()
