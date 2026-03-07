import subprocess
import os
import argparse
import sys

def run_ablation():
    parser = argparse.ArgumentParser(description="Run GANomics Ablation & Sensitivity Study")
    parser.add_argument("--config", type=str, default="configs/nb_config.yaml", help="Base config file")
    parser.add_argument("--sizes", nargs='+', type=int, help="Sample sizes to test (e.g. 10 50 100)")
    parser.add_argument("--betas", nargs='+', type=float, help="Feedback weights (beta) to test (e.g. 1.0 10.0 50.0)")
    parser.add_argument("--lambdas", nargs='+', type=float, help="Cycle weights (lambda) to test (e.g. 1.0 10.0 50.0)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent trials per setting")
    args = parser.parse_args()
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    def execute_repeats(base_cmd, base_name):
        for r in range(args.repeats):
            run_name = f"{base_name}_Run_{r}"
            cmd = [python_exe] + base_cmd[1:] + ["--name", run_name, "--seed", str(42 + r)]
            print(f"\n>>> Executing Trial {r+1}/{args.repeats}: {run_name}")
            subprocess.run(cmd, env=env)

    # 1. Sample Size Study
    if args.sizes:
        for size in args.sizes:
            base_name = f"NB_Ablation_Size_{size}"
            cmd = [
                "python", "scripts/train.py",
                "--config", args.config,
                "--max_samples", str(size)
            ]
            if size <= 50: cmd += ["--n_epochs", "250", "--n_epochs_decay", "250"]
            else: cmd += ["--n_epochs", "50", "--n_epochs_decay", "50"]
            
            execute_repeats(cmd, base_name)

    # 2. Beta (Feedback) Sensitivity Study
    if args.betas:
        fixed_size = 50 
        for beta in args.betas:
            base_name = f"NB_Sensitivity_Beta_{beta}"
            cmd = [
                "python", "scripts/train.py",
                "--config", args.config,
                "--max_samples", str(fixed_size),
                "--lambda_feedback", str(beta),
                "--n_epochs", "100", "--n_epochs_decay", "100"
            ]
            execute_repeats(cmd, base_name)

    # 3. Lambda (Cycle) Sensitivity Study
    if args.lambdas:
        fixed_size = 50
        for lam in args.lambdas:
            base_name = f"NB_Sensitivity_Lambda_{lam}"
            cmd = [
                "python", "scripts/train.py",
                "--config", args.config,
                "--max_samples", str(fixed_size),
                "--lambda_cycle", str(lam),
                "--n_epochs", "100", "--n_epochs_decay", "100"
            ]
            execute_repeats(cmd, base_name)

    # 4. Architecture Ablation
    if args.config:
        fixed_size = 50
        for direction in ['AtoB', 'BtoA']:
            base_name = f"NB_Ablation_Architecture_{direction}"
            cmd = [
                "python", "scripts/train.py",
                "--config", args.config,
                "--max_samples", str(fixed_size),
                "--direction", direction,
                "--n_epochs", "100", "--n_epochs_decay", "100"
            ]
            execute_repeats(cmd, base_name)

if __name__ == "__main__":
    run_ablation()
