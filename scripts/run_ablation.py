import subprocess
import os
import argparse
import sys
from multiprocessing import Pool, Manager

def run_trial(args_tuple):
    cmd, run_name, device_str, env = args_tuple
    # Override device for this specific run
    cmd = cmd + ["--device", device_str]
    print(f"\n>>> [{device_str}] Starting: {run_name}")
    subprocess.run(cmd, env=env)
    print(f">>> [{device_str}] Finished: {run_name}")

def run_ablation():
    parser = argparse.ArgumentParser(description="Run GANomics Ablation & Sensitivity Study")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--sizes", nargs='+', type=int, help="Sample sizes to test")
    parser.add_argument("--betas", nargs='+', type=float, help="Feedback weights (beta)")
    parser.add_argument("--lambdas", nargs='+', type=float, help="Cycle weights (lambda)")
    parser.add_argument("--direction", type=str, choices=['AtoB', 'BtoA'], help="One-directional mode")
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent trials per setting")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated GPU IDs (e.g. 0,1,2,3). If omitted, uses CPU.")
    args = parser.parse_args()
    
    # Extract Project Name from Config File (e.g., 'nb' from 'configs/nb_config.yaml')
    project_name = os.path.basename(args.config).replace('_config.yaml', '').upper()
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    
    # Device and Worker Setup
    import torch
    if args.gpu_ids and torch.cuda.is_available():
        gpu_list = [int(x) for x in args.gpu_ids.split(',')]
        devices = [f"cuda:{x}" for x in gpu_list]
        n_workers = len(gpu_list)
    else:
        devices = ["cpu"]
        # On CPU, we default to 1 worker to avoid over-subscription, 
        # unless user manually wants to parallelize on cores (not recommended for GANs)
        n_workers = 1
    
    tasks = []

    def queue_repeats(base_cmd, base_name):
        for r in range(args.repeats):
            run_name = f"{base_name}_Run_{r}"
            cmd = [python_exe] + base_cmd + ["--name", run_name, "--seed", str(42 + r)]
            tasks.append((cmd, run_name))

    # 1. Sample Size Study
    if args.sizes:
        for size in args.sizes:
            base_name = f"{project_name}_Ablation_Size_{size}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", str(size)]
            if size <= 50: cmd += ["--n_epochs", "250", "--n_epochs_decay", "250"]
            else: cmd += ["--n_epochs", "50", "--n_epochs_decay", "50"]
            queue_repeats(cmd, base_name)

    # 2. Beta Sensitivity
    if args.betas:
        for beta in args.betas:
            base_name = f"{project_name}_Sensitivity_Beta_{beta}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--lambda_feedback", str(beta)]
            queue_repeats(cmd, base_name)

    # 3. Lambda Sensitivity
    if args.lambdas:
        for lam in args.lambdas:
            base_name = f"{project_name}_Sensitivity_Lambda_{lam}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--lambda_cycle", str(lam)]
            queue_repeats(cmd, base_name)

    # 4. Architecture Ablation
    if args.config and not (args.sizes or args.betas or args.lambdas):
        for direction in ['AtoB', 'BtoA']:
            base_name = f"{project_name}_Ablation_Architecture_{direction}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--direction", direction]
            queue_repeats(cmd, base_name)

    # Execute Tasks in Parallel
    if tasks:
        print(f"\n🚀 Launching {len(tasks)} tasks across {n_workers} workers: {devices}")
        # Assign tasks to devices in a round-robin fashion
        worker_args = []
        for i, (cmd, name) in enumerate(tasks):
            device_str = devices[i % n_workers]
            worker_args.append((cmd, name, device_str, env))
            
        with Pool(processes=n_workers) as pool:
            pool.map(run_trial, worker_args)
        print("\n✅ All parallel trials completed.")

if __name__ == "__main__":
    run_ablation()

if __name__ == "__main__":
    run_ablation()
