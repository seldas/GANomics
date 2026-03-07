import subprocess
import os
import argparse
import sys
from multiprocessing import Pool, Manager

def run_trial(args_tuple):
    cmd, run_name, gpu_id, env = args_tuple
    # Override device for this specific run
    cmd = cmd + ["--device", f"cuda:{gpu_id}"]
    print(f"\n>>> [GPU {gpu_id}] Starting: {run_name}")
    subprocess.run(cmd, env=env)
    print(f">>> [GPU {gpu_id}] Finished: {run_name}")

def run_ablation():
    parser = argparse.ArgumentParser(description="Run GANomics Ablation & Sensitivity Study")
    parser.add_argument("--config", type=str, default="configs/nb_config.yaml", help="Base config file")
    parser.add_argument("--sizes", nargs='+', type=int, help="Sample sizes to test")
    parser.add_argument("--betas", nargs='+', type=float, help="Feedback weights (beta)")
    parser.add_argument("--lambdas", nargs='+', type=float, help="Cycle weights (lambda)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent trials per setting")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    args = parser.parse_args()
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    
    gpu_list = [int(x) for x in args.gpu_ids.split(',')]
    n_workers = len(gpu_list)
    
    tasks = []

    def queue_repeats(base_cmd, base_name):
        for r in range(args.repeats):
            run_name = f"{base_name}_Run_{r}"
            cmd = [python_exe] + base_cmd + ["--name", run_name, "--seed", str(42 + r)]
            tasks.append((cmd, run_name))

    # 1. Sample Size Study
    if args.sizes:
        for size in args.sizes:
            base_name = f"NB_Ablation_Size_{size}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", str(size)]
            if size <= 50: cmd += ["--n_epochs", "250", "--n_epochs_decay", "250"]
            else: cmd += ["--n_epochs", "50", "--n_epochs_decay", "50"]
            queue_repeats(cmd, base_name)

    # 2. Beta Sensitivity
    if args.betas:
        for beta in args.betas:
            base_name = f"NB_Sensitivity_Beta_{beta}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--lambda_feedback", str(beta)]
            queue_repeats(cmd, base_name)

    # 3. Lambda Sensitivity
    if args.lambdas:
        for lam in args.lambdas:
            base_name = f"NB_Sensitivity_Lambda_{lam}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--lambda_cycle", str(lam)]
            queue_repeats(cmd, base_name)

    # 4. Architecture Ablation
    if args.config and not (args.sizes or args.betas or args.lambdas):
        for direction in ['AtoB', 'BtoA']:
            base_name = f"NB_Ablation_Architecture_{direction}"
            cmd = ["scripts/train.py", "--config", args.config, "--max_samples", "50", "--direction", direction]
            queue_repeats(cmd, base_name)

    # Execute Tasks in Parallel
    if tasks:
        print(f"\n🚀 Launching {len(tasks)} tasks across {n_workers} GPUs: {gpu_list}")
        # Assign tasks to GPUs in a round-robin fashion
        worker_args = []
        for i, (cmd, name) in enumerate(tasks):
            gpu_id = gpu_list[i % n_workers]
            worker_args.append((cmd, name, gpu_id, env))
            
        with Pool(processes=n_workers) as pool:
            pool.map(run_trial, worker_args)
        print("\n✅ All parallel trials completed.")

if __name__ == "__main__":
    run_ablation()

if __name__ == "__main__":
    run_ablation()
