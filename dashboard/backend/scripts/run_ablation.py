import subprocess
import os
import argparse
import sys
import time
from multiprocessing import Pool, Manager

def update_ongoing_tasks(run_id, action='add'):
    # File is in results/1_Training/ongoing_tasks.txt
    # We are in scripts/, so ../results/1_Training/ongoing_tasks.txt
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_file = os.path.join(base_dir, "results", "1_Training", "ongoing_tasks.txt")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(task_file), exist_ok=True)
    
    # Use simple file locking/retries if needed, but for now just atomic-ish read/write
    tasks = []
    if os.path.exists(task_file):
        with open(task_file, 'r') as f:
            tasks = [line.strip() for line in f if line.strip()]
            
    if action == 'add':
        if run_id not in tasks:
            tasks.append(run_id)
    else: # remove
        if run_id in tasks:
            tasks.remove(run_id)
            
    with open(task_file, 'w') as f:
        for t in tasks:
            f.write(f"{t}\n")

def run_trial(args_tuple):
    cmd, run_name, device_str, env, progress_dict = args_tuple
    
    # Track as ongoing
    update_ongoing_tasks(run_name, 'add')
    
    # Override device for this specific run and add quiet mode
    cmd = cmd + ["--device", device_str, "--quiet"]
    
    progress_dict[run_name] = f"Starting on {device_str}..."
    
    # Run the process and capture output line by line
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    full_output = []
    error_captured = False
    
    for line in process.stdout:
        line = line.strip()
        full_output.append(line)
        if line.startswith("[PROGRESS]"):
            progress_dict[run_name] = f"Epoch: {line.replace('[PROGRESS] ', '')}"
        elif "Traceback" in line or "Error" in line or "Exception" in line:
            error_captured = True

    process.wait()
    
    # If completed successfully, remove from ongoing
    if process.returncode == 0:
        update_ongoing_tasks(run_name, 'remove')
        progress_dict[run_name] = "Completed"
    else:
        # Capture the last few lines of output as the error context if generic error string used
        error_context = "\n".join(full_output[-10:])
        progress_dict[run_name] = f"Error: {error_context}" if error_captured or process.returncode != 0 else "Failed"

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
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")
    
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
    task_features = {}

    def queue_repeats(base_cmd, base_name, features_str):
        for r in range(args.repeats):
            run_name = f"{base_name}_Run_{r}"
            cmd = [python_exe] + base_cmd + ["--name", run_name, "--seed", str(42 + r)]
            tasks.append((cmd, run_name))
            task_features[run_name] = f"{features_str} | Repeat: {r}"

    # Define path to train.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")

    # 1. Sample Size Study
    if args.sizes:
        for size in args.sizes:
            base_name = f"{project_name}_Ablation_Size_{size}"
            cmd = [train_script, "--config", args.config, "--max_samples", str(size)]
            if size <= 50: cmd += ["--n_epochs", "250", "--n_epochs_decay", "250"]
            else: cmd += ["--n_epochs", "50", "--n_epochs_decay", "50"]
            queue_repeats(cmd, base_name, f"Size: {size}")

    # 2. Beta Sensitivity
    if args.betas:
        for beta in args.betas:
            # Skip if beta is default (10.0) and size 50 is already in ablation list
            if beta == 10.0 and args.sizes and 50 in args.sizes:
                print(f"ℹ️ Skipping Beta={beta} sensitivity (covered by Size=50 ablation)")
                continue
            base_name = f"{project_name}_Sensitivity_Beta_{beta}"
            cmd = [train_script, "--config", args.config, "--max_samples", "50", "--n_epochs", "250", "--n_epochs_decay", "250", "--lambda_feedback", str(beta)]
            queue_repeats(cmd, base_name, f"Size: 50 | Beta: {beta}")

    # 3. Lambda Sensitivity
    if args.lambdas:
        for lam in args.lambdas:
            # Skip if lambda is default (10.0) and size 50 is already in ablation list
            if lam == 10.0 and args.sizes and 50 in args.sizes:
                print(f"ℹ️ Skipping Lambda={lam} sensitivity (covered by Size=50 ablation)")
                continue
            base_name = f"{project_name}_Sensitivity_Lambda_{lam}"
            cmd = [train_script, "--config", args.config, "--max_samples", "50", "--n_epochs", "250", "--n_epochs_decay", "250", "--lambda_cycle", str(lam)]
            queue_repeats(cmd, base_name, f"Size: 50 | Lambda: {lam}")

    # 4. Architecture Ablation
    if args.config and not (args.sizes or args.betas or args.lambdas):
        for direction in ['AtoB', 'BtoA']:
            base_name = f"{project_name}_Ablation_Architecture_{direction}"
            cmd = [train_script, "--config", args.config, "--max_samples", "50", "--direction", direction]
            queue_repeats(cmd, base_name, f"Size: 50 | Direction: {direction}")

    # Execute Tasks in Parallel
    if tasks:
        print(f"\n🚀 Launching {len(tasks)} tasks across {n_workers} workers: {devices}\n")
        
        manager = Manager()
        progress_dict = manager.dict()
        
        # Assign tasks to devices in a round-robin fashion
        worker_args = []
        for i, (cmd, name) in enumerate(tasks):
            device_str = devices[i % n_workers]
            worker_args.append((cmd, name, device_str, env, progress_dict))
            
        with Pool(processes=n_workers) as pool:
            result = pool.map_async(run_trial, worker_args)
            
            num_lines_printed = 0
            while not result.ready():
                active_tasks = {k: v for k, v in progress_dict.items() if v not in ("Completed", "Failed", "Error")}
                
                # Move cursor up to overwrite previous lines, then clear to end of screen
                if num_lines_printed > 0:
                    sys.stdout.write(f"\033[{num_lines_printed}A")
                sys.stdout.write("\033[J")
                
                lines_to_print = []
                for name, status in active_tasks.items():
                    features = task_features.get(name, "")
                    lines_to_print.append(f"\033[K🟢 {name} [{features}] => {status}")
                
                for line in lines_to_print:
                    sys.stdout.write(line + "\n")
                
                sys.stdout.flush()
                num_lines_printed = len(lines_to_print)
                time.sleep(0.5)
                
        print("\n✅ All parallel trials completed.")
        for k, v in progress_dict.items():
            if v in ("Failed", "Error"):
                print(f"⚠️ {k} finished with status: {v}")

if __name__ == "__main__":
    run_ablation()
