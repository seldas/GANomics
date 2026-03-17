"""Offline workflow driver for GANomics training runs."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "dashboard" / "backend"
SCRIPT_TRAIN = BACKEND_DIR / "scripts" / "train.py"


def locate_config(dataset_dir: Path) -> Path:
    candidates = list(dataset_dir.glob("*_config.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No config file ending with _config.yaml found in {dataset_dir}")
    return candidates[0]


def prepare_config(base_config: Path, dataset_dir: Path, run_name: str) -> Path:
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    df_ag = dataset_dir / "df_ag.tsv"
    df_rs = dataset_dir / "df_rs.tsv"
    if not df_ag.exists() or not df_rs.exists():
        raise FileNotFoundError("Expected df_ag.tsv and df_rs.tsv inside the dataset directory")

    config["dataset"]["path_A"] = str(df_ag.resolve())
    config["dataset"]["path_B"] = str(df_rs.resolve())
    config["dataset"]["force_index_mapping"] = True

    config["output"]["name"] = run_name
    config["output"]["checkpoints_dir"] = "results_ms/1_Training/checkpoints"
    config["output"]["logs_dir"] = "results_ms/1_Training/logs"

    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_config.yaml")
    yaml.safe_dump(config, temp)
    temp.flush()
    temp.close()
    return Path(temp.name)


def parse_gpu_ids(gpu_ids_arg: str | None) -> list[int]:
    if not gpu_ids_arg:
        return []
    return [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]


def build_command(
    config_path: Path,
    epochs: int | None,
    samples: int | None,
    beta: float | None,
    lam: float | None,
    direction: str,
    seed: int | None = None,
    device: str | None = None,
):
    cmd = [sys.executable, str(SCRIPT_TRAIN), "--config", str(config_path)]

    if epochs:
        half = max(1, epochs // 2)
        cmd += ["--n_epochs", str(half), "--n_epochs_decay", str(half)]
    if samples:
        cmd += ["--max_samples", str(samples)]
    if beta is not None:
        cmd += ["--lambda_feedback", str(beta)]
    if lam is not None:
        cmd += ["--lambda_cycle", str(lam)]
    if direction:
        cmd += ["--direction", direction]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if device is not None:
        cmd += ["--device", device]

    cmd += ["--quiet"]
    return cmd


def make_run_name(base_name: str, repeat_idx: int, repeats: int) -> str:
    if repeats <= 1:
        return base_name
    return f"{base_name}_{repeat_idx}"


def launch_batch(batch_jobs):
    """
    Launch a batch of jobs in parallel and wait for all to finish.
    batch_jobs: list of dicts with keys run_name, cmd, config_path, device, seed
    """
    processes = []

    for job in batch_jobs:
        print(f"\n=== Launching: {job['run_name']} ===")
        print(f"Seed: {job['seed']}")
        print(f"Device: {job['device'] or 'default'}")
        print("Command:")
        print(" ".join(job["cmd"]))

        proc = subprocess.Popen(job["cmd"], cwd=BACKEND_DIR)
        processes.append((proc, job))

    failed = []

    for proc, job in processes:
        ret = proc.wait()
        if ret != 0:
            failed.append((job["run_name"], ret))
        else:
            print(f"✔ Training completed for {job['run_name']}")

    for _, job in processes:
        config_path = job["config_path"]
        if config_path.exists():
            config_path.unlink()

    if failed:
        msgs = ", ".join([f"{name} (exit {code})" for name, code in failed])
        raise RuntimeError(f"Some training runs failed: {msgs}")


def main():
    parser = argparse.ArgumentParser(description="Offline GANomics training for manuscript results")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to project folder containing df_ag/df_rs and a *_config.yaml",
    )
    parser.add_argument("--epochs", type=int, help="Total epochs (split evenly between decay and warm-up)")
    parser.add_argument("--samples", type=int, help="Override max_samples for training")
    parser.add_argument("--beta", type=float, help="Override feedback weight (beta)")
    parser.add_argument("--lambda", type=float, dest="lambda_cycle", help="Override cycle weight (lambda)")
    parser.add_argument("--direction", choices=["both", "AtoB", "BtoA"], default="both", help="Training direction")
    parser.add_argument("--run-name", type=str, help="Custom run identifier (defaults to dataset folder name)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs with different seeds")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU ids to assign to tasks, e.g. '0' or '0,1,2'",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    base_config = locate_config(dataset_dir)
    base_run_name = args.run_name or dataset_dir.name
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    jobs = []
    for repeat_idx in range(args.repeats):
        run_name = make_run_name(base_run_name, repeat_idx, args.repeats)
        seed = 42 + repeat_idx

        device = None
        if gpu_ids:
            gpu_id = gpu_ids[repeat_idx % len(gpu_ids)]
            device = f"cuda:{gpu_id}"

        prepared_config = prepare_config(base_config, dataset_dir, run_name)

        cmd = build_command(
            config_path=prepared_config,
            epochs=args.epochs,
            samples=args.samples,
            beta=args.beta,
            lam=args.lambda_cycle,
            direction=args.direction,
            seed=seed,
            device=device,
        )

        jobs.append(
            {
                "run_name": run_name,
                "seed": seed,
                "device": device,
                "cmd": cmd,
                "config_path": prepared_config,
            }
        )

    if gpu_ids:
        max_parallel = len(gpu_ids)
    else:
        max_parallel = 1

    print(f"Prepared {len(jobs)} job(s)")
    print(f"Max parallel jobs: {max_parallel}")

    for i in range(0, len(jobs), max_parallel):
        batch_jobs = jobs[i:i + max_parallel]
        print(f"\n=== Starting batch {i // max_parallel + 1} ===")
        launch_batch(batch_jobs)

    print("\n✔ All training runs completed. Outputs live in results_ms/1_Training")


if __name__ == "__main__":
    main()