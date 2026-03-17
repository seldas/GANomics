"""Offline workflow driver for GANomics training runs."""
import argparse
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "dashboard" / "backend"
RESULTS_MS_DIR = BACKEND_DIR / "results_ms"
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


def build_command(config_path: Path, epochs: int, samples: int, beta: float, lam: float, direction: str):
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
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Offline GANomics training for manuscript results")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to project folder containing df_ag/df_rs and a *_config.yaml")
    parser.add_argument("--epochs", type=int, help="Total epochs (split evenly between decay and warm-up)")
    parser.add_argument("--samples", type=int, help="Override max_samples for training")
    parser.add_argument("--beta", type=float, help="Override feedback weight (beta)")
    parser.add_argument("--lambda", type=float, dest="lambda_cycle", help="Override cycle weight (lambda)")
    parser.add_argument("--direction", choices=["both", "AtoB", "BtoA"], default="both", help="Training direction")
    parser.add_argument("--run-name", type=str, help="Custom run identifier (defaults to dataset folder name)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")

    base_config = locate_config(dataset_dir)
    run_name = args.run_name or dataset_dir.name

    prepared_config = prepare_config(base_config, dataset_dir, run_name)
    try:
        cmd = build_command(prepared_config, args.epochs, args.samples, args.beta, args.lambda_cycle, args.direction)
        print("Running training with command:")
        print(" ".join(cmd))
        result = subprocess.run(cmd, cwd=BACKEND_DIR)
        if result.returncode != 0:
            raise RuntimeError(f"Training script failed with exit code {result.returncode}")
        print("✔ Training completed. Outputs live in results_ms/1_Training")
    finally:
        if prepared_config.exists():
            prepared_config.unlink()


if __name__ == "__main__":
    main()
