# Offline Training Task Log

This keeps track of the `workflow/train.py` invocations that are currently running or were just executed for manuscript data. Each entry documents the task ID, purpose, and the exact command used so you can reproduce or rerun it manually.

| Task ID | Description | Command | Notes |
| --- | --- | --- | --- |
| `NB_Size_400` | Train on the Neuroblastoma project with 400 samples per epoch; repeat 5 independent seeds for stability. All other config values (beta=10, lambda=10, epochs=500/decay) remain at their defaults from `*_config.yaml`. | ```bash
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/NB \
  --samples 400 \
  --epochs 500 \
  --run-name NB_Size_400_Run_0
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/NB \
  --samples 400 \
  --epochs 500 \
  --run-name NB_Size_400_Run_1
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/NB \
  --samples 400 \
  --epochs 500 \
  --run-name NB_Size_400_Run_2
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/NB \
  --samples 400 \
  --epochs 500 \
  --run-name NB_Size_400_Run_3
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/NB \
  --samples 400 \
  --epochs 500 \
  --run-name NB_Size_400_Run_4
``` | Each run writes checkpoints/logs under `dashboard/backend/results_ms/1_Training`. Adjust `--run-name` suffix to keep outputs tied to unique repeats. |
| `CycleGAN_Size_50` | Benchmark CycleGAN architecture on the 50-sample run from the manuscript to compare legacy vs. GANomics performance. Uses the project folder `dashboard/backend/dataset/CycleGAN_50_0` and default betas/lambdas. | ```bash
python workflow/train.py \
  --dataset-dir dashboard/backend/dataset/CycleGAN_50_0 \
  --samples 50 \
  --epochs 500 \
  --run-name CycleGAN_50_0_Run_0
``` | Add additional repeats by rerunning with `_Run_1`, `_Run_2`, etc., if you need more statistical confidence. |

Add new rows here whenever you queue another offline batch task so the manuscript team can track what’s already been generated.
