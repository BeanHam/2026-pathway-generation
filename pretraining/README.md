# Supervised Pretraining (`pretraining/`)

This folder trains **TraversRL-Pre**, the supervised graph-generation policy used as the reference model for later GRPO fine-tuning.

The model learns to expand pedestrian pathway graphs through teacher-forced node expansion. At each step, the policy receives a local RGB crop plus a canvas of the current partial graph and predicts a discretized direction-distance action or `Stop`.

## Files

```text
pretraining/
├── config.py              # dataset paths, model/action settings, training hyperparameters
├── model.py               # PathwayModel with 4-channel RGB+canvas input
├── utils.py               # graph loading, crop generation, action encoding, training utilities
├── train.py               # DDP supervised training entry point
└── parameter_tuning.py    # rollout-parameter tuning on validation scenes
```

## Configure Paths

Edit `pretraining/config.py` before running:

```python
CONFIG = {
    "data_root": "/path/to/DATASET_ROOT/pairs/",
    "train_index": "/path/to/DATASET_ROOT/train.txt",
    "val_index": "/path/to/DATASET_ROOT/val.txt",
    "save_path": "/path/to/save/reference_logs/",
    ...
}
```

The default code uses absolute paths from the original experiment environment, so these paths must be changed for a new machine.

## Train TraversRL-Pre

`train.py` initializes distributed training, so use `torchrun` even for a single GPU:

```bash
cd pretraining
torchrun --nproc_per_node=1 train.py --model resnet18 --accum_graphs 1

## supported models:
{resnet18, resnet50, convnext_tiny, convnext_small, swin_tiny, swin_small, vit_tiny, vit_small, vit_b_16}
```

Common model names include:

```text
{resnet18, resnet50, convnext_tiny, convnext_small, swin_tiny, swin_small, vit_tiny, vit_small, vit_b_16}
```

## Outputs

Each run creates a timestamped folder under `CONFIG["save_path"]`, for example:

```text
reference_logs/
└── resnet18_YYYYMMDD_HHMMSS/
    ├── model_best.pth
    ├── model_last.pth
    ├── loss_log.pkl
    └── run_config.json
```

`model_best.pth` is selected by validation loss and is used as the initialization/reference checkpoint for GRPO fine-tuning.

## Rollout-Parameter Tuning

After supervised training, run:

```bash
python parameter_tuning.py
```

This script searches validation-time rollout parameters such as top-k action selection, restart count, stop threshold, and maximum node expansions.

Before running, edit the hard-coded validation path and checkpoint list inside `parameter_tuning.py`. The script writes:

```text
inference_config.json
```

into the corresponding supervised run folder. Both `grpo_global/` and `grpo_local/` expect the reference folder to contain:

```text
model_best.pth
inference_config.json
```

## Typical Pretraining Flow

```bash
# 1. Edit config.py paths
cd pretraining

# 2. Train supervised reference model
torchrun --nproc_per_node=1 train.py --model resnet18 --accum_graphs 1

# 3. Tune rollout parameters for the trained checkpoint
python parameter_tuning.py
```
