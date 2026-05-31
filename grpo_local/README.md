# GRPO Local Reward Fine-Tuning (`grpo_local/`)

This folder trains **TraversRL-Local**, the local-reward version of TraversRL. It starts from a supervised `TraversRL-Pre` checkpoint and fine-tunes the policy with GRPO using:

1. a terminal graph-level reward based on final rollout quality, and
2. a local stepwise reward that gives additional credit to edge actions that add buffered geometry overlapping the ground-truth pathway.

The goal is to keep the stable graph-level optimization from `grpo_global/` while adding finer credit assignment for local edge placement.

## Files

```text
grpo_local/
├── config.py      # dataset paths, reference checkpoints, and local-GRPO settings
├── model.py       # PathwayModel architecture
├── utils.py       # rollout, reward computation, and local-GRPO training utilities
├── train.py       # training entry point
└── README.md
```

## Prerequisites

Run supervised pretraining first. Each pretrained reference folder must contain:

```text
model_best.pth
inference_config.json
```

- `model_best.pth` initializes both the trainable policy and the frozen reference policy.
- `inference_config.json` stores rollout parameters selected by `pretraining/parameter_tuning.py`.

## Configure Paths

Edit `config.py` before training:

```python
CONFIG["data_root"] = "/path/to/DATASET_ROOT/pairs/"
CONFIG["train_index"] = "/path/to/DATASET_ROOT/train.txt"
CONFIG["val_index"] = "/path/to/DATASET_ROOT/val.txt"
CONFIG["save_path"] = "/path/to/save/grpo_local_logs/"

REFERENCE_DIRS = {
    "resnet18": "/path/to/reference_logs/resnet18_run/",
}
```

## Run Training

Single GPU:

```bash
cd grpo_global
python train.py --model resnet18 --device cuda:0
```

Multi-GPU with DDP:

```bash
cd grpo_global
torchrun --nproc_per_node=4 train.py --model resnet18
```

## Outputs

Each run creates a timestamped directory under `CONFIG["save_path"]`, for example:

```text
grpo_local_logs/
└── resnet18_YYYYMMDD_HHMMSS/
    ├── run_config.json
    ├── history.json
    ├── model_last.pth
    └── model_best.pth
```

## Typical Workflow

```bash
# 1. Train TraversRL-Pre in ../pretraining
# 2. Run ../pretraining/parameter_tuning.py to create inference_config.json
# 3. Edit grpo_local/config.py and REFERENCE_DIRS
# 4. Train TraversRL-Local
cd grpo_local
python train.py --model resnet18 --device cuda:0
```