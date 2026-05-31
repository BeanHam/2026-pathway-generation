# GRPO Global Reward Fine-Tuning (`grpo_global/`)

This folder trains **TraversRL-Global**, initialized from a supervised `TraversRL-Pre` checkpoint and fine-tuned with GRPO using a terminal graph-level reward.

For each training graph, the policy samples a group of rollouts. Each rollout is decoded into a predicted pathway graph and scored using a graph-level reward such as buffered IoU. Rewards are converted into group-relative advantages, and the policy is updated with an advantage-weighted objective plus KL regularization to the frozen supervised reference policy.

## Files

```text
grpo_global/
├── config.py      # dataset paths, reference checkpoints, GRPO hyperparameters
├── model.py       # PathwayModel architecture
├── utils.py       # rollout, reward computation, GRPO training utilities
└── train.py       # global-reward GRPO training entry point
```

## Prerequisites

Run supervised pretraining first. Each reference directory in `REFERENCE_DIRS` must contain:

```text
model_best.pth
inference_config.json   ## from rollout parameter tuning
```

`model_best.pth` initializes the policy and frozen reference model. `inference_config.json` provides the rollout parameters selected during validation tuning.

## Configure Paths

Edit `grpo_global/config.py`:

```python
CONFIG["data_root"] = "/path/to/DATASET_ROOT/pairs/"
CONFIG["train_index"] = "/path/to/DATASET_ROOT/train.txt"
CONFIG["val_index"] = "/path/to/DATASET_ROOT/val.txt"
CONFIG["save_path"] = "/path/to/save/grpo_logs/"

REFERENCE_DIRS = {
    "resnet18": "/path/to/reference_logs/resnet18_run/",
    "vit_small": "/path/to/reference_logs/vit_small_run/",
    "swin_tiny": "/path/to/reference_logs/swin_tiny_run/",
}
```

## Run Training

Single GPU:

```bash
cd grpo_global
python train.py --model resnet18 --device cuda:0
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 train.py --model resnet18
```

`train.py` reads rollout settings from `inference_config.json`:

```python
"topk"
"restarts_per_graph"
"stop_prob_thresh"
"max_expand_per_node"
```

## Outputs

Training writes checkpoints and logs under `CONFIG["save_path"]`. The exact run directory is created inside the GRPO training utility.

Typical output files include model checkpoints, logs, and validation summaries. The best checkpoint can then be used by `evaluation/inference.py`.

## Typical Global-GRPO Flow

```bash
# 1. Train TraversRL-Pre in ../pretraining
# 2. Run parameter_tuning.py so the reference folder has inference_config.json
# 3. Edit grpo_global/config.py and REFERENCE_DIRS
cd grpo_global
python train.py --model resnet18 --device cuda:0
```
