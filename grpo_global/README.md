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
inference_config.json
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

## Important Config Values

Key GRPO parameters in `config.py`:

```python
"num_epochs": 9,
"grad_accum": 8,
"lr": 3e-5,
"weight_decay": 1e-4,
"grad_clip": 1.0,
"kl_beta": 0.05,
"K": 4,
"buffer_radius": 0.01,
"temperature": 1,
"max_global_steps": 200,
"seed_strategy": "gt_random",
"seeds_per_graph": 1,
```

`train.py` reads rollout settings from `inference_config.json`:

```python
"topk"
"restarts_per_graph"
"stop_prob_thresh"
"max_expand_per_node"
```

## Implementation Note: `fp_lambda`

In the current snapshot, `train.py` sets:

```python
CONFIG["fp_lambda"] = float(args.fp_lambda)
```

but the parser does not define `--fp_lambda`. Before running, either add:

```python
parser.add_argument("--fp_lambda", type=float, default=0.0)
```

to `train.py`, or set `CONFIG["fp_lambda"]` directly in `config.py` and remove that line from `train.py`.

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
