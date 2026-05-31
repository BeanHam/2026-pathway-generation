# GRPO Global Reward Fine-Tuning (`grpo_global/`)

This folder trains **TraversRL-Global**, the RL fine-tuned version of TraversRL that optimizes a terminal graph-level reward.

Training starts from a supervised `TraversRL-Pre` checkpoint. For each training scene, the policy samples a group of rollouts, converts each rollout into a predicted pathway graph, scores each graph with buffered IoU, and converts the rollout rewards into group-relative advantages. The policy is then updated with an advantage-weighted log-likelihood objective plus KL regularization to the frozen supervised reference policy.

## Files

```text
grpo_global/
├── config.py      # dataset paths, reference checkpoint paths, and GRPO hyperparameters
├── model.py       # PathwayModel with 4-channel RGB+canvas input
├── utils.py       # rollout collection, buffered-IoU reward, GRPO update, validation
├── train.py       # GRPO-global training entry point
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

Edit `grpo_global/config.py` before training:

```python
CONFIG["data_root"] = "/path/to/DATASET_ROOT/pairs/"
CONFIG["train_index"] = "/path/to/DATASET_ROOT/train.txt"
CONFIG["val_index"] = "/path/to/DATASET_ROOT/val.txt"
CONFIG["save_path"] = "/path/to/save/grpo_logs/"

REFERENCE_DIRS = {
    "resnet18": "/path/to/reference_logs/resnet18/",
}

## Make sure the selected model exists in REFERENCE_DIRS and that the corresponding reference folder contains both `model_best.pth` and `inference_config.json`.
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
grpo_logs/
└── resnet18_YYYYMMDD_HHMMSS/
    ├── run_config.json
    ├── history.json
    ├── model_last.pth
    └── model_best.pth
```

## Typical Global-GRPO Flow

```bash
# 1. Train TraversRL-Pre in ../pretraining
# 2. Run ../pretraining/parameter_tuning.py to create inference_config.json
# 3. Edit grpo_global/config.py and REFERENCE_DIRS
cd grpo_global
python train.py --model resnet18 --device cuda:0
```
