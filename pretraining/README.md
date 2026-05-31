# Supervised Pretraining

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

Edit path information in `pretraining/config.py` before running:

```python
CONFIG = {
    "data_root": "/path/to/DATASET_ROOT/pairs/",
    "train_index": "/path/to/DATASET_ROOT/train.txt",
    "val_index": "/path/to/DATASET_ROOT/val.txt",
    "save_path": "/path/to/save/reference_logs/",
    ...
}
```

## Train TraversRL-Pre

```bash
cd pretraining
torchrun --nproc_per_node=1 train.py --model resnet18 --accum_graphs 1     ## single GPU
## torchrun --nproc_per_node=4 train.py --model resnet18 --accum_graphs 1  ## multiple GPUs

## supported models:
## {resnet18, resnet50, convnext_tiny, convnext_small, swin_tiny, swin_small, vit_tiny, vit_small, vit_b_16}
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

After supervised training, we select inference-time rollout parameters on validation scenes by maximizing buffered IoU. Edit the validation path, checkpoint list, and parameter grids inside `parameter_tuning.py`:

```python
## path info
LOGS = ["/path/to/reference_logs/resnet18/"]
VAL_PATH = "/path/to/DATASET_ROOT/val.txt"

## parameter grids
RESTART_GRID = (1, 5)
TOPK_GRID = (5, 10)
STOP_PROB_GRID = (0.4, 0.5, 0.6, 0.7)
MAX_EXPAND_GRID = (4, 6, 8)
```

Parameter meanings:

- `LOGS`: supervised run folders to tune. Each folder should contain `model_best.pth` and `run_config.json`.
- `VAL_PATH`: validation scene index used for tuning.
- `RESTART_GRID`: candidate values for `restarts_per_graph`. Each restart runs a full rollout from a newly sampled start node; the script keeps the rollout with the highest internal confidence score. Larger values improve robustness but increase runtime.
- `TOPK_GRID`: candidate values for `topk`. At each step, the rollout considers the top-`k` actions by logit and executes the first valid action. `topk=1` is greedy decoding; larger values allow backoff when the top action is invalid, repeated, or an under-threshold `Stop`.
- `STOP_PROB_GRID`: candidate values for `stop_prob_thresh`. `Stop` is accepted only when its probability is at least this threshold. Higher values make the model less likely to stop early; lower values make shorter rollouts more likely.
- `MAX_EXPAND_GRID`: candidate values for `max_expand_per_node`. This limits how many outgoing actions can be generated from one node. Larger values allow more branching but can add noisy edges.

Then run:

```bash
torchrun --nproc_per_node=1 parameter_tuning.py  ## single GPUs
## torchrun --nproc_per_node=4 parameter_tuning.py  ## multiple GPUs
```

The script writes `inference_config.json` into the corresponding supervised run folder.


## Typical Pretraining Flow

```bash
# 1. Edit config.py paths
cd pretraining

# 2. Train supervised reference model
torchrun --nproc_per_node=1 train.py --model resnet18 --accum_graphs 1

# 3. Tune rollout parameters for the trained checkpoint
torchrun --nproc_per_node=1 parameter_tuning.py
```
