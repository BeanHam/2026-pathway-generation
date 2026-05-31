# Segmentation-Guided Seeding (`segmentation_seeding/`)

This folder trains a lightweight segmentation model and converts its probability maps into seed points for TraversRL inference. The goal is to remove the deployment-time dependence on ground-truth start nodes: the segmentation model proposes likely pathway locations, and TraversRL starts rollouts from those predicted seeds.

## Files

```text
segmentation_seeding/
├── config.py             # dataset paths, model settings, training and seed parameters
├── model.py              # Swin encoder + FPN-style decoder for binary pathway segmentation
├── utils.py              # dataset loading, graph rasterization, IoU, seed selection utilities
├── train.py              # DDP segmentation training entry point
└── seed_generation.py    # inference script that exports segmentation-derived seeds
```

## Configure Paths

Edit `segmentation_seeding/config.py` before running:

```python
CONFIG = {
    "data_root": "/path/to/DATASET_ROOT/pairs/",
    "train_index": "/path/to/DATASET_ROOT/train.txt",
    "val_index": "/path/to/DATASET_ROOT/val.txt",
    "test_index": "/path/to/DATASET_ROOT/test.txt",
    "save_path": "/path/to/save/segmentation_logs/",
    ...
}
```

The expected dataset format is the same as the rest of the repository:

```text
DATASET_ROOT/
├── pairs/
│   ├── <scene_id>/
│   │   ├── satellite.png
│   │   ├── graph.pickle
│   │   └── bbox.json
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

## Train the Segmentation Model

`train.py` uses distributed training, so launch it with `torchrun` even for one GPU:

```bash
cd segmentation_seeding
torchrun --nproc_per_node=1 train.py --model swin_tiny

## supported models:
## {swin_tiny, swin_small}
```

The model takes RGB imagery and predicts a single-channel pathway logit map. Training masks are created by rasterizing the pathway graph with `mask_line_width` from `config.py`.

## Outputs

Training creates a timestamped folder under `CONFIG["save_path"]`, for example:

```text
segmentation_logs/
└── swin_tiny_YYYYMMDD_HHMMSS/
    ├── model_best.pth
    ├── model_last.pth
    ├── loss_log.pkl
    └── run_config.json
```

`model_best.pth` is the checkpoint used by `seed_generation.py`.

## Generate Segmentation-Derived Seeds

After training, run:

```bash
python seed_generation.py \
  --checkpoint /path/to/segmentation_logs/swin_tiny_run/model_best.pth \
  --model swin_tiny \
  --threshold 0.5 \
  --k 200 \
  --min_distance 50
```

The seed-generation script:

1. predicts a pathway probability map for each scene;
2. thresholds the map using `--threshold`;
3. keeps up to `--k` high-confidence points;
4. enforces `--min_distance` pixel spacing between seeds;
5. writes normalized and pixel-coordinate seeds to JSON.

Before running, edit the `datasets` list inside `seed_generation.py` so that each `data_root`, `index_file`, and `output_json` points to your local datasets.

The output JSON has the form:

```json
{
  "<scene_id>": {
    "points_px": [[x_px, y_px_top_left], ...],
    "points_norm": [[x_norm, y_norm], ...],
    "scores": [score, ...]
  }
}
```

## Typical Flow

```bash
# 1. Edit segmentation_seeding/config.py
cd segmentation_seeding

# 2. Train segmentation model
torchrun --nproc_per_node=1 train.py --model swin_tiny

# 3. Edit seed_generation.py dataset paths

# 4. Generate seeds
python seed_generation.py \
  --checkpoint /path/to/model_best.pth \
  --model swin_tiny \
  --threshold 0.5 \
  --k 200 \
  --min_distance 50

```
