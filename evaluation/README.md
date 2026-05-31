# Evaluation

This folder contains inference and evaluation utilities for TraversRL. It is used after supervised pretraining and/or GRPO fine-tuning to generate GeoJSON pathway predictions and compute graph-quality metrics.

## Files

- `inference.py`: runs rollout inference from trained TraversRL checkpoints and saves predicted/ground-truth GeoJSON pairs.
- `cache.py`: builds cached ground-truth geometry and tile-level traversability objects for faster evaluation.
- `evaluation.py`: computes final metrics from saved GeoJSON predictions and cached ground truth.
- `metrics.py`: metric implementations, including buffered IoU, graph statistics, and TravSim-related utilities.
- `model.py`: TraversRL policy model definition used for checkpoint loading during inference.
- `utils.py`: shared graph loading, crop extraction, rollout, and helper functions.
- `config.py`: shared model/action-space configuration used by inference utilities.

## Expected Inputs

The evaluation scripts expect the same dataset structure used during training:

```text
DATASET_ROOT/
├── pairs/
│   ├── <scene_id>/
│   │   ├── satellite.png
│   │   ├── graph.pickle
│   │   └── bbox.json
│   └── ...
└── test.txt
```

Each trained run directory used for inference should contain:

```text
RUN_DIR/
├── model_best.pth
└── inference_config.json
```

`inference_config.json` is produced during rollout parameter tuning in `pretraining/` and stores the selected rollout settings.

## 1. Configure Inference

Before running inference, edit the lists near the bottom of `inference.py`:

```python
LOGS = [
    "REFERENCE_RUN_DIR/",
    "GRPO_GLOBAL_RUN_DIR/",
    "GRPO_LOCAL_RUN_DIR/",
]

MODELS = ["resnet18", "resnet18", "resnet18"]

DATASETS = [
    {
        "name": "DATASET_NAME",
        "data_root": "DATASET_ROOT/pairs/",
        "test_txt": "DATASET_ROOT/test.txt",
        "out_root": "PREDICTION_OUTPUT_ROOT/",
    },
]
```

The script uses the checkpoint in each run directory and writes predictions under `out_root`.

## 2. Run Inference

Single GPU:

```bash
python inference.py
```

Distributed/sharded inference:

```bash
torchrun --nproc_per_node=4 inference.py
```

The output structure is:

```text
PREDICTION_OUTPUT_ROOT/
├── resnet18/
│   ├── base/
│   │   └── seed_0/
│   │       ├── <scene_id>_seed0_gt.geojson
│   │       └── <scene_id>_seed0_pred.geojson
│   ├── global/
│   └── local/
└── ...
```

The tags are inferred from the run directory name:

- `base`: supervised pretrained model;
- `global`: GRPO global-reward model;
- `local`: GRPO local-stepwise-reward model.

## 3. Build Ground-Truth Cache

Before running evaluation, edit the `DATASETS` list in `cache.py`:

```python
DATASETS = [
    {
        "name": "DATASET_NAME",
        "pred_root": "PREDICTION_OUTPUT_ROOT/",
        "split": "DATASET_ROOT/test.txt",
    },
]
```

Then run:

```bash
python cache.py
```

This creates a cache folder under `pred_root`, named using the projection and buffer settings from `metrics.py`.

## 4. Run Evaluation

Edit the `DATASETS` list in `evaluation.py`:

```python
DATASETS = [
    {
        "name": "DATASET_NAME",
        "pred_root": "PREDICTION_OUTPUT_ROOT/",
        "split": "DATASET_ROOT/test.txt",
        "out_csv": "PREDICTION_OUTPUT_ROOT/evaluation_results.csv",
    },
]
```

Then run:

```bash
python evaluation.py
```

The script reports metrics for each model/stage/seed and saves an aggregated CSV.

## Metrics

The main metrics include:

- **Buffered IoU**: geometric overlap between buffered predicted and ground-truth graphs.
- **TravSim**: traversability similarity based on boundary-to-boundary connectivity within local tiles.
- **Graph statistics**: average degree, average betweenness, number of connected components, and related measures.

Metric constants such as projection and buffer distance are defined in `metrics.py`:

```python
PROJ = "epsg:26910"
BUFF_DIS = 1  # meters
```

## Notes

- `inference.py`, `cache.py`, and `evaluation.py` contain small editable configuration lists near the bottom of each file. Update those paths before running.
- The current scripts use placeholder paths such as `DATASET_ROOT/`, `RUN_DIR/`, and `PREDICTION_OUTPUT_ROOT/`. Replace them with your local dataset, checkpoint, and output locations.
- For reproducibility, keep each run directory paired with the `inference_config.json` produced by the pretraining rollout-parameter tuning step.
