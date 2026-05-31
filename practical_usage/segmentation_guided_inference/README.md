# Segmentation-Guided Inference

This folder runs the practical inference pipeline for TraversRL when ground-truth start nodes are not available. Instead of starting rollout from a node in the annotated graph, the script starts from high-confidence seed points produced by the `segmentation_seeding/` pipeline.

The key difference from the paper's standard evaluation setting is that this script supports an iterative coverage pipeline for scenes that may contain multiple disconnected pathway components. It repeatedly generates from segmentation-derived seeds, accepts non-overlapping generations, removes seeds already covered by accepted paths, and stops when no uncovered seeds remain.

## Files

```text
segmentation_guided_inference/
├── config.py      # Model/action-space configuration and placeholder paths
├── inference.py   # Segmentation-guided iterative coverage inference
├── model.py       # TraversRL policy model/backbones
├── utils.py       # Shared rollout/model utilities
└── README.md
```

## Inputs

Before running this folder, you should already have:

1. **Trained TraversRL checkpoints** from `pretraining/`, `grpo_global/`, or `grpo_local/`.
2. **Rollout configs**: each run folder should contain `inference_config.json`, produced by `pretraining/parameter_tuning.py`.
3. **Segmentation seed file** from `segmentation_seeding/seed_generation.py`, typically:

```text
DATASET_ROOT/segmentation_seeds.json
```

The seed file should map each scene id to candidate seed points:

```json
{
  "123": {
    "points_px": [[1200, 900], [1450, 1100]],
    "points_norm": [[0.43, 0.67], [0.52, 0.60]],
    "scores": [0.91, 0.86]
  }
}
```

The dataset root should contain scene folders:

```text
DATASET_ROOT/pairs/
├── <scene_id>/
│   ├── satellite.png
│   ├── graph.pickle
│   └── bbox.json
└── ...
```

`bbox.json` is used to convert normalized graph coordinates back to longitude/latitude GeoJSON.

## Path Setup

Edit the module-level lists in `inference.py`:

```python
LOGS = [
    "/path/to/reference_logs/resnet18/",
    "/path/to/grpo_global_logs/resnet18/",
    "/path/to/grpo_local_logs/resnet18/",
]
MODELS = ["resnet18", "resnet18", "resnet18"]

DATASETS = [
    {
        "name":      "DATASET_NAME",
        "data_root": "/path/to/DATASET_ROOT/pairs/",
        "test_txt":  "/path/to/DATASET_ROOT/test.txt",
        "seed_root": "/path/to/DATASET_ROOT/segmentation_seeds.json",
        "out_root":  "/path/to/save/predictions/",
    },
]
```

Each path in `LOGS` should point to a trained run folder containing:

```text
model_best.pth
inference_config.json
```

## How the Iterative Coverage Pipeline Works

For each scene, `inference.py` loads the segmentation-derived seed pool and repeats:

1. Select a seed from the remaining seed pool.
2. Run TraversRL rollout from that seed.
3. Reject the generation if it overlaps previously accepted paths.
4. If accepted, add the generated edges to the accepted path set.
5. Remove remaining seeds that fall within a buffer around the accepted path.
6. Repeat until no uncovered seeds remain or `max_generations` is reached.

This allows inference to recover multiple disconnected pathway components in the same scene. Coverage is defined relative to the segmentation seed map: if the seed generator misses an entire component, this inference script cannot recover it.

## Coverage Parameters

The main coverage settings are set inside `inference.py` after loading `inference_config.json`:

```python
infer_cfg["coverage_buffer_radius_px"] = 108   # approximately 4m
infer_cfg["coverage_line_width"] = 2
infer_cfg["min_generated_edges"] = 1
infer_cfg["edge_round_decimals"] = 6
infer_cfg["max_generations"] = 1000
infer_cfg["save_iteration_geojsons"] = False
```

- `coverage_buffer_radius_px`: pixel buffer used to decide whether seeds are already covered and whether a new generation overlaps an accepted path. The default `108` corresponds to roughly 4 meters in the original experiments.
- `coverage_line_width`: rasterized line width for generated paths.
- `min_generated_edges`: minimum number of edges required for a generation to be considered.
- `max_generations`: maximum number of generation attempts per scene.
- `save_iteration_geojsons`: if `True`, saves each accepted generation separately for debugging.

Adjust `coverage_buffer_radius_px` if your image resolution or physical scale differs from the original data.

## Run Inference

Single GPU:

```bash
python inference.py
```

Single GPUs:

```bash
torchrun --nproc_per_node=4 inference.py
```

## Outputs

Predictions are written under `out_root` with the structure:

```text
PREDICTION_ROOT/
├── resnet18/
│   ├── base/
│   │   └── seed_0/
│   │       ├── <scene_id>_seed0_gt.geojson
│   │       ├── <scene_id>_seed0_pred.geojson
│   │       └── <scene_id>_seed0_iteration_debug.json
│   ├── global/
│   └── local/
└── ...
```