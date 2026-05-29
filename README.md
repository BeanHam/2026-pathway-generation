# TraversRL: Traversable Pedestrian Pathway Generation With Reinforcement Learning

This repository contains the research code for **TraversRL**, a vision-conditioned framework for generating connected pedestrian pathway graphs from overhead imagery.

TraversRL treats pedestrian pathway extraction as an iterative graph-generation problem rather than a segmentation-only problem. Given an aerial image and a current partial graph, the model predicts direction-distance actions to extend the pathway network. The repository supports:

* supervised pretraining of the graph generator (`TraversRL-Pre`);
* GRPO fine-tuning with a terminal graph-level reward (`TraversRL-Global`);
* GRPO fine-tuning with local stepwise reward shaping (`TraversRL-Local`);
* segmentation-guided seed generation for deployment-style inference;
* rollout, GeoJSON export, and evaluation with buffered IoU and TravSim.

## Repository Structure

```text
2026-pathway-generation/
├── pretraining/              # Supervised TraversRL-Pre training and rollout-parameter tuning
├── grpo_global/              # GRPO fine-tuning with terminal graph-level reward
├── grpo_local/               # GRPO fine-tuning with local stepwise reward shaping
├── segmentation_seeding/     # Segmentation model and seed generation for deployment-style inference
├── evaluation/               # Inference, caching, metrics, and evaluation scripts
├── requirements.txt          # Python dependencies
└── README.md
```

## Important Setup Note

This is research code. Many scripts currently contain absolute paths used in the original experiments, such as:

```text
/data/aiclinic/pathways/...
/data/models/bin/2026_pathway_generation/...
```

Before running experiments, update the paths in:

* `pretraining/config.py`
* `grpo_global/config.py`
* `grpo_local/config.py`
* `segmentation_seeding/config.py`
* `segmentation_seeding/seed_generation.py`
* `evaluation/inference.py`
* `evaluation/inference_segmentation_seeding.py`
* `evaluation/cache.py`
* `evaluation/evaluation.py`
* `evaluation/evaluation_segmentation_seeding.py`

The inference and evaluation scripts also contain hard-coded lists of datasets, checkpoints, and output directories. Edit those lists to match your local paths.

## Environment Setup

Create a fresh environment:

```bash
conda create -n traversrl python=3.10 -y
conda activate traversrl
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If installation fails because of `cv2`, install OpenCV directly:

```bash
pip install opencv-python
```

For geospatial dependencies, `conda-forge` is often more reliable:

```bash
conda install -c conda-forge geopandas shapely fiona pyproj osmnx -y
```


## Expected Data Format

The code expects a dataset root containing scene folders listed by index files.

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

Each index file should contain one scene id per line. Each scene folder should include:

* `satellite.png`: overhead RGB image.
* `graph.pickle`: a NetworkX graph whose nodes are normalized `(x, y)` coordinates in `[0, 1]^2`.
* `bbox.json`: geographic bounding box used to export predictions as GeoJSON. The expected format is:

```json
{"bbox": [min_lon, max_lon, min_lat, max_lat]}
```

`bbox.json` is required by the inference/evaluation scripts that convert normalized graph edges into longitude/latitude GeoJSON.


## Overall Workflow

The typical workflow is:

1. Update config paths.
2. Train a supervised graph generator (`pretraining/`).
3. Tune rollout parameters and save `inference_config.json`.
4. Fine-tune with GRPO global reward (`grpo_global/`).
5. Fine-tune with GRPO local reward shaping (`grpo_local/`).
6. Optionally train a segmentation model and generate deployment seeds (`segmentation_seeding/`).
7. Run inference and save GeoJSON predictions (`evaluation/inference*.py`).
8. Build ground-truth caches and evaluate metrics (`evaluation/cache.py`, `evaluation/evaluation*.py`).