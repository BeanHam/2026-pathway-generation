# TraversRL: Traversable Pedestrian Pathway Generation With Reinforcement Learning

This repository contains research code for **TraversRL**, a vision-conditioned framework for generating connected pedestrian pathway graphs from overhead imagery.

TraversRL treats pedestrian pathway extraction as an iterative graph-generation problem instead of a segmentation-only problem. Given an aerial image and a partial graph canvas, the model predicts direction-distance actions to grow a pedestrian network. The codebase supports:


## Repository Structure

```text
2026-pathway-generation/
├── pretraining/              # Supervised TraversRL-Pre training and rollout tuning
├── grpo_global/              # GRPO fine-tuning with terminal graph-level reward
├── grpo_local/               # GRPO fine-tuning with local stepwise reward shaping
├── segmentation_seeding/     # Segmentation model and seed generation
├── evaluation/               # Inference, GeoJSON export, caching, and metrics
├── requirements.txt          # Python dependencies
└── README.md
```

Detailed usage for the three training stages is documented inside each folder:

- [`pretraining/README.md`](pretraining/README.md)
- [`grpo_global/README.md`](grpo_global/README.md)
- [`grpo_local/README.md`](grpo_local/README.md)


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

If `cv2` installation fails, install OpenCV directly:

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

Each scene folder should include:

- `satellite.png`: overhead RGB image;
- `graph.pickle`: a NetworkX graph whose nodes are normalized `(x, y)` coordinates in `[0, 1]^2`;
- `bbox.json`: geographic bounding box used to export predictions as GeoJSON.

Expected `bbox.json` format:

```json
{"bbox": [min_lon, max_lon, min_lat, max_lat]}
```

## Overall Workflow

A typical experiment follows this order:

1. Update dataset/checkpoint paths in config files.
2. Train the supervised reference model in `pretraining/`.
3. Tune rollout parameters with `pretraining/parameter_tuning.py`.
4. Fine-tune with global reward in `grpo_global/`.
5. Fine-tune with local reward shaping in `grpo_local/`.
6. Optionally train the segmentation seed model in `segmentation_seeding/`.
7. Run inference and evaluation from `evaluation/`.