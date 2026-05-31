# Practical Usage

This folder contains the deployment-oriented pipeline for using TraversRL without ground-truth start nodes. In the paper experiments, inference can start from ground-truth graph nodes for controlled evaluation. In practice, those nodes are unavailable. Therefore, this practical pipeline uses segmentation-guided seeds and iterative coverage inference.

## Folder Structure

```text
practical_usage/
├── segmentation_seeding/           # Train segmentation model and generate seed points
└── segmentation_guided_inference/  # Run TraversRL from segmentation-derived seeds
```

Detailed instructions are provided in each folder:

- [`segmentation_guided_seeding/README.md`](segmentation_guided_seeding/README.md)
- [`segmentation_guided_inference/README.md`](segmentation_guided_inference/README.md)

## Workflow

The practical pipeline has two stages:

1. **Segmentation-guided seed generation**
   Train a lightweight segmentation model to predict pathway probability maps from overhead imagery. High-probability pathway pixels are selected as candidate seed points.

2. **Segmentation-guided TraversRL inference**
   Use the generated seeds to run TraversRL. Since a scene may contain disconnected pathway components, inference follows an iterative coverage procedure: generate from one seed, accept non-overlapping paths, remove nearby covered seeds, and repeat until no uncovered seeds remain.


Please follow the README under each folder.

## Notes

This practical pipeline is intended for deployment-style usage where ground-truth graph nodes are unavailable. Its quality depends on both the segmentation seed quality and the TraversRL generator. If the segmentation model misses an entire disconnected component, TraversRL may not recover that component because no seed is available there.
