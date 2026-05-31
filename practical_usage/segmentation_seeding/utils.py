import json
import os
import pickle
import random
from functools import lru_cache

import cv2
import numpy as np
import torch
from skimage.morphology import remove_small_objects, skeletonize
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_pickle_dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


@lru_cache(maxsize=256)
def _load_graph_and_image(folder_path: str):
    with open(os.path.join(folder_path, 'graph.pickle'), 'rb') as f:
        graph = pickle.load(f)

    img_bgr = cv2.imread(os.path.join(folder_path, 'satellite.png'), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(os.path.join(folder_path, 'satellite.png'))
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    
    target_h, target_w = 2771, 2772
    if image_rgb.shape[0] != target_h or image_rgb.shape[1] != target_w:
        image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)        
    return graph, image_rgb

def read_index_file(path: str):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def normalized_to_pixel(xy, width: int, height: int):
    x = int(np.clip(round(xy[0] * (width - 1)), 0, width - 1))
    y = int(np.clip(round((1.0 - xy[1]) * (height - 1)), 0, height - 1))
    return x, y


def pixel_to_normalized_bottom_left(x_px: int, y_px_top_left: int, width: int, height: int):
    x_norm = 0.0 if width <= 1 else float(x_px) / float(width - 1)
    y_norm = 0.0 if height <= 1 else 1.0 - (float(y_px_top_left) / float(height - 1))
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    return x_norm, y_norm


def rasterize_graph_mask(graph, image_shape, line_width: int = 5):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for u, v in graph.edges:
        p1 = normalized_to_pixel(u, width, height)
        p2 = normalized_to_pixel(v, width, height)
        cv2.line(mask, p1, p2, color=255, thickness=line_width, lineType=cv2.LINE_AA)

    return (mask > 0).astype(np.float32)


def preprocess_image(image_rgb: np.ndarray, output_size):
    output_size = tuple(output_size)
    image_rs = cv2.resize(image_rgb, output_size, interpolation=cv2.INTER_LINEAR)
    image_rs = image_rs.astype(np.float32) / 255.0
    image_rs = (image_rs - IMAGENET_MEAN) / IMAGENET_STD
    image_t = torch.from_numpy(image_rs).permute(2, 0, 1).float()
    return image_t


def preprocess_mask(mask: np.ndarray, output_size):
    output_size = tuple(output_size)
    mask_rs = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    mask_t = torch.from_numpy(mask_rs[None, ...]).float()
    return mask_t


class PathwaySegmentationDataset(Dataset):
    def __init__(self, data_root: str, index_path: str, output_size=(896, 896), mask_line_width: int = 5):
        self.data_root = data_root
        self.indices = read_index_file(index_path)
        self.output_size = tuple(output_size)
        self.mask_line_width = int(mask_line_width)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample_id = self.indices[i]
        folder = os.path.join(self.data_root, str(sample_id))
        graph, image_rgb = _load_graph_and_image(folder)
        mask = rasterize_graph_mask(graph, image_rgb.shape[:2], line_width=self.mask_line_width)
        image_t = preprocess_image(image_rgb, self.output_size)
        mask_t = preprocess_mask(mask, self.output_size)
        return image_t, mask_t


class PathwayInferenceDataset(Dataset):
    def __init__(self, data_root: str, index_path: str, output_size=(896, 896)):
        self.data_root = data_root
        self.indices = read_index_file(index_path)
        self.output_size = tuple(output_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample_id = self.indices[i]
        folder = os.path.join(self.data_root, str(sample_id))
        _graph, image_rgb = _load_graph_and_image(folder)
        image_t = preprocess_image(image_rgb, self.output_size)
        orig_h, orig_w = image_rgb.shape[:2]
        return image_t, sample_id, orig_h, orig_w


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(iou.mean().item())


def select_seed_points(prob_map: np.ndarray, k: int = 10, threshold: float = 0.35, min_distance: int = 20):
    mask = prob_map >= float(threshold)
    mask = np.asarray(mask, dtype=bool)

    if mask.any():
        skel = skeletonize(mask)
        candidate_mask = skel if skel.any() else mask
    else:
        candidate_mask = np.ones_like(mask, dtype=bool)

    ys, xs = np.where(candidate_mask)
    if len(xs) == 0:
        flat_idx = np.argsort(prob_map.ravel())[::-1]
        ys, xs = np.unravel_index(flat_idx, prob_map.shape)

    scores = prob_map[ys, xs]
    order = np.argsort(scores)[::-1]

    height, width = prob_map.shape[:2]
    selected = []
    selected_xy = []
    min_dist_sq = float(min_distance) * float(min_distance)

    for idx in order:
        x = int(xs[idx])
        y = int(ys[idx])
        score = float(scores[idx])

        keep = True
        for px, py in selected_xy:
            dx = x - px
            dy = y - py
            if (dx * dx + dy * dy) < min_dist_sq:
                keep = False
                break
        if not keep:
            continue

        x_norm, y_norm = pixel_to_normalized_bottom_left(x, y, width, height)
        selected.append({
            'x_px': x,
            'y_px_top_left': y,
            'x_norm': x_norm,
            'y_norm': y_norm,
            'score': score,
        })
        selected_xy.append((x, y))

        if len(selected) >= int(k):
            break

    return selected
