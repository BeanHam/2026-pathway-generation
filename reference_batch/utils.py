import os
import cv2
import torch
import pickle
import random
import numpy as np
import torch.nn as nn

from functools import lru_cache
from torch.amp import autocast
from config import *

# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
@lru_cache(maxsize=256)
def _load_graph_and_image(path: str):
    with open(os.path.join(path, "graph.pickle"), "rb") as f:
        G = pickle.load(f)
    img_path = os.path.join(path, "satellite.png")  # change if needed
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    satellite_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return G, satellite_img

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------
def should_freeze_bn(model_name: str) -> bool:
    return model_name.lower().startswith("resnet")

def freeze_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()
        m.requires_grad_(False)

def apply_noise(node_pos, scale=0.02):
    x, y = node_pos
    dx = np.random.uniform(-scale, scale)
    dy = np.random.uniform(-scale, scale)
    return (np.clip(x + dx, 0, 1), np.clip(y + dy, 0, 1))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# Crop (fast)
# ------------------------------------------------------------
def crop_image_fast(img_arr, center, crop_size=0.5, output_size=(256, 256)):
    H, W = img_arr.shape[:2]
    cx_norm, cy_norm = center

    cx = cx_norm * W
    cy = (1.0 - cy_norm) * H
    half = (crop_size * W) / 2.0

    out_h, out_w = output_size
    sx = out_w / (2.0 * half)
    sy = out_h / (2.0 * half)

    M = np.array([
        [sx, 0.0, (out_w / 2.0) - sx * cx],
        [0.0, sy, (out_h / 2.0) - sy * cy],
    ], dtype=np.float32)

    return cv2.warpAffine(
        img_arr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

# ------------------------------------------------------------
# Label encoder (same as yours)
# ------------------------------------------------------------
class IterativeEncoder:
    def __init__(self, config):
        self.num_angles = config['num_angles']
        self.stop_idx = config['stop_class_idx']
        self.dist_bins = np.array([
            0.0, 0.02, 0.04, 0.06, 0.08,
            0.1, 0.12, 0.14, 0.16, 0.18, 0.2
        ])
        self.num_dists = len(self.dist_bins) - 1
        self.angle_step = 360.0 / self.num_angles
        self.angle_offset = -(self.angle_step / 2.0)

    def get_action_label(self, u, v):
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        deg = np.degrees(np.arctan2(dy, dx)) % 360
        deg_shift = (deg - self.angle_offset) % 360
        a_idx = int(deg_shift / self.angle_step) % self.num_angles

        dist = np.sqrt(dx**2 + dy**2)
        d_idx = np.digitize(dist, self.dist_bins) - 1
        d_idx = max(0, min(d_idx, self.num_dists - 1))
        return (a_idx * self.num_dists) + d_idx

# ------------------------------------------------------------
# Batched graph runner
# ------------------------------------------------------------
class GraphRunner:
    """
    Runs ONE graph teacher-forced rollout and yields ONE STEP at a time.

    Key: preserves your old behavior:
      - global_canvas per-graph
      - stack traversal with neighbor pushes
      - canonical neighbor order
      - STOP when remaining_set empty for node
      - node_loss = mean(step losses for that node)
      - noise applied when node popped (train only)
    """
    def __init__(self, folder_idx: str, encoder: IterativeEncoder, CONFIG: dict, mode: str,
                 teacher_noise: float):
        self.folder_idx = str(folder_idx)
        self.encoder = encoder
        self.CONFIG = CONFIG
        self.mode = mode
        self.teacher_noise = teacher_noise if (mode == "train") else 0.0

        path = os.path.join(CONFIG["data_root"], self.folder_idx)
        self.G, self.sat = _load_graph_and_image(path)
        self.H, self.W = self.sat.shape[:2]

        if len(self.G.nodes) == 0:
            self.done = True
            return

        # rollout state (same as your process_graph)
        self.canvas = np.zeros((self.H, self.W), dtype=np.uint8)
        self.generated_edges = set()
        self.expanded_nodes = set()
        self.queued_nodes = set()

        all_nodes = list(self.G.nodes)
        start = random.choice(all_nodes)
        self.stack = [(start, start)]
        self.queued_nodes.add(start)

        self.num_expanded = 0
        self.done = False

        # per-node state
        self.u_node = None
        self.u_pos = None
        self.px_u = None
        self.canon = None
        self.remaining_set = None

        # collect step losses (tensors) for node
        self.node_step_losses = []

        # for associating step with STOP
        self._last_step_is_stop = False

    def _begin_next_node(self) -> bool:
        # stack empty -> reseed from unexpanded node
        if not self.stack:
            remaining = [n for n in self.G.nodes if n not in self.expanded_nodes]
            if not remaining:
                self.done = True
                return False
            new_seed = random.choice(remaining)
            self.stack.append((new_seed, new_seed))
            self.queued_nodes.add(new_seed)

        u_node, u_pos = self.stack.pop()

        if u_node in self.expanded_nodes:
            return self._begin_next_node()

        if self.teacher_noise > 0 and self.mode == "train":
            u_pos = apply_noise(u_node, scale=self.teacher_noise)

        self.u_node = u_node
        self.u_pos = u_pos
        self.px_u = (int(u_pos[0] * self.W), int((1 - u_pos[1]) * self.H))

        self.expanded_nodes.add(u_node)
        self.num_expanded += 1

        neighbors = list(self.G.neighbors(u_node))
        remaining_neighbors = [n for n in neighbors if frozenset({u_node, n}) not in self.generated_edges]

        self.canon = sorted(
            remaining_neighbors,
            key=lambda n: (self.encoder.get_action_label(u_node, n), float(n[0]), float(n[1])),
        )
        self.remaining_set = set(self.canon)
        self.node_step_losses = []
        return True

    def _current_expert(self):
        if not self.remaining_set:
            return None, int(self.CONFIG["stop_class_idx"])
        # canonical smallest remaining
        for n in self.canon:
            if n in self.remaining_set:
                return n, int(self.encoder.get_action_label(self.u_pos, n))
        # should not happen, but safe
        return None, int(self.CONFIG["stop_class_idx"])

    def make_step_example(self):
        """
        Returns:
          x_np: float32 (4,H,W) in [0,1]
          y: int
          is_stop: bool
        or None if graph is done.
        """
        if self.done:
            return None

        if self.num_expanded >= int(self.CONFIG["max_steps"]):
            self.done = True
            return None

        if self.u_node is None or self.remaining_set is None:
            ok = self._begin_next_node()
            if not ok:
                return None

        expert_target, y = self._current_expert()
        is_stop = (expert_target is None)

        # build input from CURRENT canvas state (dot at crop center)
        can_crop = crop_image_fast(self.canvas, self.u_pos, self.CONFIG["crop_size"], self.CONFIG["output_size"])
        ch, cw = can_crop.shape[:2]
        cv2.circle(can_crop, (cw // 2, ch // 2), radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)

        sat_crop = crop_image_fast(self.sat, self.u_pos, self.CONFIG["crop_size"], self.CONFIG["output_size"])

        inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)  # H,W,4
        x = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)        # 4,H,W

        # advance teacher state AFTER producing input
        if is_stop:
            # finish node after this STOP step is scored
            self._last_step_is_stop = True
        else:
            v = expert_target
            px_v = (int(v[0] * self.W), int((1 - v[1]) * self.H))
            cv2.line(self.canvas, self.px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)

            self.generated_edges.add(frozenset({self.u_node, v}))
            self.remaining_set.remove(v)

            # push neighbor for future expansion (restores traversal)
            if v not in self.expanded_nodes and v not in self.queued_nodes:
                self.stack.append((v, v))
                self.queued_nodes.add(v)

            self._last_step_is_stop = False

        return x, y, is_stop

    def record_step_loss(self, loss_step: torch.Tensor):
        """
        Called by trainer after computing loss for the step.
        If the step completed a node (STOP), returns the node_loss tensor (mean of step losses).
        Else returns None.
        """
        self.node_step_losses.append(loss_step)

        if self._last_step_is_stop:
            node_loss = torch.stack(self.node_step_losses).mean()

            # reset node state for next call
            self.u_node = None
            self.u_pos = None
            self.px_u = None
            self.canon = None
            self.remaining_set = None
            self.node_step_losses = []
            self._last_step_is_stop = False

            return node_loss

        return None
