import os
import cv2
import json
import time
import pickle
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from functools import lru_cache
from shapely.ops import unary_union
from shapely.geometry import LineString

from model import PathwayModel


# ============================================================
# Repro / BN helpers
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()
        m.requires_grad_(False)

def should_freeze_bn(model_name: str) -> bool:
    return model_name.lower().startswith("resnet")


# ============================================================
# Cached loaders (per-process cache)
# ============================================================
@lru_cache(maxsize=256)
def _load_graph_and_sat(data_root: str, folder_idx: str, sat_name: str = "satellite.png"):
    path = os.path.join(data_root, str(folder_idx))
    with open(os.path.join(path, "graph.pickle"), "rb") as f:
        G = pickle.load(f)

    img_path = os.path.join(path, sat_name)
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return G, sat

def _mean_std_4(device):
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)
    return mean, std


# ============================================================
# Cropping
# ============================================================
def crop_image(img_arr, center, crop_size=0.5, output_size=(256, 256)):
    H, W = img_arr.shape[:2]
    cx_norm, cy_norm = center

    px_cx = int(cx_norm * W)
    px_cy = int((1 - cy_norm) * H)

    px_half = int((crop_size * W) / 2)

    x1, y1 = px_cx - px_half, px_cy - px_half
    x2, y2 = px_cx + px_half, px_cy + px_half

    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)

    x1_v, y1_v = max(0, x1), max(0, y1)
    x2_v, y2_v = min(W, x2), min(H, y2)

    valid_crop = img_arr[y1_v:y2_v, x1_v:x2_v]

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        pads = ((pad_t, pad_b), (pad_l, pad_r))
        if len(img_arr.shape) == 3:
            pads += ((0, 0),)
        valid_crop = np.pad(valid_crop, pads, mode="constant", constant_values=0)

    return cv2.resize(valid_crop, output_size, interpolation=cv2.INTER_LINEAR)


# ============================================================
# Label Encoder
# ============================================================
class IterativeEncoder:
    def __init__(self, config):
        self.num_angles = config["num_angles"]
        self.stop_idx = config["stop_class_idx"]
        self.dist_bins = np.array([0.0, 0.02, 0.04, 0.06, 0.08,
                                   0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
        self.num_dists = len(self.dist_bins) - 1
        self.angle_step = 360.0 / self.num_angles
        self.angle_offset = -(self.angle_step / 2.0)

    def decode_action(self, action_idx):
        if action_idx == self.stop_idx:
            return None

        a_idx = action_idx // self.num_dists
        d_idx = action_idx % self.num_dists

        angle_deg = (a_idx * self.angle_step) + self.angle_offset + (self.angle_step / 2.0)
        bin_min = self.dist_bins[d_idx]
        bin_max = self.dist_bins[d_idx + 1]
        dist_val = (bin_min + bin_max) / 2.0

        angle_rad = np.radians(angle_deg)
        dx = dist_val * np.cos(angle_rad)
        dy = dist_val * np.sin(angle_rad)
        return dx, dy


# ============================================================
# Buffered IoU reward + caching
# ============================================================
GT_BUF_CACHE = {}
GT_EDGES_CACHE = {}

def gt_graph_to_edges_xy(G):
    return [(tuple(u), tuple(v)) for (u, v) in G.edges]

def buffered_union_from_edges(edges_xy, buffer_radius):
    lines = []
    for (u, v) in edges_xy:
        if u is None or v is None:
            continue
        if (u[0] == v[0]) and (u[1] == v[1]):
            continue
        lines.append(LineString([u, v]))
    if not lines:
        return None
    merged = unary_union(lines)
    return merged.buffer(buffer_radius)

def get_gt_buf_cached(folder_idx, G_gt, buffer_radius):
    key = (str(folder_idx), float(buffer_radius))
    if key in GT_BUF_CACHE:
        return GT_BUF_CACHE[key]

    gt_edges_xy = GT_EDGES_CACHE.get(str(folder_idx))
    if gt_edges_xy is None:
        gt_edges_xy = gt_graph_to_edges_xy(G_gt)
        GT_EDGES_CACHE[str(folder_idx)] = gt_edges_xy

    gt_buf = buffered_union_from_edges(gt_edges_xy, buffer_radius)
    GT_BUF_CACHE[key] = gt_buf
    return gt_buf

def overlap_stats_from_unions(gt_buf, pred_buf):
    if gt_buf is None and pred_buf is None:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0,
                "gt_area": 0.0, "pred_area": 0.0, "inter_area": 0.0, "union_area": 0.0}
    if gt_buf is None:
        pred_area = float(pred_buf.area) if pred_buf is not None else 0.0
        return {"iou": 0.0, "precision": 0.0, "recall": 1.0,
                "gt_area": 0.0, "pred_area": pred_area, "inter_area": 0.0, "union_area": pred_area}
    if pred_buf is None:
        gt_area = float(gt_buf.area) if gt_buf is not None else 0.0
        return {"iou": 0.0, "precision": 1.0, "recall": 0.0,
                "gt_area": gt_area, "pred_area": 0.0, "inter_area": 0.0, "union_area": gt_area}

    inter = float(gt_buf.intersection(pred_buf).area)
    union = float(gt_buf.union(pred_buf).area)
    gt_area = float(gt_buf.area)
    pred_area = float(pred_buf.area)

    iou = inter / (union + 1e-12)
    precision = inter / (pred_area + 1e-12)
    recall = inter / (gt_area + 1e-12)

    return {"iou": float(iou), "precision": float(precision), "recall": float(recall),
            "gt_area": gt_area, "pred_area": pred_area, "inter_area": inter, "union_area": union}

def shaped_reward_option_a(stats, fp_lambda: float, eps: float = 1e-12):
    gt_area = float(stats.get("gt_area", 0.0))
    pred_area = float(stats.get("pred_area", 0.0))
    inter_area = float(stats.get("inter_area", 0.0))
    fp_area = max(0.0, pred_area - inter_area)
    fp_ratio = fp_area / (gt_area + eps) if gt_area > 0 else fp_area
    reward = float(stats["iou"]) - float(fp_lambda) * float(fp_ratio)
    return reward, fp_area, fp_ratio


# ============================================================
# Rollout env helpers
# ============================================================
def node_to_px(xy, H, W):
    x, y = xy
    px = int(np.clip(x * (W - 1), 0, W - 1))
    py = int(np.clip((1 - y) * (H - 1), 0, H - 1))
    return (px, py)

def build_input_tensor(satellite_img, canvas, u_pos, CONFIG, device, mean4, std4):
    H, W = satellite_img.shape[:2]
    px_u = node_to_px(u_pos, H, W)

    canvas_with_dot = canvas.copy()
    cv2.circle(canvas_with_dot, px_u, radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)

    sat_crop = crop_image(satellite_img, u_pos, CONFIG["crop_size"], CONFIG["output_size"])
    can_crop = crop_image(canvas_with_dot, u_pos, CONFIG["crop_size"], CONFIG["output_size"])

    inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)
    x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)
    x = (x - mean4) / std4
    return x

def choose_seed_positions(strategy, num_seeds, rng, gt_nodes=None):
    if strategy == "random":
        return [(rng.random(), rng.random()) for _ in range(num_seeds)]
    if strategy == "grid":
        k = int(np.ceil(np.sqrt(num_seeds)))
        xs = np.linspace(0.1, 0.9, k)
        ys = np.linspace(0.1, 0.9, k)
        pts = [(float(x), float(y)) for y in ys for x in xs]
        rng.shuffle(pts)
        return pts[:num_seeds]
    if strategy == "gt_random":
        if gt_nodes is None or len(gt_nodes) == 0:
            raise ValueError("gt_nodes must be provided for seed_strategy='gt_random'")
        return [tuple(rng.choice(gt_nodes)) for _ in range(num_seeds)]
    raise ValueError(f"Unknown seed_strategy: {strategy}")


# ============================================================
# Rollout (NO GRAD) with step rewards (ΔIoU)
# ============================================================
@torch.no_grad()
def rollout_collect_actions(
    *,
    satellite_img, H, W,
    policy_model,
    encoder,
    CONFIG,
    mean4, std4,
    seed_strategy="random",
    seeds_per_graph=1,
    rng_seed=0,
    gt_nodes=None,
    max_global_steps=2000,
    max_expand_per_node=4,
    topk=5,
    temperature=1.0,
    stop_prob_thresh=0.5,
    step_advantages=False,
    gt_buf=None,
    buffer_radius=0.01,
):
    device = CONFIG["device"]
    rng = random.Random(rng_seed)

    canvas = np.zeros((H, W), dtype=np.uint8)
    nodes_xy, edges_xy, actions = [], [], []
    stack, expanded = [], set()
    steps_used = 0

    stop_idx = int(CONFIG["stop_class_idx"])

    step_rewards = []
    prev_iou = 0.0
    pending_action_idxs = []
    stride = max(1, int(CONFIG.get("step_iou_stride", 1)))
    max_edges = int(CONFIG.get("step_iou_max_edges", 10**9))

    starts = choose_seed_positions(seed_strategy, seeds_per_graph, rng, gt_nodes=gt_nodes)
    for p in starts:
        p = (float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1)))
        nodes_xy.append(p)
        stack.append(len(nodes_xy) - 1)

    while stack and steps_used < max_global_steps:
        u_id = stack.pop()
        if u_id in expanded:
            continue
        expanded.add(u_id)

        u_pos = nodes_xy[u_id]
        px_u = node_to_px(u_pos, H, W)
        used_actions = set()

        for _k in range(max_expand_per_node):
            if steps_used >= max_global_steps:
                break

            x = build_input_tensor(satellite_img, canvas, u_pos, CONFIG, device, mean4, std4)
            logits = policy_model(x)[0]

            probs = torch.softmax(logits, dim=0)
            stop_prob = float(probs[stop_idx].item())

            logits_s = logits / max(float(temperature), 1e-8)

            if stop_prob < stop_prob_thresh:
                logits_s = logits_s.clone()
                logits_s[stop_idx] = -1e9

            if used_actions:
                logits_s = logits_s.clone()
                logits_s[list(used_actions)] = -1e9

            if torch.max(logits_s).item() <= -1e8:
                break

            if topk == 1:
                a_idx = int(torch.argmax(logits_s).item())
            else:
                if 0 < topk < logits_s.numel():
                    topk_vals, topk_idx = torch.topk(logits_s, k=int(topk), largest=True)
                    probs_k = torch.softmax(topk_vals, dim=0)
                    pick = torch.multinomial(probs_k, 1).item()
                    a_idx = int(topk_idx[pick].item())
                else:
                    probs_full = torch.softmax(logits_s, dim=0)
                    a_idx = int(torch.multinomial(probs_full, 1).item())

            actions.append(a_idx)
            if step_advantages:
                step_rewards.append(0.0)

            if a_idx == stop_idx:
                break

            used_actions.add(a_idx)

            deltas = encoder.decode_action(a_idx)
            if deltas is None:
                break

            dx, dy = deltas
            new_pos = (float(np.clip(u_pos[0] + dx, 0, 1)),
                       float(np.clip(u_pos[1] + dy, 0, 1)))
            if new_pos == u_pos:
                break

            nodes_xy.append(new_pos)
            v_id = len(nodes_xy) - 1
            stack.append(v_id)

            px_v = node_to_px(new_pos, H, W)
            cv2.line(canvas, px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)
            edges_xy.append((u_pos, new_pos))
            steps_used += 1

            if step_advantages and (gt_buf is not None) and (len(edges_xy) <= max_edges):
                pending_action_idxs.append(len(actions) - 1)
                if (len(edges_xy) % stride) == 0:
                    pred_buf_t = buffered_union_from_edges(edges_xy, buffer_radius)
                    iou_t = overlap_stats_from_unions(gt_buf, pred_buf_t)["iou"]
                    d_iou = float(iou_t) - float(prev_iou)
                    per = d_iou / float(len(pending_action_idxs))
                    for j in pending_action_idxs:
                        step_rewards[j] += per
                    pending_action_idxs.clear()
                    prev_iou = float(iou_t)

    if step_advantages and (gt_buf is not None) and pending_action_idxs:
        pred_buf_t = buffered_union_from_edges(edges_xy, buffer_radius)
        iou_t = overlap_stats_from_unions(gt_buf, pred_buf_t)["iou"]
        d_iou = float(iou_t) - float(prev_iou)
        per = d_iou / float(len(pending_action_idxs))
        for j in pending_action_idxs:
            step_rewards[j] += per
        pending_action_idxs.clear()
        prev_iou = float(iou_t)

    out = {"edges_xy": edges_xy, "actions": actions}
    if step_advantages:
        assert len(step_rewards) == len(actions), (len(step_rewards), len(actions))
        out["step_rewards"] = step_rewards
    return out


# ============================================================
# DDP helpers (sync once per optimizer step)
# ============================================================
def _is_ddp(model) -> bool:
    return isinstance(model, DDP)

def ddp_set_sync(model, sync: bool):
    if hasattr(model, "require_backward_grad_sync"):
        model.require_backward_grad_sync = bool(sync)

def ddp_sync_grads_with_dummy_backward(model):
    if not _is_ddp(model):
        return
    ddp_set_sync(model, True)
    dummy = None
    for p in model.parameters():
        if p.requires_grad:
            term = p.sum() * 0.0
            dummy = term if dummy is None else (dummy + term)
    if dummy is not None:
        dummy.backward()


# ============================================================
# Replay with grad (per-step advantages)
# ============================================================
def replay_actions_backward(
    *,
    satellite_img,
    H, W,
    policy_model,
    ref_model,
    encoder,
    CONFIG,
    mean4, std4,
    actions,
    advantages_seq,
    kl_beta: float,
    seed_strategy="random",
    seeds_per_graph=1,
    rng_seed=0,
    gt_nodes=None,
    max_global_steps=256,
    max_expand_per_node=4,
    temperature=1.0,
    stop_prob_thresh=0.5,
):
    if len(advantages_seq) != len(actions):
        raise ValueError(f"advantages_seq len {len(advantages_seq)} != actions len {len(actions)}")

    device = CONFIG["device"]
    rng = random.Random(rng_seed)

    ddp_set_sync(policy_model, False)

    canvas = np.zeros((H, W), dtype=np.uint8)
    nodes_xy = []
    stack = []
    expanded = set()
    steps_used = 0

    starts = choose_seed_positions(seed_strategy, seeds_per_graph, rng, gt_nodes=gt_nodes)
    for p in starts:
        p = (float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1)))
        nodes_xy.append(p)
        stack.append(len(nodes_xy) - 1)

    T = max(1, int(len(actions)))
    invT = 1.0 / float(T)

    act_ptr = 0

    kl_raw_sum = 0.0
    entropy_sum = 0.0
    stop_prob_sum = 0.0
    action_count = 0
    stop_count = 0
    stop_idx = int(CONFIG["stop_class_idx"])

    pg_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_loss_sum = 0.0

    while stack and steps_used < max_global_steps and act_ptr < len(actions):
        u_id = stack.pop()
        if u_id in expanded:
            continue
        expanded.add(u_id)

        u_pos = nodes_xy[u_id]
        px_u = node_to_px(u_pos, H, W)
        used_actions = set()

        for _k in range(max_expand_per_node):
            if steps_used >= max_global_steps or act_ptr >= len(actions):
                break

            x = build_input_tensor(satellite_img, canvas, u_pos, CONFIG, device, mean4, std4)
            logits = policy_model(x)[0]
            with torch.no_grad():
                logits_ref = ref_model(x)[0]

            with torch.no_grad():
                probs = torch.softmax(logits, dim=0)
                stop_prob_sum += float(probs[stop_idx].item())

            a_idx = int(actions[act_ptr])
            A_t = float(advantages_seq[act_ptr])
            act_ptr += 1

            logp_all = F.log_softmax(logits, dim=0)
            p_all = torch.exp(logp_all)
            with torch.no_grad():
                logp_ref_all = F.log_softmax(logits_ref, dim=0)

            kl_raw = torch.sum(p_all * (logp_all - logp_ref_all))
            entropy = -torch.sum(p_all * logp_all)
            logp_a = logp_all[a_idx]

            pg_term = (-A_t * invT) * logp_a
            kl_term = (float(kl_beta) * invT) * kl_raw
            loss_step = pg_term + kl_term
            loss_step.backward()

            action_count += 1
            pg_loss_sum += float(pg_term.detach().item())
            kl_loss_sum += float(kl_term.detach().item())
            total_loss_sum += float(loss_step.detach().item())

            kl_raw_sum += float(kl_raw.detach().item())
            entropy_sum += float(entropy.detach().item())

            if a_idx == stop_idx:
                stop_count += 1
                break

            used_actions.add(a_idx)

            deltas = encoder.decode_action(a_idx)
            if deltas is None:
                break

            dx, dy = deltas
            new_pos = (float(np.clip(u_pos[0] + dx, 0, 1)),
                       float(np.clip(u_pos[1] + dy, 0, 1)))
            if new_pos == u_pos:
                break

            nodes_xy.append(new_pos)
            v_id = len(nodes_xy) - 1
            stack.append(v_id)

            px_v = node_to_px(new_pos, H, W)
            cv2.line(canvas, px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)
            steps_used += 1

    mean_kl_raw = (kl_raw_sum / action_count) if action_count > 0 else 0.0
    mean_entropy = (entropy_sum / action_count) if action_count > 0 else 0.0
    mean_stop_prob = (stop_prob_sum / action_count) if action_count > 0 else 0.0
    stop_rate = (float(stop_count) / float(action_count)) if action_count > 0 else 0.0

    return {
        "pg_loss": float(pg_loss_sum),
        "kl_loss": float(kl_loss_sum),
        "total_loss": float(total_loss_sum),
        "kl_raw_mean": float(mean_kl_raw),
        "entropy_mean": float(mean_entropy),
        "stop_prob_mean": float(mean_stop_prob),
        "actions": int(action_count),
        "stop_count": int(stop_count),
        "stop_rate": float(stop_rate),
        "steps_used": int(steps_used),
    }


# ============================================================
# DDP sharding helpers
# ============================================================
def pad_to_world_size(items, world_size):
    r = len(items) % world_size
    if r == 0:
        return items
    pad = world_size - r
    return items + items[:pad]

def shard_list(items, rank, world_size):
    return items[rank::world_size]

def allreduce_mean(x: float, device):
    if not dist.is_available() or not dist.is_initialized():
        return x
    t = torch.tensor([x, 1.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t[0] / t[1]).item())


# ============================================================
# Main GRPO Training Loop (DDP-compatible)
# Multiplicative shaping + within-rollout centered local signal
# ============================================================
def grpo_train_main(CONFIG, ddp_rank=0, ddp_world_size=1):
    set_seed(CONFIG["seed"] + ddp_rank)
    device = CONFIG["device"]
    use_ddp = dist.is_available() and dist.is_initialized()

    with open(CONFIG["train_index"], "r") as f:
        train_all = [line.strip() for line in f]
    with open(CONFIG["val_index"], "r") as f:
        val_indices = [line.strip() for line in f]

    if ddp_rank == 0:
        run_name = f"{CONFIG['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(CONFIG["save_path"], run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)
    else:
        run_dir = None

    if use_ddp:
        obj = [run_dir]
        dist.broadcast_object_list(obj, src=0)
        run_dir = obj[0]

    best_path = os.path.join(run_dir, "model_best.pth")
    last_path = os.path.join(run_dir, "model_last.pth")

    policy = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=CONFIG["model"]).to(device)
    ref    = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=CONFIG["model"]).to(device)

    ckpt = torch.load(CONFIG["init_ckpt"], map_location=device)
    policy.load_state_dict(ckpt)
    ref.load_state_dict(ckpt)

    ref.eval()
    if should_freeze_bn(CONFIG["model"]):
        ref.apply(freeze_bn)
    for p in ref.parameters():
        p.requires_grad_(False)

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                     broadcast_buffers=False, find_unused_parameters=False)

    if should_freeze_bn(CONFIG["model"]):
        policy.apply(freeze_bn)

    encoder = IterativeEncoder(CONFIG)
    optimizer = optim.AdamW(policy.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    mean4, std4 = _mean_std_4(device)

    # shaping params
    alpha = float(CONFIG.get("shape_alpha", 0.2))
    local_clip = float(CONFIG.get("local_clip", 2.0))
    local_std_floor = float(CONFIG.get("local_std_floor", 1e-2))
    mult_min = float(CONFIG.get("mult_min", 0.0))
    mult_max = float(CONFIG.get("mult_max", 2.0))
    global_adv_clip = float(CONFIG.get("global_adv_clip", 5.0))

    history = {"train_iou": [], "val_iou": [], "train_reward": [], "train_kl_raw_mean": [], "train_entropy_mean": []}
    best_val_iou = -1.0

    step_advantages = bool(CONFIG.get("step_advantages", False))
    if not step_advantages and ddp_rank == 0:
        print("[WARN] step_advantages=False: local shaping will be all zeros. Set step_advantages=True.")

    for epoch in range(CONFIG["num_epochs"]):
        rng = random.Random(CONFIG["seed"] + epoch)
        shuffled = train_all[:]
        rng.shuffle(shuffled)

        if CONFIG["graphs_per_epoch"] is not None:
            shuffled = shuffled[: CONFIG["graphs_per_epoch"]]

        if use_ddp:
            shuffled = pad_to_world_size(shuffled, ddp_world_size)
            train_indices = shard_list(shuffled, ddp_rank, ddp_world_size)
        else:
            train_indices = shuffled

        policy.train()
        if should_freeze_bn(CONFIG["model"]):
            policy.apply(freeze_bn)

        pbar = tqdm(train_indices, desc=f"GRPO global+local {epoch+1}/{CONFIG['num_epochs']} (rank0)") if ddp_rank == 0 else train_indices

        optimizer.zero_grad(set_to_none=True)
        update_count = 0
        e_iou, e_reward, e_klraw, e_ent = [], [], [], []

        for folder_idx in pbar:
            G_gt, sat = _load_graph_and_sat(CONFIG["data_root"], str(folder_idx), sat_name="satellite.png")
            gt_nodes = list(G_gt.nodes)
            H, W = sat.shape[:2]
            gt_buf = get_gt_buf_cached(folder_idx, G_gt, CONFIG["buffer_radius"])

            # ---- rollouts (no grad) ----
            policy.eval()
            if should_freeze_bn(CONFIG["model"]):
                policy.apply(freeze_bn)

            rollouts, rollout_stats, rewards = [], [], []

            for k in range(CONFIG["K"]):
                traj = rollout_collect_actions(
                    satellite_img=sat, H=H, W=W,
                    policy_model=policy,
                    encoder=encoder,
                    CONFIG=CONFIG,
                    mean4=mean4, std4=std4,
                    seed_strategy=CONFIG["seed_strategy"],
                    seeds_per_graph=CONFIG["seeds_per_graph"],
                    rng_seed=(CONFIG["seed"] + epoch * 100000 + k * 1000 + int(folder_idx)),
                    gt_nodes=gt_nodes,
                    max_global_steps=CONFIG["max_global_steps"],
                    max_expand_per_node=CONFIG["max_expand_per_node"],
                    topk=CONFIG["topk"],
                    temperature=CONFIG["temperature"],
                    stop_prob_thresh=CONFIG["stop_prob_thresh"],
                    step_advantages=step_advantages,
                    gt_buf=gt_buf,
                    buffer_radius=CONFIG["buffer_radius"],
                )

                pred_buf = buffered_union_from_edges(traj["edges_xy"], CONFIG["buffer_radius"])
                stats = overlap_stats_from_unions(gt_buf, pred_buf)
                reward, _, _ = shaped_reward_option_a(stats, CONFIG["fp_lambda"])

                rollouts.append(traj)
                rollout_stats.append(stats)
                rewards.append(float(reward))

            rewards_np = np.asarray(rewards, dtype=np.float32)
            r_mean = float(rewards_np.mean())
            r_std = float(rewards_np.std()) + 1e-8
            A_global = (rewards_np - r_mean) / r_std
            A_global = np.clip(A_global, -global_adv_clip, global_adv_clip)

            # ---- build per-rollout per-step advantages via multiplicative shaping ----
            advantages_list = []
            for i, out in enumerate(rollouts):
                actions = out["actions"]
                T = len(actions)
                if T == 0:
                    advantages_list.append(np.zeros((0,), dtype=np.float32))
                    continue

                # local step rewards
                if step_advantages:
                    r = out.get("step_rewards", None)
                    if r is None:
                        raise RuntimeError("Missing step_rewards in rollout output.")
                    if len(r) != T:
                        raise RuntimeError(f"step_rewards len {len(r)} != actions len {T}")
                    r = np.asarray(r, dtype=np.float32)
                else:
                    r = np.zeros((T,), dtype=np.float32)

                # within-rollout centering + scaling
                sd = float(r.std())
                sd = max(sd, local_std_floor)

                r_norm = r / (sd + 1e-8)
                r_norm = np.clip(r_norm, -local_clip, local_clip)

                mult = 1.0 + alpha * r_norm
                mult = np.clip(mult, mult_min, mult_max)

                A_seq = float(A_global[i]) * mult
                advantages_list.append(A_seq.astype(np.float32))

            # ---- DEBUG (rank0): print one graph / one rollout ----
            if ddp_rank == 0 and bool(CONFIG.get("debug_one_graph", False)) and (epoch == 0):
                # only print for the first training graph processed on rank0
                # (use update_count==0 BEFORE you increment it; so check len(e_iou)==0)
                if len(e_iou) == 0:
                    k_dbg = int(CONFIG.get("debug_print_rollout", 0))
                    k_dbg = max(0, min(k_dbg, len(rollouts) - 1))

                    out_dbg = rollouts[k_dbg]
                    T_dbg = len(out_dbg["actions"])
                    r_dbg = np.asarray(out_dbg.get("step_rewards", np.zeros(T_dbg, dtype=np.float32)), dtype=np.float32)

                    # local stats
                    mu = float(r_dbg.mean()) if T_dbg > 0 else 0.0
                    sd = float(r_dbg.std()) if T_dbg > 0 else 0.0
                    sd_eff = max(sd, float(CONFIG.get("local_std_floor", 1e-2)))

                    r_norm = r_dbg / (sd_eff + 1e-8)
                    local_clip = float(CONFIG.get("local_clip", 2.0))
                    r_norm_clip = np.clip(r_norm, -local_clip, local_clip)

                    alpha = float(CONFIG.get("shape_alpha", 0.2))
                    mult = 1.0 + alpha * r_norm_clip
                    mult_min = float(CONFIG.get("mult_min", 0.0))
                    mult_max = float(CONFIG.get("mult_max", 2.0))
                    mult_clip = np.clip(mult, mult_min, mult_max)

                    A_g = float(A_global[k_dbg])
                    A_seq = np.asarray(advantages_list[k_dbg], dtype=np.float32)

                    # telescoping sanity (only guaranteed if rewards cover all edges + gt_buf not None)
                    final_iou = float(np.mean([s["iou"] for s in rollout_stats]))  # mean IoU across K (not per rollout)
                    # per-rollout final IoU:
                    pred_buf_dbg = buffered_union_from_edges(out_dbg["edges_xy"], CONFIG["buffer_radius"])
                    stats_dbg = overlap_stats_from_unions(gt_buf, pred_buf_dbg)
                    final_iou_dbg = float(stats_dbg["iou"])
                    sum_step_dbg = float(r_dbg.sum())

                    print("\n[DEBUG global+local]")
                    print(f"  graph(folder_idx)={folder_idx}  rollout=k{k_dbg}  T={T_dbg}")
                    print(f"  final_iou(rollout)={final_iou_dbg:.6f}  sum(step_rewards)={sum_step_dbg:.6f}  diff={final_iou_dbg - sum_step_dbg:.6f}")
                    print(f"  A_global(k)={A_g:.4f}   rewards across K: min/mean/max=({float(rewards_np.min()):.4f},{float(rewards_np.mean()):.4f},{float(rewards_np.max()):.4f})")
                    print(f"  step_rewards: min/mean/max=({float(r_dbg.min() if T_dbg else 0):.6f},{float(r_dbg.mean() if T_dbg else 0):.6f},{float(r_dbg.max() if T_dbg else 0):.6f})")
                    print(f"  local mean={mu:.6f}  local std={sd:.6f}  std_eff={sd_eff:.6f}")
                    print(f"  r_norm_clip: min/mean/max=({float(r_norm_clip.min() if T_dbg else 0):.3f},{float(r_norm_clip.mean() if T_dbg else 0):.3f},{float(r_norm_clip.max() if T_dbg else 0):.3f})")
                    print(f"  mult_clip:   min/mean/max=({float(mult_clip.min() if T_dbg else 1):.3f},{float(mult_clip.mean() if T_dbg else 1):.3f},{float(mult_clip.max() if T_dbg else 1):.3f})")
                    print(f"  A_seq:       min/mean/max=({float(A_seq.min() if T_dbg else 0):.3f},{float(A_seq.mean() if T_dbg else 0):.3f},{float(A_seq.max() if T_dbg else 0):.3f})")
                    print("")
        
            # ---- replay with grad (DDP sync OFF) ----
            policy.eval()
            if should_freeze_bn(CONFIG["model"]):
                policy.apply(freeze_bn)

            replay_kls, replay_ents = [], []
            for k, out in enumerate(rollouts):
                rep = replay_actions_backward(
                    satellite_img=sat, H=H, W=W,
                    policy_model=policy,
                    ref_model=ref,
                    encoder=encoder,
                    CONFIG=CONFIG,
                    mean4=mean4, std4=std4,
                    actions=out["actions"],
                    advantages_seq=advantages_list[k],
                    kl_beta=CONFIG["kl_beta"],
                    seed_strategy=CONFIG["seed_strategy"],
                    seeds_per_graph=CONFIG["seeds_per_graph"],
                    rng_seed=(CONFIG["seed"] + epoch * 100000 + k * 1000 + int(folder_idx)),
                    gt_nodes=gt_nodes,
                    max_global_steps=CONFIG["max_global_steps"],
                    max_expand_per_node=CONFIG["max_expand_per_node"],
                    temperature=CONFIG["temperature"],
                    stop_prob_thresh=CONFIG["stop_prob_thresh"],
                )
                replay_kls.append(rep["kl_raw_mean"])
                replay_ents.append(rep["entropy_mean"])

            policy.train()
            if should_freeze_bn(CONFIG["model"]):
                policy.apply(freeze_bn)

            g_iou = float(np.mean([s["iou"] for s in rollout_stats]))
            g_reward = float(rewards_np.mean())
            g_klraw = float(np.mean(replay_kls)) if replay_kls else 0.0
            g_ent = float(np.mean(replay_ents)) if replay_ents else 0.0

            e_iou.append(g_iou); e_reward.append(g_reward); e_klraw.append(g_klraw); e_ent.append(g_ent)

            update_count += 1
            if update_count % CONFIG["grad_accum"] == 0:
                ddp_sync_grads_with_dummy_backward(policy)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=CONFIG["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if ddp_rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    "IoU": f"{np.mean(e_iou):.3f}",
                    "R": f"{np.mean(e_reward):.3f}",
                    "KL": f"{np.mean(e_klraw):.3f}",
                    "H": f"{np.mean(e_ent):.2f}",
                })

        # flush remainder
        if update_count % CONFIG["grad_accum"] != 0:
            ddp_sync_grads_with_dummy_backward(policy)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=CONFIG["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # aggregate epoch means across ranks
        train_iou = float(np.mean(e_iou)) if e_iou else 0.0
        train_reward = float(np.mean(e_reward)) if e_reward else 0.0
        train_klraw = float(np.mean(e_klraw)) if e_klraw else 0.0
        train_ent = float(np.mean(e_ent)) if e_ent else 0.0

        train_iou = allreduce_mean(train_iou, device)
        train_reward = allreduce_mean(train_reward, device)
        train_klraw = allreduce_mean(train_klraw, device)
        train_ent = allreduce_mean(train_ent, device)

        if ddp_rank == 0:
            history["train_iou"].append(train_iou)
            history["train_reward"].append(train_reward)
            history["train_kl_raw_mean"].append(train_klraw)
            history["train_entropy_mean"].append(train_ent)

        if use_ddp:
            dist.barrier()

        # validation only on rank0
        if ddp_rank == 0:
            policy.eval()
            if should_freeze_bn(CONFIG["model"]):
                policy.apply(freeze_bn)

            val_subset = val_indices[: min(len(val_indices), CONFIG["val_graphs"])]
            v_iou = []
            for folder_idx in val_subset:
                G_gt, sat = _load_graph_and_sat(CONFIG["data_root"], str(folder_idx), sat_name="satellite.png")
                H, W = sat.shape[:2]
                gt_buf = get_gt_buf_cached(folder_idx, G_gt, CONFIG["buffer_radius"])

                traj = rollout_collect_actions(
                    satellite_img=sat, H=H, W=W,
                    policy_model=policy,
                    encoder=encoder,
                    CONFIG=CONFIG,
                    mean4=mean4, std4=std4,
                    seed_strategy=CONFIG["seed_strategy"],
                    seeds_per_graph=CONFIG["seeds_per_graph"],
                    rng_seed=(CONFIG["seed"] + epoch * 999 + int(folder_idx)),
                    gt_nodes=list(G_gt.nodes),
                    max_global_steps=CONFIG["max_global_steps"],
                    max_expand_per_node=CONFIG["max_expand_per_node"],
                    topk=CONFIG["topk"],
                    temperature=CONFIG["temperature"],
                    stop_prob_thresh=CONFIG["stop_prob_thresh"],
                    step_advantages=False,
                    gt_buf=None,
                    buffer_radius=CONFIG["buffer_radius"],
                )

                pred_buf = buffered_union_from_edges(traj["edges_xy"], CONFIG["buffer_radius"])
                stats = overlap_stats_from_unions(gt_buf, pred_buf)
                v_iou.append(float(stats["iou"]))

            avg_val_iou = float(np.mean(v_iou)) if v_iou else 0.0
            history["val_iou"].append(avg_val_iou)

            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            state = policy.module.state_dict() if _is_ddp(policy) else policy.state_dict()
            torch.save(state, last_path)

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save(state, best_path)
                print(f"  [+] New Best Model! val_iou={best_val_iou:.4f}")

        if use_ddp:
            dist.barrier()

    if ddp_rank == 0:
        print("GRPO global+local training complete. Saved to:", run_dir)