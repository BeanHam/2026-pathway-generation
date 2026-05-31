import os
import cv2
import json
import torch
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from functools import lru_cache
from shapely.ops import unary_union
from shapely.geometry import LineString

from utils import crop_image, IterativeEncoder, should_freeze_bn, freeze_bn
from model import PathwayModel


# ----------------------------
# DDP helpers
# ----------------------------
def ddp_setup():
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

def shard_list(items, rank, world_size):
    return items[rank::world_size]

def allreduce_sum_count(sum_val: float, count: int, device):
    """
    Allreduce (sum, count) across ranks.
    """
    t = torch.tensor([sum_val, float(count)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


# ----------------------------
# Basic helpers
# ----------------------------
def node_to_px(xy, H, W):
    x, y = xy
    px = int(np.clip(x * (W - 1), 0, W - 1))
    py = int(np.clip((1 - y) * (H - 1), 0, H - 1))
    return (px, py)

def build_norm_tensors(device):
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)
    return mean, std

def build_input_tensor(satellite_img, canvas, u_pos, CONFIG, device, mean, std):
    H, W = satellite_img.shape[:2]
    px_u = node_to_px(u_pos, H, W)

    canvas_with_dot = canvas.copy()
    cv2.circle(canvas_with_dot, px_u, radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)

    sat_crop = crop_image(satellite_img, u_pos, CONFIG["crop_size"], CONFIG["output_size"])
    can_crop = crop_image(canvas_with_dot, u_pos, CONFIG["crop_size"], CONFIG["output_size"])

    inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)

    x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)
    x = (x - mean) / std
    return x

def choose_seed_positions(strategy, satellite_img, num_seeds, rng, gt_graph=None):
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
        assert gt_graph is not None
        nodes = list(gt_graph.nodes)
        return [rng.choice(nodes) for _ in range(num_seeds)]
    raise ValueError(f"Unknown seed strategy: {strategy}")

def topk_action_indices(logits_1d, k):
    k = min(int(k), int(logits_1d.numel()))
    return torch.topk(logits_1d, k=k, largest=True).indices.tolist()


# ----------------------------
# Cached loaders
# ----------------------------
@lru_cache(maxsize=256)
def load_gt_graph(data_root: str, idx: str):
    path = os.path.join(data_root, str(idx))
    with open(os.path.join(path, "graph.pickle"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=256)
def load_sat_rgb(data_root: str, idx: str, sat_name: str = "satellite.png"):
    path = os.path.join(data_root, str(idx))
    img_path = os.path.join(path, sat_name)
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def gt_graph_to_edges_xy(G):
    return [(tuple(u), tuple(v)) for (u, v) in G.edges]


# ----------------------------
# Inference (same logic)
#   IMPORTANT: build mean/std ONCE per process (per trial calls re-use)
# ----------------------------
@torch.no_grad()
def generate_one_graph(
    folder_idx,
    model,
    encoder,
    CONFIG,
    mean,
    std,
    device=None,
    seed_strategy="gt_random",
    seeds_per_graph=1,
    restarts_per_graph=1,
    topk=5,
    stop_prob_thresh=0.5,
    prevent_repeat_actions=True,
    rng_seed=0,
    max_global_steps=200,
    max_expand_per_node=12,
):
    if device is None:
        device = CONFIG["device"]

    rng = random.Random(rng_seed)
    data_root = CONFIG["data_root"]

    G_gt = load_gt_graph(data_root, str(folder_idx))
    sat = load_sat_rgb(data_root, str(folder_idx), sat_name="satellite.png")
    H, W = sat.shape[:2]

    model.eval()
    if should_freeze_bn(CONFIG["model"]):
        model.apply(freeze_bn)

    stop_idx = int(CONFIG["stop_class_idx"])

    def run_once(starts):
        canvas = np.zeros((H, W), dtype=np.uint8)
        nodes_xy = []
        edges_set = set()
        edges_xy = []
        stack = []
        expanded = set()
        confidence = 0.0
        steps_used = 0

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

                x = build_input_tensor(sat, canvas, u_pos, CONFIG, device, mean, std)
                logits = model(x)[0]
                probs = torch.softmax(logits, dim=0)
                stop_prob = float(probs[stop_idx].item())

                top2 = torch.topk(logits, k=2).values
                if top2.numel() == 2:
                    confidence += float((top2[0] - top2[1]).item())

                if topk == 1:
                    a0 = int(torch.argmax(logits).item())
                    if a0 == stop_idx and stop_prob < stop_prob_thresh:
                        logits_ns = logits.clone()
                        logits_ns[stop_idx] = -1e9
                        cand_actions = [int(torch.argmax(logits_ns).item())]
                    else:
                        cand_actions = [a0]
                else:
                    cand_actions = topk_action_indices(logits, topk)

                chosen_action = None
                chosen_end = None

                for a_idx in cand_actions:
                    a_idx = int(a_idx)

                    if prevent_repeat_actions and a_idx in used_actions:
                        continue

                    if a_idx == stop_idx:
                        if stop_prob >= stop_prob_thresh:
                            chosen_action = a_idx
                            break
                        else:
                            continue

                    deltas = encoder.decode_action(a_idx)
                    if deltas is None:
                        continue

                    dx, dy = deltas
                    new_pos = (
                        float(np.clip(u_pos[0] + dx, 0, 1)),
                        float(np.clip(u_pos[1] + dy, 0, 1)),
                    )
                    if new_pos == u_pos:
                        continue

                    chosen_action = a_idx
                    chosen_end = new_pos
                    break

                if chosen_action is None or chosen_action == stop_idx:
                    break

                if prevent_repeat_actions:
                    used_actions.add(chosen_action)

                nodes_xy.append(chosen_end)
                v_id = len(nodes_xy) - 1
                stack.append(v_id)

                e = frozenset({u_id, v_id})
                if e not in edges_set:
                    edges_set.add(e)
                    v_pos = nodes_xy[v_id]
                    px_v = node_to_px(v_pos, H, W)
                    cv2.line(canvas, px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)
                    edges_xy.append((u_pos, v_pos))

                steps_used += 1

        return {"edges_xy": edges_xy}, {"confidence": float(confidence)}

    best_gen = {"edges_xy": []}
    best_score = -1e18

    for _ in range(restarts_per_graph):
        starts = choose_seed_positions(seed_strategy, sat, seeds_per_graph, rng, gt_graph=G_gt)
        gen, stats = run_once(starts)
        score = float(stats["confidence"])
        if score > best_score:
            best_score = score
            best_gen = gen

    return best_gen


# ----------------------------
# Shapely buffered IoU (IoU only)
# ----------------------------
def edges_xy_to_lines(edges_xy):
    geoms = []
    for (u, v) in edges_xy:
        if u is None or v is None:
            continue
        if (u[0] == v[0]) and (u[1] == v[1]):
            continue
        geoms.append(LineString([u, v]))
    return geoms

@lru_cache(maxsize=256)
def gt_buffered_union(data_root: str, idx: str, buffer_radius: float, cap_style: int, join_style: int):
    G_gt = load_gt_graph(data_root, str(idx))
    gt_edges_xy = gt_graph_to_edges_xy(G_gt)
    lines = edges_xy_to_lines(gt_edges_xy)
    if not lines:
        return None
    bufs = [ln.buffer(buffer_radius, cap_style=cap_style, join_style=join_style) for ln in lines]
    return unary_union(bufs)

def buffered_iou(gt_buf_union, pred_edges_xy, *, buffer_radius=0.01, cap_style=2, join_style=2):
    pred_lines = edges_xy_to_lines(pred_edges_xy)
    if not pred_lines:
        return 1.0 if gt_buf_union is None else 0.0
    pred_bufs = [ln.buffer(buffer_radius, cap_style=cap_style, join_style=join_style) for ln in pred_lines]
    pr_union = unary_union(pred_bufs)
    if gt_buf_union is None:
        return 0.0
    inter_area = float(gt_buf_union.intersection(pr_union).area)
    union_area = float(gt_buf_union.union(pr_union).area)
    return float(inter_area / (union_area + 1e-12))


# ----------------------------
# DDP Eval: shard indices each trial, allreduce sums
# ----------------------------
@torch.no_grad()
def eval_iou_ddp(
    indices,
    CONFIG,
    model,
    encoder,
    mean,
    std,
    *,
    rng_seed=0,
    buffer_radius=0.01,
    cap_style=2,
    join_style=2,
    seed_strategy="grid",
    seeds_per_graph=1,
    restarts_per_graph=3,
    topk=5,
    stop_prob_thresh=0.5,
    max_expand_per_node=4,
    rank=0,
    world_size=1,
):
    data_root = CONFIG["data_root"]
    device = CONFIG["device"]

    my_indices = shard_list(list(indices), rank, world_size)
    iou_sum_local = 0.0
    n_local = 0

    for idx in my_indices:
        idx = str(idx)
        per_graph_seed = int(rng_seed) + int(idx)

        gen = generate_one_graph(
            folder_idx=idx,
            model=model,
            encoder=encoder,
            CONFIG=CONFIG,
            mean=mean,
            std=std,
            seed_strategy=seed_strategy,
            seeds_per_graph=seeds_per_graph,
            restarts_per_graph=restarts_per_graph,
            rng_seed=per_graph_seed,
            topk=topk,
            stop_prob_thresh=stop_prob_thresh,
            max_expand_per_node=max_expand_per_node,
        )
        pred_edges_xy = gen.get("edges_xy", [])

        gt_union = gt_buffered_union(data_root, idx, float(buffer_radius), int(cap_style), int(join_style))
        iou = buffered_iou(gt_union, pred_edges_xy, buffer_radius=buffer_radius, cap_style=cap_style, join_style=join_style)

        iou_sum_local += float(iou)
        n_local += 1

    if dist.is_initialized():
        iou_sum, n_total = allreduce_sum_count(iou_sum_local, n_local, device)
    else:
        iou_sum, n_total = iou_sum_local, n_local

    mean_iou = float(iou_sum / max(n_total, 1))
    return {"mean_iou": mean_iou, "n_graphs": int(n_total)}


def tune_inference_hparams_ddp(
    val_indices,
    CONFIG,
    model,
    encoder,
    *,
    seed_strategy="gt_random",
    seeds_per_graph=1,
    buffer_radius=0.01,
    cap_style=2,
    join_style=2,
    rng_seed=0,
    restarts_grid=(1, 5),
    topk_grid=(1, 5, 10),
    stop_prob_grid=(0.3, 0.4, 0.5, 0.6),
    max_expand_grid=(4, 8, 12),
    max_graphs=200,
    subset_seed=123,
    rank=0,
    world_size=1,
    verbose=True,
):
    # deterministic subset selection (rank-safe)
    indices = list(val_indices)
    if max_graphs is not None and max_graphs < len(indices):
        rs = np.random.RandomState(subset_seed)
        indices = [indices[i] for i in rs.choice(len(indices), size=max_graphs, replace=False)]

    candidates = list(itertools.product(restarts_grid, topk_grid, stop_prob_grid, max_expand_grid))
    if rank == 0 and verbose:
        print(f"[tune] Trials: {len(candidates)} | graphs: {len(indices)} | world_size={world_size}")

    mean, std = build_norm_tensors(CONFIG["device"])

    rows = []
    best_score = -1e18
    best_cfg = None
    best_metrics = None

    iterator = enumerate(candidates, 1)
    if rank == 0:
        iterator = tqdm(list(iterator), desc="Tuning trials (rank0)")

    for t_idx, (restarts_per_graph, topk, stop_prob_thresh, max_expand_per_node) in iterator:
        metrics_out = eval_iou_ddp(
            indices=indices,
            CONFIG=CONFIG,
            model=model,
            encoder=encoder,
            mean=mean,
            std=std,
            seed_strategy=seed_strategy,
            seeds_per_graph=seeds_per_graph,
            restarts_per_graph=int(restarts_per_graph),
            topk=int(topk),
            stop_prob_thresh=float(stop_prob_thresh),
            rng_seed=rng_seed,
            buffer_radius=buffer_radius,
            cap_style=cap_style,
            join_style=join_style,
            max_expand_per_node=int(max_expand_per_node),
            rank=rank,
            world_size=world_size,
        )
        mean_iou = float(metrics_out["mean_iou"])

        # only rank0 logs/records rows
        if rank == 0:
            rows.append({
                "trial": t_idx,
                "restarts_per_graph": int(restarts_per_graph),
                "topk": int(topk),
                "stop_prob_thresh": float(stop_prob_thresh),
                "max_expand_per_node": int(max_expand_per_node),
                "mean_iou": mean_iou,
            })

            if mean_iou > best_score:
                best_score = mean_iou
                best_cfg = {
                    "seed_strategy": seed_strategy,
                    "seeds_per_graph": int(seeds_per_graph),
                    "restarts_per_graph": int(restarts_per_graph),
                    "topk": int(topk),
                    "stop_prob_thresh": float(stop_prob_thresh),
                    "max_expand_per_node": int(max_expand_per_node),
                    "buffer_radius": float(buffer_radius),
                    "cap_style": int(cap_style),
                    "join_style": int(join_style),
                    "rng_seed": int(rng_seed),
                    "max_global_steps": int(CONFIG.get("max_global_steps", 200)),
                }
                best_metrics = dict(metrics_out)

            if verbose:
                print(
                    f"[tune] {t_idx:03d}/{len(candidates)} "
                    f"iou={mean_iou:.4f} "
                    f"(restarts={restarts_per_graph}, topk={topk}, stop={stop_prob_thresh}, maxexp={max_expand_per_node}) "
                    f"| best_iou={best_score:.4f}"
                )

        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

    if rank == 0:
        df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False).reset_index(drop=True)
        return best_cfg, best_metrics, df
    return None, None, None




LOGS = ['/path/to/reference_logs/resnet18/']
VAL_PATH = '/path/to/DATASET_ROOT/val.txt'
RESTART_GRID=(1, 5),
TOPK_GRID=(5, 10),
STOP_PROB_GRID=(0.4, 0.5, 0.6, 0.7),
MAX_EXPAND_GRID=(4, 6, 8),

# ----------------------------
# Main (torchrun)
# ----------------------------
if __name__ == "__main__":
    use_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank, rank, world_size = ddp_setup()
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = "cuda:0"

    with open(VAL_PATH, "r") as f:
        val_indices = [line.strip() for line in f if line.strip()]

    for log in LOGS:
        # load run config
        with open(os.path.join(log, "run_config.json"), "r") as f:
            config = json.load(f)

        config["device"] = device  # each rank uses its own GPU
        encoder = IterativeEncoder(config)

        model = PathwayModel(num_classes=config["total_classes"], backbone_name=config["model"]).to(device)
        model.load_state_dict(torch.load(os.path.join(log, "model_best.pth"), map_location=device))
        model.eval()
        if should_freeze_bn(config["model"]):
            model.apply(freeze_bn)

        if rank == 0:
            print("\n" + "=" * 80)
            print("Tuning:", log)
            print("device:", device, "| world_size:", world_size)
            print("=" * 80)

        best_cfg, best_metrics, df = tune_inference_hparams_ddp(
            val_indices=val_indices,
            CONFIG=config,
            model=model,
            encoder=encoder,
            seed_strategy="gt_random",
            seeds_per_graph=1,
            buffer_radius=0.01,
            rng_seed=0,
            restarts_grid=RESTART_GRID,
            topk_grid=TOPK_GRID,
            stop_prob_grid=STOP_PROB_GRID,
            max_expand_grid=MAX_EXPAND_GRID,
            max_graphs=200,
            rank=rank,
            world_size=world_size,
            verbose=True,
        )

        if rank == 0:
            with open(os.path.join(log, "inference_config.json"), "w") as f:
                json.dump(best_cfg, f, indent=4)            
            print("[tune] saved:", os.path.join(log, "inference_config.json"))
            
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

    if use_ddp:
        ddp_cleanup()