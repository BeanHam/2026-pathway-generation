import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
import geopandas as gpd
import torch.distributed as dist

from tqdm import tqdm
from datetime import timedelta
from functools import lru_cache
from shapely.geometry import LineString

from model import PathwayModel
from utils import crop_image, IterativeEncoder, should_freeze_bn, freeze_bn
from config import CONFIG


# ============================================================
# DDP setup (for sharded inference)
# ============================================================
def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    # key change: device_id
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank, rank, world_size

def ddp_barrier(local_rank: int):
    # key change: pass device_ids
    dist.barrier(device_ids=[local_rank])

def ddp_cleanup():
    dist.destroy_process_group()

def shard_list(items, rank, world_size):
    return items[rank::world_size]


# ============================================================
# Cached loaders
# ============================================================
@lru_cache(maxsize=2048)
def load_bbox_cached(bbox_path: str):
    with open(bbox_path, "r") as f:
        b = json.load(f)["bbox"]
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError(f"Unexpected bbox format in {bbox_path}: {b}")
    min_lon, max_lon, min_lat, max_lat = map(float, b)
    return (min_lon, max_lon, min_lat, max_lat)

@lru_cache(maxsize=2048)
def load_graph_cached(graph_path: str):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=2048)
def load_sat_cached(sat_path: str):
    bgr = cv2.imread(sat_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(sat_path)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    target_h, target_w = 2771, 2772
    if sat.shape[0] != target_h or sat.shape[1] != target_w:
        sat = cv2.resize(sat, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return sat

# ============================================================
# Lon/Lat conversion + GeoJSON saving
# ============================================================
def norm_xy_to_lonlat(xy, bbox):
    x, y = float(xy[0]), float(xy[1])
    min_lon, max_lon, min_lat, max_lat = bbox
    lon = min_lon + x * (max_lon - min_lon)
    lat = min_lat + y * (max_lat - min_lat)
    return (lon, lat)

def edges_norm_to_lonlat_gdf(edges_xy, bbox, crs="EPSG:4326"):
    geoms = []
    for (u, v) in edges_xy:
        if u is None or v is None:
            continue
        if (u[0] == v[0]) and (u[1] == v[1]):
            continue
        u_ll = norm_xy_to_lonlat(u, bbox)
        v_ll = norm_xy_to_lonlat(v, bbox)
        geoms.append(LineString([u_ll, v_ll]))
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)

def save_geojson_pair(out_dir: str, idx: str, seed: int, gt_gdf: gpd.GeoDataFrame, pr_gdf: gpd.GeoDataFrame):
    os.makedirs(out_dir, exist_ok=True)
    gt_path = os.path.join(out_dir, f"{idx}_seed{seed}_gt.geojson")
    pr_path = os.path.join(out_dir, f"{idx}_seed{seed}_pred.geojson")
    gt_gdf.to_file(gt_path, driver="GeoJSON")
    pr_gdf.to_file(pr_path, driver="GeoJSON")


# ============================================================
# Model input helpers
# ============================================================
def build_norm_tensors(device):
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)
    return mean, std

def node_to_px(xy, H, W):
    x, y = xy
    px = int(np.clip(x * (W - 1), 0, W - 1))
    py = int(np.clip((1 - y) * (H - 1), 0, H - 1))
    return (px, py)

def build_input_tensor(satellite_img, canvas, u_pos, cfg, device, mean, std):
    H, W = satellite_img.shape[:2]
    px_u = node_to_px(u_pos, H, W)

    canvas_with_dot = canvas.copy()
    cv2.circle(canvas_with_dot, px_u, radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)

    sat_crop = crop_image(satellite_img, u_pos, cfg["crop_size"], cfg["output_size"])
    can_crop = crop_image(canvas_with_dot, u_pos, cfg["crop_size"], cfg["output_size"])
    inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)

    x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)
    x = (x - mean) / std
    return x

def choose_seed_positions(strategy, num_seeds, rng, gt_graph=None):
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

def gt_graph_to_edges_xy(G):
    return [(tuple(u), tuple(v)) for (u, v) in G.edges]


# ============================================================
# Generator (best restart by confidence)
# ============================================================
@torch.inference_mode()
def generate_one_graph(
    *,
    G_gt,
    satellite_img,
    model,
    encoder,
    cfg,
    mean, std,
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
    device = cfg["device"]
    rng = random.Random(rng_seed)

    H, W = satellite_img.shape[:2]
    stop_idx = int(cfg["stop_class_idx"])

    model.eval()
    if should_freeze_bn(cfg["model"]):
        model.apply(freeze_bn)

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

        while stack and steps_used < int(max_global_steps):
            u_id = stack.pop()
            if u_id in expanded:
                continue
            expanded.add(u_id)

            u_pos = nodes_xy[u_id]
            px_u = node_to_px(u_pos, H, W)
            used_actions = set()

            for _ in range(int(max_expand_per_node)):
                if steps_used >= int(max_global_steps):
                    break

                x = build_input_tensor(satellite_img, canvas, u_pos, cfg, device, mean, std)
                logits = model(x)[0]
                probs = torch.softmax(logits, dim=0)
                stop_prob = float(probs[stop_idx].item())

                top2 = torch.topk(logits, k=2).values
                if top2.numel() == 2:
                    confidence += float((top2[0] - top2[1]).item())

                if int(topk) == 1:
                    a0 = int(torch.argmax(logits).item())
                    if a0 == stop_idx and stop_prob < float(stop_prob_thresh):
                        logits_ns = logits.clone()
                        logits_ns[stop_idx] = -1e9
                        cand_actions = [int(torch.argmax(logits_ns).item())]
                    else:
                        cand_actions = [a0]
                else:
                    cand_actions = topk_action_indices(logits, int(topk))

                chosen_action = None
                chosen_end = None

                for a_idx in cand_actions:
                    a_idx = int(a_idx)

                    if prevent_repeat_actions and a_idx in used_actions:
                        continue

                    if a_idx == stop_idx:
                        if stop_prob >= float(stop_prob_thresh):
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

        gen = {"nodes_xy": nodes_xy, "edges_xy": edges_xy, "starts": starts}
        stats = {"confidence": float(confidence)}
        return gen, stats

    best_gen = {"nodes_xy": [], "edges_xy": [], "starts": []}
    best_score = -1e18

    for _ in range(int(restarts_per_graph)):
        starts = choose_seed_positions(seed_strategy, int(seeds_per_graph), rng, gt_graph=G_gt)
        gen, stats = run_once(starts)
        score = float(stats["confidence"])
        if score > best_score:
            best_score = score
            best_gen = gen

    return best_gen


# ============================================================
# Main
# ============================================================
def load_indices(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def load_inference_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    logs = [
        # ------------------------------
        # reference (best checkpoints)
        # ------------------------------
        # '/data/models/bin/2026_pathway_generation/reference_logs/resnet18_20260217_102812/',
        # '/data/models/bin/2026_pathway_generation/reference_logs/vit_small_20260217_102936/',
        # '/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260212_123147_cont/',
        # "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260211_220834_0.2/",
        # "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260213_154110_0.4_cont/",
        # "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260213_234446_0.6_cont/",
        # "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260213_154830_0.8_cont/",
        
        # ------------------------------
        # grpo (best checkpoints)
        # ------------------------------
        # '/data/models/bin/2026_pathway_generation/grpo_logs/resnet18_20260218_230403/',
        # '/data/models/bin/2026_pathway_generation/grpo_logs/vit_small_20260218_230534/',
        # '/data/models/bin/2026_pathway_generation/grpo_logs/swin_tiny_20260218_142801_cont/',
        # "/data/models/bin/2026_pathway_generation/grpo_logs/swin_tiny_20260214_002621_0.2/",
        # "/data/models/bin/2026_pathway_generation/grpo_logs/swin_tiny_20260214_112255_0.4/",
        # "/data/models/bin/2026_pathway_generation/grpo_logs/swin_tiny_20260214_112349_0.6/",
        # "/data/models/bin/2026_pathway_generation/grpo_logs/swin_tiny_20260215_121343_0.8/",
        '/data/models/bin/2026_pathway_generation/grpo_logs_steps/resnet18_20260220_102056/',
        '/data/models/bin/2026_pathway_generation/grpo_logs_steps/resnet18_20260220_102056/',
    ]
    # models = ["resnet18", "vit_small", "swin_tiny"]
    models = ["resnet18_steps", "resnet18_steps_last"]
    
    datasets = [
        {
            "name": "generation_full_data",
            "data_root": "/data/aiclinic/pathways/generation_full_data/pairs/",
            "test_txt":  "/data/aiclinic/pathways/generation_full_data/test.txt",
            "out_root":  "/data/models/bin/2026_pathway_generation/predictions/",
        },
        {
            "name": "pathwaybench_seattle",
            "data_root": "/data/aiclinic/pathways/pathwaybench_seattle/pairs/",
            "test_txt":  "/data/aiclinic/pathways/pathwaybench_seattle/test.txt",
            "out_root":  "/data/models/bin/2026_pathway_generation/predictions_seattle/",
        }
    ]

    use_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank, rank, world_size = ddp_setup()
        ddp_barrier(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = "cuda:0"

    for ds in datasets:
        out_root = ds["out_root"]
        if rank == 0:
            os.makedirs(out_root, exist_ok=True)
            print("\n" + "=" * 80)
            print(f"Dataset: {ds['name']}")
            print(f"  test_txt : {ds['test_txt']}")
            print(f"  data_root: {ds['data_root']}")
            print(f"  out_root : {out_root}")
            print("=" * 80)

        if use_ddp:
            ddp_barrier(local_rank)

        # IMPORTANT: set dataset-specific root before processing indices
        CONFIG["data_root"] = ds["data_root"]

        test_indices = load_indices(ds["test_txt"])
        my_indices = shard_list(test_indices, rank, world_size)

        for run_dir, model_name in zip(logs, models):
            tag = "grpo" if "grpo_logs" in run_dir else "base"
            infer_cfg = load_inference_config(os.path.join(run_dir, "inference_config.json"))

            # model config
            CONFIG["device"] = device
            CONFIG["model"] = model_name
            CONFIG["output_size"] = (224, 224) if model_name.startswith("vit") else (256, 256)

            encoder = IterativeEncoder(CONFIG)
            mean, std = build_norm_tensors(device)

            if rank == 0:
                print("-" * 50)
                print("Model:", model_name, "| tag:", tag, "| world_size:", world_size)
                print("-" * 50)

            # load model weights once per (run_dir, model_name)            
            if model_name in ['swin_tiny_20', 'swin_tiny_40', 'swin_tiny_60', 'swin_tiny_80']:
                m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name='swin_tiny').to(device)
                m.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth"), map_location=device))
                m.eval()
            elif model_name == "resnet18_steps":
                m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name="resnet18").to(device)
                m.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth"), map_location=device))
                m.eval()
            elif model_name == "resnet18_steps_last":
                m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name="resnet18").to(device)
                m.load_state_dict(torch.load(os.path.join(run_dir, "model_last.pth"), map_location=device))
                m.eval()
            # elif '_last' in model_name:
            #     m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=model_name.replace('_last', '')).to(device)
            #     m.load_state_dict(torch.load(os.path.join(run_dir, "model_last.pth"), map_location=device))
            #     m.eval()
            else:
                m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=model_name).to(device)
                m.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth"), map_location=device))
                m.eval()
            if should_freeze_bn(model_name):
                m.apply(freeze_bn)

            # seeds
            for seed in range(1):
                out_dir = os.path.join(out_root, model_name, tag, f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)

                iterable = my_indices
                if use_ddp and rank != 0:
                    iterator = iterable
                else:
                    iterator = tqdm(iterable, desc=f"{ds['name']} rank{rank} {model_name} {tag} seed{seed}")

                for idx in iterator:
                    graph_dir = os.path.join(CONFIG["data_root"], str(idx))

                    bbox = load_bbox_cached(os.path.join(graph_dir, "bbox.json"))
                    G_gt = load_graph_cached(os.path.join(graph_dir, "graph.pickle"))
                    sat = load_sat_cached(os.path.join(graph_dir, "satellite.png"))

                    gt_edges_xy = gt_graph_to_edges_xy(G_gt)

                    per_graph_seed = int(seed) + int(idx)

                    gen = generate_one_graph(
                        G_gt=G_gt,
                        satellite_img=sat,
                        model=m,
                        encoder=encoder,
                        cfg=CONFIG,
                        mean=mean,
                        std=std,
                        seed_strategy=infer_cfg.get("seed_strategy", "gt_random"),
                        seeds_per_graph=int(infer_cfg.get("seeds_per_graph", 1)),
                        restarts_per_graph=int(infer_cfg.get("restarts_per_graph", 1)),
                        topk=int(infer_cfg.get("topk", 1)),
                        stop_prob_thresh=float(infer_cfg.get("stop_prob_thresh", 0.5)),
                        max_expand_per_node=int(infer_cfg.get("max_expand_per_node", 4)),
                        max_global_steps=int(infer_cfg.get("max_global_steps", 200)),
                        rng_seed=per_graph_seed,
                    )

                    pred_edges_xy = gen.get("edges_xy", [])

                    gt_gdf = edges_norm_to_lonlat_gdf(gt_edges_xy, bbox, crs="EPSG:4326")
                    pr_gdf = edges_norm_to_lonlat_gdf(pred_edges_xy, bbox, crs="EPSG:4326")
                    save_geojson_pair(out_dir, str(idx), seed, gt_gdf, pr_gdf)

            if use_ddp:
                ddp_barrier(local_rank)

        if use_ddp:
            ddp_barrier(local_rank)

    if use_ddp:
        ddp_barrier(local_rank)
        ddp_cleanup()

    if rank == 0:
        print("Done. Saved GeoJSONs under dataset-specific out_root folders.")