import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
import geopandas as gpd
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

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
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank, rank, world_size


def ddp_barrier(local_rank: int):
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


def save_iteration_geojson(out_dir: str, idx: str, seed: int, iter_idx: int, bbox, edges_xy):
    os.makedirs(out_dir, exist_ok=True)
    pr_gdf = edges_norm_to_lonlat_gdf(edges_xy, bbox, crs="EPSG:4326")
    out_path = os.path.join(out_dir, f"{idx}_seed{seed}_iter{iter_idx:02d}.geojson")
    pr_gdf.to_file(out_path, driver="GeoJSON")


# ============================================================
# Model input helpers
# ============================================================
def build_norm_tensors(device):
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)
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


def choose_seed_positions(strategy, num_seeds, rng, gt_graph=None, starting_points=None):
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
    if strategy == "segmentation":
        if not starting_points:
            raise ValueError("No segmentation seeds available.")
        return [rng.choice(starting_points) for _ in range(num_seeds)]
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
    mean,
    std,
    starting_points,
    seed_strategy="gt_random",
    seeds_per_graph=1,
    restarts_per_graph=1,
    topk=5,
    stop_prob_thresh=0.5,
    prevent_repeat_actions=True,
    rng_seed=0,
    max_global_steps=200,
    max_expand_per_node=12,
    reject_on_overlap=False,
    accepted_cover_mask=None,
    overlap_line_width=2,
    overlap_buffer_radius_px=108,
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

                if reject_on_overlap and edge_buffered_overlap_exists(
                    u_pos,
                    chosen_end,
                    accepted_cover_mask=accepted_cover_mask,
                    image_shape=(H, W),
                    line_width=overlap_line_width,
                    buffer_radius_px=overlap_buffer_radius_px,
                ):
                    gen = {"nodes_xy": nodes_xy, "edges_xy": [], "starts": starts, "rejected_overlap": True}
                    stats = {"confidence": float(confidence)}
                    return gen, stats

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

        gen = {"nodes_xy": nodes_xy, "edges_xy": edges_xy, "starts": starts, "rejected_overlap": False}
        stats = {"confidence": float(confidence)}
        return gen, stats

    best_gen = {"nodes_xy": [], "edges_xy": [], "starts": []}
    best_score = -1e18
    best_stats = {"confidence": -1e18}

    for _ in range(int(restarts_per_graph)):
        starts = choose_seed_positions(
            seed_strategy,
            int(seeds_per_graph),
            rng,
            gt_graph=G_gt,
            starting_points=starting_points,
        )
        gen, stats = run_once(starts)
        score = float(stats["confidence"])
        if score > best_score:
            best_score = score
            best_gen = gen
            best_stats = stats

    best_gen["confidence"] = float(best_score)
    best_gen["best_restart_stats"] = best_stats
    return best_gen


# ============================================================
# Seed helpers (flat seed pool)
# ============================================================
def unpack_seed_points(seed_entry):
    if not isinstance(seed_entry, dict):
        return []

    points_px = seed_entry.get("points_px", [])
    points_norm = seed_entry.get("points_norm", [])
    scores = seed_entry.get("scores", [])

    n = min(len(points_px), len(points_norm), len(scores))
    out = []
    for i in range(n):
        x_px, y_px = points_px[i]
        x_norm, y_norm = points_norm[i]
        out.append({
            "x_px": int(x_px),
            "y_px_top_left": int(y_px),
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "score": float(scores[i]),
        })
    return out


def seed_points_to_starting_points(seed_points):
    return [[float(p["x_norm"]), float(p["y_norm"])] for p in seed_points]


def remove_seed_points_matching_starts(seed_points, starts_norm, H, W):
    if not starts_norm:
        return seed_points, []

    start_keys = {node_to_px(s, H, W) for s in starts_norm}
    kept = []
    removed = []
    for p in seed_points:
        key = (int(p["x_px"]), int(p["y_px_top_left"]))
        if key in start_keys:
            removed.append(p)
        else:
            kept.append(p)
    return kept, removed


def rasterize_edges_mask_norm(edges_xy, image_shape, line_width=2):
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for u, v in edges_xy:
        if u is None or v is None:
            continue
        pu = node_to_px(u, H, W)
        pv = node_to_px(v, H, W)
        if pu == pv:
            continue
        cv2.line(mask, pu, pv, color=255, thickness=int(line_width), lineType=cv2.LINE_AA)
    return mask


@lru_cache(maxsize=32)
def _get_dilate_kernel(radius: int):
    r = int(radius)
    if r <= 0:
        return None
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def dilate_binary_mask(mask, radius):
    mask = (mask > 0).astype(np.uint8) * 255
    r = int(radius)
    if r <= 0:
        return mask
    kernel = _get_dilate_kernel(r)
    return cv2.dilate(mask, kernel)


def remove_seed_points_covered_by_mask(seed_points, cover_mask):
    H, W = cover_mask.shape[:2]
    if not seed_points:
        return [], []

    xs = np.fromiter((int(p["x_px"]) for p in seed_points), dtype=np.int32, count=len(seed_points))
    ys = np.fromiter((int(p["y_px_top_left"]) for p in seed_points), dtype=np.int32, count=len(seed_points))
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    covered = np.zeros(len(seed_points), dtype=bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        covered[valid_idx] = cover_mask[ys[valid_idx], xs[valid_idx]] > 0

    kept = [p for i, p in enumerate(seed_points) if not covered[i]]
    removed = [p for i, p in enumerate(seed_points) if covered[i]]
    return kept, removed


def dedup_undirected_edges(edges_xy, decimals=6):
    out = []
    seen = set()
    for e in edges_xy:
        if e is None or len(e) != 2:
            continue
        u, v = e
        if u is None or v is None:
            continue

        ku = (round(float(u[0]), decimals), round(float(u[1]), decimals))
        kv = (round(float(v[0]), decimals), round(float(v[1]), decimals))
        if ku == kv:
            continue

        key = (ku, kv) if ku <= kv else (kv, ku)
        if key in seen:
            continue
        seen.add(key)
        out.append(((float(u[0]), float(u[1])), (float(v[0]), float(v[1]))))
    return out


# ============================================================
# Generation-level overlap rejection
# ============================================================
def generation_buffered_overlap_exists(
    raw_edges_xy,
    accepted_cover_mask,
    image_shape,
    line_width=2,
    buffer_radius_px=108,
):
    """
    Aggressive whole-generation rule:
    if the buffered new generation touches the accepted buffered graph at all,
    reject the whole generation.

    Uses a local ROI around the full generation for speed.
    """
    H, W = image_shape
    if len(raw_edges_xy) == 0:
        return False, 0

    if not np.any(accepted_cover_mask):
        return False, 0

    pts = []
    for u, v in raw_edges_xy:
        if u is None or v is None:
            continue
        pu = node_to_px(u, H, W)
        pv = node_to_px(v, H, W)
        pts.append(pu)
        pts.append(pv)

    if not pts:
        return False, 0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = int(buffer_radius_px) + int(max(1, line_width)) + 2

    x0 = max(0, min(xs) - pad)
    x1 = min(W, max(xs) + pad + 1)
    y0 = max(0, min(ys) - pad)
    y1 = min(H, max(ys) + pad + 1)

    if x1 <= x0 or y1 <= y0:
        return False, 0

    roi_accept = accepted_cover_mask[y0:y1, x0:x1]
    if not np.any(roi_accept):
        return False, 0

    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for u, v in raw_edges_xy:
        if u is None or v is None:
            continue
        pu = node_to_px(u, H, W)
        pv = node_to_px(v, H, W)
        p1 = (pu[0] - x0, pu[1] - y0)
        p2 = (pv[0] - x0, pv[1] - y0)
        if p1 == p2:
            continue
        cv2.line(local_mask, p1, p2, color=255, thickness=int(line_width), lineType=cv2.LINE_AA)

    kernel = _get_dilate_kernel(int(buffer_radius_px))
    if kernel is None:
        gen_cover_mask = local_mask
    else:
        gen_cover_mask = cv2.dilate(local_mask, kernel)

    overlap_pixels = int(np.count_nonzero((gen_cover_mask > 0) & (roi_accept > 0)))
    has_overlap = overlap_pixels > 0
    return has_overlap, overlap_pixels


def edge_buffered_overlap_exists(
    u,
    v,
    accepted_cover_mask,
    image_shape,
    line_width=2,
    buffer_radius_px=108,
):
    """Fast per-edge ROI overlap check used for early abort inside rollout."""
    H, W = image_shape
    if accepted_cover_mask is None or not np.any(accepted_cover_mask):
        return False

    pu = node_to_px(u, H, W)
    pv = node_to_px(v, H, W)
    if pu == pv:
        return False

    pad = int(buffer_radius_px) + int(max(1, line_width)) + 2
    x0 = max(0, min(pu[0], pv[0]) - pad)
    x1 = min(W, max(pu[0], pv[0]) + pad + 1)
    y0 = max(0, min(pu[1], pv[1]) - pad)
    y1 = min(H, max(pu[1], pv[1]) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return False

    roi_accept = accepted_cover_mask[y0:y1, x0:x1]
    if not np.any(roi_accept):
        return False

    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    p1 = (pu[0] - x0, pu[1] - y0)
    p2 = (pv[0] - x0, pv[1] - y0)
    cv2.line(local_mask, p1, p2, color=255, thickness=int(line_width), lineType=cv2.LINE_AA)

    kernel = _get_dilate_kernel(int(buffer_radius_px))
    edge_cover = local_mask if kernel is None else cv2.dilate(local_mask, kernel)
    return bool(np.any((edge_cover > 0) & (roi_accept > 0)))


# ============================================================
# Iterative coverage-based generation with overlap pruning
# ============================================================
@torch.inference_mode()
def generate_iterative_coverage(
    *,
    G_gt,
    satellite_img,
    model,
    encoder,
    cfg,
    mean,
    std,
    seed_entry,
    infer_cfg,
    base_rng_seed,
):
    H, W = satellite_img.shape[:2]

    remaining_seed_points = unpack_seed_points(seed_entry)
    accepted_edges_xy = []
    accepted_generations = []
    iteration_debug = []

    # 4m on each side ~= 108 px with ~27 px/m.
    coverage_buffer_radius_px = int(infer_cfg.get("coverage_buffer_radius_px", 108))
    coverage_line_width = int(infer_cfg.get("coverage_line_width", 2))
    min_generated_edges = int(infer_cfg.get("min_generated_edges", 1))
    edge_round_decimals = int(infer_cfg.get("edge_round_decimals", 6))
    max_generations = int(infer_cfg.get("max_generations", max(1, len(remaining_seed_points))))

    accepted_base_mask = np.zeros((H, W), dtype=np.uint8)
    accepted_cover_mask = np.zeros((H, W), dtype=np.uint8)

    iter_idx = 0
    while remaining_seed_points and iter_idx < max_generations:
        removed_precovered = []
        if np.any(accepted_cover_mask):
            remaining_seed_points, removed_precovered = remove_seed_points_covered_by_mask(
                remaining_seed_points,
                accepted_cover_mask,
            )
            if not remaining_seed_points:
                break

        seeds_before = len(remaining_seed_points)
        starting_points = seed_points_to_starting_points(remaining_seed_points)
        iter_rng_seed = int(base_rng_seed) + 100003 * (iter_idx + 1)

        gen = generate_one_graph(
            G_gt=G_gt,
            satellite_img=satellite_img,
            model=model,
            encoder=encoder,
            cfg=cfg,
            mean=mean,
            std=std,
            seed_strategy="segmentation",
            seeds_per_graph=int(infer_cfg.get("seeds_per_graph", 1)),
            restarts_per_graph=int(infer_cfg.get("restarts_per_graph", 1)),
            topk=int(infer_cfg.get("topk", 1)),
            stop_prob_thresh=float(infer_cfg.get("stop_prob_thresh", 0.5)),
            max_expand_per_node=int(infer_cfg.get("max_expand_per_node", 4)),
            max_global_steps=int(infer_cfg.get("max_global_steps", 200)),
            rng_seed=iter_rng_seed,
            starting_points=starting_points,
            reject_on_overlap=bool(np.any(accepted_cover_mask)),
            accepted_cover_mask=accepted_cover_mask,
            overlap_line_width=coverage_line_width,
            overlap_buffer_radius_px=coverage_buffer_radius_px,
        )

        raw_edges_xy = gen.get("edges_xy", [])
        chosen_starts = gen.get("starts", [])
        confidence = float(gen.get("confidence", -1e18))
        early_rejected_overlap = bool(gen.get("rejected_overlap", False))

        remaining_seed_points, removed_start = remove_seed_points_matching_starts(
            remaining_seed_points,
            chosen_starts,
            H,
            W,
        )

        accepted = False
        removed_cover = []
        kept_edges_xy = []
        overlap_pixels = 0
        has_generation_overlap = False

        if len(raw_edges_xy) >= min_generated_edges and not early_rejected_overlap:
            has_generation_overlap, overlap_pixels = generation_buffered_overlap_exists(
                raw_edges_xy=raw_edges_xy,
                accepted_cover_mask=accepted_cover_mask,
                image_shape=(H, W),
                line_width=coverage_line_width,
                buffer_radius_px=coverage_buffer_radius_px,
            )

            if not has_generation_overlap:
                kept_edges_xy = raw_edges_xy
                kept_mask = rasterize_edges_mask_norm(
                    kept_edges_xy,
                    image_shape=(H, W),
                    line_width=coverage_line_width,
                )
                accepted_base_mask = np.maximum(accepted_base_mask, kept_mask)
                accepted_cover_mask = dilate_binary_mask(accepted_base_mask, coverage_buffer_radius_px)

                remaining_seed_points, removed_cover = remove_seed_points_covered_by_mask(
                    remaining_seed_points,
                    accepted_cover_mask,
                )

                accepted_edges_xy.extend(kept_edges_xy)
                accepted_generations.append({
                    "iter_idx": int(iter_idx),
                    "confidence": confidence,
                    "num_pred_edges_raw": int(len(raw_edges_xy)),
                    "num_pred_edges_kept": int(len(kept_edges_xy)),
                    "starts": chosen_starts,
                    "edges_xy": kept_edges_xy,
                })
                accepted = True

        seeds_after = len(remaining_seed_points)
        removed_total = seeds_before - seeds_after

        if removed_total <= 0:
            forced_removed = remaining_seed_points[:1]
            remaining_seed_points = remaining_seed_points[1:]
        else:
            forced_removed = []

        iteration_debug.append({
            "iter_idx": int(iter_idx),
            "confidence": confidence,
            "accepted": bool(accepted),
            "num_pred_edges_raw": int(len(raw_edges_xy)),
            "num_pred_edges_kept": int(len(kept_edges_xy)),
            "num_starts": int(len(chosen_starts)),
            "seeds_before": int(seeds_before),
            "removed_precovered": int(len(removed_precovered)),
            "removed_start": int(len(removed_start)),
            "removed_covered": int(len(removed_cover)),
            "forced_removed": int(len(forced_removed)),
            "seeds_after": int(len(remaining_seed_points)),
            "has_generation_overlap": bool(has_generation_overlap),
            "early_rejected_overlap": bool(early_rejected_overlap),
            "overlap_pixels": int(overlap_pixels),
        })

        iter_idx += 1

    merged_edges_xy = dedup_undirected_edges(accepted_edges_xy, decimals=edge_round_decimals)

    return {
        "edges_xy": merged_edges_xy,
        "iteration_debug": iteration_debug,
        "accepted_generations": accepted_generations,
        "num_initial_seeds": int(len(unpack_seed_points(seed_entry))),
        "num_final_generations": int(len(accepted_generations)),
    }


def load_indices(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def load_inference_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


LOGS = [    
    '/path/to/reference_logs/resnet18/',
    '/path/to/grpo_global_logs/resnet18/',
    '/path/to/grpo_local_logs/resnet18/' 
]
MODELS = ["resnet18"]*3
DATASETS = [
    {
        "name":      "DATASET_NAME",
        "data_root": "/path/to/DATASET_ROOT/pairs/",
        "test_txt":  "/path/to/DATASET_ROOT/test.txt",
        "seed_root": "/path/to/DATASET_ROOT/segmentation_seeds.json",
        "out_root":  "/path/to/save/predictions/",
    },
]

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    use_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank, rank, world_size = ddp_setup()
        ddp_barrier(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = "cuda:0"

    for ds in DATASETS:
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

        CONFIG["data_root"] = ds["data_root"]
        test_indices = load_indices(ds["test_txt"])
        my_indices = shard_list(test_indices, rank, world_size)
        with open(ds["seed_root"], "r") as f:
            all_starting_points = json.load(f)

        for run_dir, model_name in zip(LOGS, MODELS):
            if "grpo_local" in run_dir: tag = "local"
            elif "grpo_global" in run_dir: tag = "global"
            else: tag = "base"

            infer_cfg = load_inference_config(os.path.join(run_dir, "inference_config.json"))
            infer_cfg["coverage_buffer_radius_px"] = 108   # ~4m on each side
            infer_cfg["coverage_line_width"] = 2
            infer_cfg["min_generated_edges"] = 1
            infer_cfg["edge_round_decimals"] = 6
            infer_cfg["max_generations"] = 1000
            infer_cfg["save_iteration_geojsons"] = False

            CONFIG["device"] = device
            CONFIG["model"] = model_name
            CONFIG["output_size"] = (224, 224) if model_name.startswith("vit") else (256, 256)

            encoder = IterativeEncoder(CONFIG)
            mean, std = build_norm_tensors(device)

            if rank == 0:
                print("-" * 50)
                print("Model:", model_name, "| tag:", tag, "| world_size:", world_size)
                print("-" * 50)

            m = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=model_name).to(device)
            m.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth"), map_location=device))
            m.eval()
            if should_freeze_bn(model_name):
                m.apply(freeze_bn)

            for seed in range(5):
                out_dir = os.path.join(out_root, model_name, tag, f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)

                iterable = my_indices if (use_ddp and rank != 0) else tqdm(
                    my_indices,
                    desc=f"{ds['name']} rank{rank} {model_name} {tag} seed{seed}",
                )

                for idx in iterable:
                    graph_dir = os.path.join(CONFIG["data_root"], str(idx))

                    bbox = load_bbox_cached(os.path.join(graph_dir, "bbox.json"))
                    G_gt = load_graph_cached(os.path.join(graph_dir, "graph.pickle"))
                    sat = load_sat_cached(os.path.join(graph_dir, "satellite.png"))
                    gt_edges_xy = gt_graph_to_edges_xy(G_gt)

                    per_graph_seed = int(seed) + int(idx)
                    seed_entry = all_starting_points.get(str(idx), {})

                    if len(seed_entry.get("points_norm", [])) == 0:
                        pred_edges_xy = []
                        iteration_debug = []
                        accepted_generations = []
                    else:
                        gen_all = generate_iterative_coverage(
                            G_gt=G_gt,
                            satellite_img=sat,
                            model=m,
                            encoder=encoder,
                            cfg=CONFIG,
                            mean=mean,
                            std=std,
                            seed_entry=seed_entry,
                            infer_cfg=infer_cfg,
                            base_rng_seed=per_graph_seed,
                        )
                        pred_edges_xy = gen_all["edges_xy"]
                        iteration_debug = gen_all["iteration_debug"]
                        accepted_generations = gen_all["accepted_generations"]

                    debug_path = os.path.join(out_dir, f"{idx}_seed{seed}_iteration_debug.json")
                    with open(debug_path, "w") as f:
                        json.dump(iteration_debug, f, indent=2)

                    if infer_cfg.get("save_iteration_geojsons", True):
                        iter_out_dir = os.path.join(out_dir, "iteration_predictions")
                        for gen_item in accepted_generations:
                            save_iteration_geojson(
                                out_dir=iter_out_dir,
                                idx=str(idx),
                                seed=seed,
                                iter_idx=int(gen_item["iter_idx"]),
                                bbox=bbox,
                                edges_xy=gen_item["edges_xy"],
                            )

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
