import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
import pandas as pd
import geopandas as gpd

import itertools
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from shapely.ops import unary_union
from shapely.geometry import LineString
from utils import *
from model import *

# ----------------------------
# Basic helpers
# ----------------------------
def node_to_px(xy, H, W):
    x, y = xy
    return (int(x * W), int((1 - y) * H))

def build_input_tensor(satellite_img, canvas, u_pos, CONFIG, device):
    """Build [1,4,256,256] input exactly like training."""
    H, W = satellite_img.shape[:2]
    px_u = node_to_px(u_pos, H, W)

    canvas_with_dot = canvas.copy()
    cv2.circle(canvas_with_dot, px_u, radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)

    sat_crop = crop_image(satellite_img, u_pos, CONFIG["crop_size"], CONFIG["output_size"])
    can_crop = crop_image(canvas_with_dot, u_pos, CONFIG["crop_size"], CONFIG["output_size"])

    inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)  # (256,256,4)

    x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)
    x = (x - mean) / std
    return x

def choose_seed_positions(strategy, satellite_img, num_seeds, rng, gt_graph=None):
    """
    Realistic options: "random" or "grid".
    Debug only: "gt_random" (uses gt_graph).
    """
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
# 2) Unconstrained inference (deployment-style)
# ----------------------------
@torch.no_grad()
def infer_one_graph_unconstrained(
    folder_idx,
    model,
    encoder,
    CONFIG,
    device=None,
    # seeding
    seed_strategy="gt_random",
    seeds_per_graph=1,
    restarts_per_graph=1,
    topk=5,
    stop_prob_thresh=0.5,

    # fixed params
    stop_fallback_thresh=0.2,
    prevent_repeat_actions=True,
    rng_seed=0,
    max_global_steps=200,
    max_expand_per_node=12,
):
    """
    Unconstrained rollout (no snapping/min_step):
      - decoding:
          * topk=1 => greedy with STOP-veto fallback (matches GRPO greedy):
              if argmax is STOP but stop_prob < stop_prob_thresh, pick best non-STOP
          * topk>1 => "first valid among top-k" with STOP gating
      - restarts: choose best run by confidence (top1-top2 margin sum)
    Returns:
      gen = {"nodes_xy": [...], "edges_xy": [...], "edges_ij": [...]}
    """
    if device is None:
        device = CONFIG["device"]

    rng = random.Random(rng_seed)
    path = os.path.join(CONFIG["data_root"], str(folder_idx))
    with open(os.path.join(path, "graph.pickle"), "rb") as f:
        G_gt = pickle.load(f)
    sat = np.array(Image.open(os.path.join(path, "satellite.png")).convert("RGB"))
    H, W = sat.shape[:2]

    model.eval()
    if should_freeze_bn(CONFIG["model"]):
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

        # seed nodes (no snapping; just add)
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

                x = build_input_tensor(sat, canvas, u_pos, CONFIG, device)
                logits = model(x)[0]  # [361]
                probs = torch.softmax(logits, dim=0)
                stop_prob = float(probs[CONFIG["stop_class_idx"]].item())

                # confidence margin (top1-top2 logits)
                top2 = torch.topk(logits, k=2).values
                if top2.numel() == 2:
                    confidence += float((top2[0] - top2[1]).item())

                # --- Candidate actions ---
                if topk == 1:
                    # greedy with STOP-veto fallback
                    a0 = int(torch.argmax(logits).item())
                    if a0 == CONFIG["stop_class_idx"] and stop_prob < stop_prob_thresh:
                        logits_ns = logits.clone()
                        logits_ns[CONFIG["stop_class_idx"]] = -1e9
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

                    # STOP gating (for topk>1; for topk==1 it's already vetoed above)
                    if a_idx == CONFIG["stop_class_idx"]:
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

                # fallback: stop if nothing valid
                if chosen_action is None:
                    if stop_prob >= stop_fallback_thresh:
                        break
                    else:
                        break

                if chosen_action == CONFIG["stop_class_idx"]:
                    break

                if prevent_repeat_actions:
                    used_actions.add(chosen_action)

                # commit: always create a new node
                nodes_xy.append(chosen_end)
                v_id = len(nodes_xy) - 1
                stack.append(v_id)

                # add edge
                e = frozenset({u_id, v_id})
                if e not in edges_set:
                    edges_set.add(e)
                    v_pos = nodes_xy[v_id]
                    px_v = node_to_px(v_pos, H, W)
                    cv2.line(canvas, px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)
                    edges_xy.append((u_pos, v_pos))

                steps_used += 1

        gen = {
            "nodes_xy": nodes_xy,
            "edges_xy": edges_xy,
            "edges_ij": [tuple(e) for e in edges_set],
            "starts": starts,
        }
        stats = {
            "edges": len(edges_xy),
            "nodes": len(nodes_xy),
            "confidence": float(confidence),
            "steps": int(steps_used),
        }
        return gen, stats

    # restarts: choose best by confidence
    best_gen = {"nodes_xy": [], "edges_xy": [], "edges_ij": [], "starts": []}
    best_score = -1e18

    for _ in range(restarts_per_graph):
        starts = choose_seed_positions(seed_strategy, sat, seeds_per_graph, rng, gt_graph=G_gt)
        gen, stats = run_once(starts)
        score = float(stats["confidence"])
        if score > best_score:
            best_score = score
            best_gen = gen
        
    return best_gen

def edges_xy_to_gdf(edges_xy, crs=None):
    geoms = []
    for (u, v) in edges_xy:
        if u is None or v is None:
            continue
        if (u[0] == v[0]) and (u[1] == v[1]):
            continue
        geoms.append(LineString([u, v]))
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)

def gt_graph_to_edges_xy(G):
    return [(tuple(u), tuple(v)) for (u, v) in G.edges]

def buffered_overlap_stats(
    gt_edges_xy,
    pred_edges_xy,
    *,
    buffer_radius=0.01,
    cap_style=2,
    join_style=2
):
    """
    Buffered geometry overlap stats:
      precision = inter_area / pred_area
      recall    = inter_area / gt_area
      iou       = inter_area / union_area

    Added logging fields:
      - gt_edges, pred_edges
      - gt_area, pred_area, inter_area, union_area
    All areas are in normalized coordinate units^2.
    """
    gt_edges = len(gt_edges_xy)
    pred_edges = len(pred_edges_xy)

    gt_gdf = edges_xy_to_gdf(gt_edges_xy)
    pr_gdf = edges_xy_to_gdf(pred_edges_xy)

    # Both empty
    if len(gt_gdf) == 0 and len(pr_gdf) == 0:
        return {
            "iou": 1.0, "precision": 1.0, "recall": 1.0,
            "gt_edges": gt_edges, "pred_edges": pred_edges,
            "gt_area": 0.0, "pred_area": 0.0, "inter_area": 0.0, "union_area": 0.0
        }

    # GT empty, pred non-empty
    if len(gt_gdf) == 0:
        pr_buf = unary_union(pr_gdf.geometry.buffer(buffer_radius, cap_style=cap_style, join_style=join_style))
        pr_area = float(pr_buf.area)
        return {
            "iou": 0.0, "precision": 0.0, "recall": 1.0,
            "gt_edges": gt_edges, "pred_edges": pred_edges,
            "gt_area": 0.0, "pred_area": pr_area, "inter_area": 0.0, "union_area": pr_area
        }

    # Pred empty, GT non-empty
    if len(pr_gdf) == 0:
        gt_buf = unary_union(gt_gdf.geometry.buffer(buffer_radius, cap_style=cap_style, join_style=join_style))
        gt_area = float(gt_buf.area)
        return {
            "iou": 0.0, "precision": 1.0, "recall": 0.0,
            "gt_edges": gt_edges, "pred_edges": pred_edges,
            "gt_area": gt_area, "pred_area": 0.0, "inter_area": 0.0, "union_area": gt_area
        }

    # Both non-empty
    gt_buf = unary_union(gt_gdf.geometry.buffer(buffer_radius, cap_style=cap_style, join_style=join_style))
    pr_buf = unary_union(pr_gdf.geometry.buffer(buffer_radius, cap_style=cap_style, join_style=join_style))

    inter_area = float(gt_buf.intersection(pr_buf).area)
    union_area = float(gt_buf.union(pr_buf).area)
    gt_area = float(gt_buf.area)
    pred_area = float(pr_buf.area)

    precision = inter_area / (pred_area + 1e-12)
    recall    = inter_area / (gt_area + 1e-12)
    iou       = inter_area / (union_area + 1e-12)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_edges": gt_edges,
        "pred_edges": pred_edges,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "inter_area": inter_area,
        "union_area": union_area,
    }

@torch.no_grad()
def eval_on_split(
    indices,
    CONFIG,
    model,
    encoder,
    infer_one_graph_unconstrained,   # your rollout function returning {"edges_xy": ...}
    *,
    # inference hyperparams to sweep
    rng_seed=0,
    buffer_radius=0.01,
    seed_strategy="grid",
    seeds_per_graph=1,
    restarts_per_graph=3,
    topk=5,
    stop_prob_thresh=0.5,
    max_expand_per_node=4,    
):
    """
    Returns mean IoU across graphs (buffered geometry IoU).
    """
    ious = []
    recalls = []
    precisions = []
    gt_edges = []
    pred_edges = []
    gt_area = []
    pred_area = []
    inter_area = []
    union_area = []
        
    for idx in tqdm(indices, desc="Eval buffered IoU"):
        path = os.path.join(CONFIG["data_root"], str(idx))
        with open(os.path.join(path, "graph.pickle"), "rb") as f:
            G_gt = pickle.load(f)

        gt_edges_xy = gt_graph_to_edges_xy(G_gt)
        per_graph_seed = int(rng_seed) + int(idx)

        gen = infer_one_graph_unconstrained(
            folder_idx=str(idx),
            model=model,
            encoder=encoder,
            CONFIG=CONFIG,
            seed_strategy=seed_strategy,
            seeds_per_graph=seeds_per_graph,
            restarts_per_graph=restarts_per_graph,
            rng_seed=per_graph_seed,
            topk=topk,
            stop_prob_thresh=stop_prob_thresh,
            max_expand_per_node=max_expand_per_node,
        )
        pred_edges_xy = gen.get("edges_xy", [])

        results = buffered_overlap_stats(
            gt_edges_xy, pred_edges_xy,
            buffer_radius=buffer_radius
        ) 
        ious.append(results['iou'])
        precisions.append(results['precision'])
        recalls.append(results['recall'])
        gt_edges.append(results['gt_edges'])
        pred_edges.append(results['pred_edges'])
        gt_area.append(results['gt_area'])
        pred_area.append(results['pred_area'])
        inter_area.append(results['inter_area'])
        union_area.append(results['union_area']) 
    
    return {
        'ious': ious, 
        'precisions': precisions,
        'recalls': recalls,
        'gt_edges': gt_edges,
        'pred_edges': pred_edges,
        'gt_area': gt_area,
        'pred_area': pred_area,
        'inter_area': inter_area,
        'union_area': union_area,
        'mean_iou': float(np.mean(ious)) if len(ious) else 0.0,
        'mean_precision': float(np.mean(precisions)) if len(precisions) else 0.0,
        'mean_recall': float(np.mean(recalls)) if len(recalls) else 0.0,
        'mean_gt_edges': float(np.mean(gt_edges)) if len(gt_edges) else 0.0,
        'mean_pred_edges': float(np.mean(pred_edges)) if len(pred_edges) else 0.0,
        'mean_gt_area': float(np.mean(gt_area)) if len(gt_area) else 0.0,
        'mean_pred_area': float(np.mean(pred_area)) if len(pred_area) else 0.0,
        'mean_inter_area': float(np.mean(inter_area)) if len(inter_area) else 0.0,
        'mean_union_area': float(np.mean(union_area)) if len(union_area) else 0.0,
    }

def tune_inference_hparams(
    val_indices,
    CONFIG,
    model,
    encoder,
    infer_one_graph_unconstrained,
    *,
    # fixed eval settings
    seed_strategy="gt_random",
    seeds_per_graph=1,
    buffer_radius=0.01,
    rng_seed=0,

    # sweep ranges (edit these)
    restarts_grid=(1, 5),
    topk_grid=(1, 5, 10),
    stop_prob_grid=(0.3, 0.4, 0.5, 0.6),
    max_expand_grid=(4, 8, 12),

    # speed controls
    max_graphs=None,          # e.g. 50 for quick tuning; None = full val
    random_subset_seed=123,   # stable subset
    mode="grid",              # "grid" or "random"
    random_trials=25,         # if mode="random"

    # selection
    metric="mean_iou",        # could also use "mean_f1" if you compute it
    verbose=True,
):
    """
    Returns:
      best_cfg: dict
      best_metrics: dict (output of eval_on_split)
      results_df: pandas DataFrame with one row per trial
    """

    # --- choose subset of graphs for speed if requested ---
    indices = list(val_indices)
    if max_graphs is not None and max_graphs < len(indices):
        rs = np.random.RandomState(random_subset_seed)
        indices = [indices[i] for i in rs.choice(len(indices), size=max_graphs, replace=False)]
        if verbose:
            print(f"[tune] Using subset: {len(indices)}/{len(val_indices)} graphs")

    # --- build candidate list ---
    if mode == "grid":
        candidates = list(itertools.product(restarts_grid, topk_grid, stop_prob_grid, max_expand_grid))
    elif mode == "random":
        rs = np.random.RandomState(random_subset_seed)
        candidates = []
        for _ in range(int(random_trials)):
            r = int(rs.choice(restarts_grid))
            k = int(rs.choice(topk_grid))
            s = float(rs.choice(stop_prob_grid))
            m = int(rs.choice(max_expand_grid))
            candidates.append((r, k, s, m))
    else:
        raise ValueError("mode must be 'grid' or 'random'")

    if verbose:
        print(f"[tune] Trials: {len(candidates)} (mode={mode})")

    rows = []
    best_score = -1e18
    best_cfg = None
    best_metrics = None

    for t_idx, (restarts_per_graph, topk, stop_prob_thresh, max_expand_per_node) in enumerate(candidates, 1):
        # Run eval
        metrics_out = eval_on_split(
            indices=indices,
            CONFIG=CONFIG,
            model=model,
            encoder=encoder,
            infer_one_graph_unconstrained=infer_one_graph_unconstrained,
            seed_strategy=seed_strategy,
            seeds_per_graph=seeds_per_graph,
            restarts_per_graph=restarts_per_graph,
            topk=topk,
            stop_prob_thresh=stop_prob_thresh,
            rng_seed=rng_seed,
            buffer_radius=buffer_radius,
            max_expand_per_node=max_expand_per_node,
        )

        score = float(metrics_out.get(metric, 0.0))

        row = {
            "trial": t_idx,
            "restarts_per_graph": restarts_per_graph,
            "topk": topk,
            "stop_prob_thresh": stop_prob_thresh,
            "max_expand_per_node": max_expand_per_node,
            "mean_iou": float(metrics_out.get("mean_iou", 0.0)),
            "mean_precision": float(metrics_out.get("mean_precision", 0.0)),
            "mean_recall": float(metrics_out.get("mean_recall", 0.0)),
            "mean_pred_edges": float(metrics_out.get("mean_pred_edges", 0.0)),
            "mean_gt_edges": float(metrics_out.get("mean_gt_edges", 0.0)),
        }
        rows.append(row)

        # best tracking (tie-breakers: higher recall, then precision)
        if (score > best_score) or (
            abs(score - best_score) < 1e-12 and row["mean_recall"] > (best_metrics.get("mean_recall", -1.0) if best_metrics else -1.0)
        ):
            best_score = score
            best_cfg = {
                "seed_strategy": seed_strategy,
                "seeds_per_graph": seeds_per_graph,
                "restarts_per_graph": restarts_per_graph,
                "topk": topk,
                "stop_prob_thresh": stop_prob_thresh,
                "max_expand_per_node": max_expand_per_node,
                "buffer_radius": buffer_radius,
                "rng_seed": rng_seed,
            }
            best_metrics = deepcopy(metrics_out)

        if verbose:
            print(
                f"[tune] {t_idx:03d}/{len(candidates)} "
                f"iou={row['mean_iou']:.4f} "
                f"p={row['mean_precision']:.4f} r={row['mean_recall']:.4f} "
                f"(restarts={restarts_per_graph}, topk={topk}, stop={stop_prob_thresh}, maxexp={max_expand_per_node}) "
                f"| best_iou={best_cfg and best_metrics['mean_iou']:.4f}"
            )

    results_df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False).reset_index(drop=True)
    return best_cfg, best_metrics, results_df


if __name__ == "__main__":
    
    logs = [
        '/data/models/bin/reference_logs/resnet18_20260105_201712/',                          # old model
        # '/data/models/bin/2026_pathway_generation/reference_logs/resnet18_20260203_210148/',  # new model
    ]
        
    path = "/data/aiclinic/pathways/generation_full_data/"
    with open(path+"val.txt", "r") as f:
        val_indices = [line.strip() for line in f]
    
    for i, log in enumerate(logs):
        with open(log+'run_config.json', 'r') as f:
            config = json.load(f)                    
        
        config['data_root']= path+"pairs/"
        
        encoder = IterativeEncoder(config)
        model = PathwayModel(num_classes=config['total_classes'], backbone_name=config["model"]).to(config['device'])
        model.load_state_dict(torch.load(log+"model_best.pth", map_location=config["device"]))
        best_cfg, best_metrics, df = tune_inference_hparams(
            val_indices=val_indices,
            CONFIG=config,
            model=model,
            encoder=encoder,
            infer_one_graph_unconstrained=infer_one_graph_unconstrained,
            seed_strategy="gt_random",
            seeds_per_graph=1,
            buffer_radius=0.01,
            rng_seed=0,
            restarts_grid=(1, 5),
            topk_grid=(5, 10),
            stop_prob_grid=(0.4, 0.5, 0.6, 0.7),
            max_expand_grid=(4, 8, 12),
            max_graphs=200,
            mode="grid",
            verbose=True,
        )
        # with open(log+'inference_config.json', 'w') as f:
        #     json.dump(best_cfg, f, indent=4)        