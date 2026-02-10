import os
import re
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import osmnx as ox

from tqdm import tqdm
from shapely.ops import unary_union

import metrics as M  # your helper functions module

warnings.filterwarnings("ignore")


# ============================
# Helpers
# ============================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def union_geoms(geoms):
    """Shapely 2 has union_all; otherwise unary_union."""
    geoms = list(geoms)
    if not geoms:
        return None
    if hasattr(shapely, "union_all"):
        return shapely.union_all(geoms)
    return unary_union(geoms)

def seed_from_dir(run_dir: str) -> int:
    base = os.path.basename(run_dir.rstrip("/"))
    if base.startswith("seed_"):
        return int(base.split("_")[1])
    raise ValueError(f"Cannot parse seed from directory: {run_dir}")

def read_geojson(path: str):
    gdf = gpd.read_file(path)

    # Ensure there is an active geometry column
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=M.PROJ)
        return gdf

    # Ensure GeoDataFrame has geometry set
    try:
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry("geometry")
    except Exception:
        # if geopandas couldn't interpret geometry, force empty
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=M.PROJ)
        return gdf

    # Ensure CRS is set
    if gdf.crs is None:
        gdf = gdf.set_crs(M.PROJ, allow_override=True)

    return gdf

def buffered_union_geom(gdf: gpd.GeoDataFrame):
    """Union of buffered lines for IoU computation."""
    if gdf is None or len(gdf) == 0:
        return None
    # buffering can return empty geometries; filter them out
    bufs = [geom.buffer(M.BUFF_DIS) for geom in gdf.geometry if geom is not None]
    bufs = [b for b in bufs if (b is not None and not b.is_empty)]
    return union_geoms(bufs)

def buffered_iou_from_unions(gt_buf, pred_buf) -> float:
    if gt_buf is None and pred_buf is None:
        return 1.0
    if gt_buf is None or pred_buf is None:
        return 0.0
    inter = float(pred_buf.intersection(gt_buf).area)
    union = float(pred_buf.union(gt_buf).area)
    return float(inter / union) if union > 0 else 0.0


# ============================
# One-time GT discovery
# ============================
GT_RE = re.compile(r"^(?P<idx>.+)_seed(?P<seed>\d+)_gt\.geojson$")

def build_gt_index(pred_root: str):
    """
    Build mapping:
      idx_str -> any GT geojson path
    We only need one GT file per idx (GT is identical across runs).
    """
    gt_map = {}
    for model_name in os.listdir(pred_root):
        if model_name.startswith("_"):
            continue
        model_dir = os.path.join(pred_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        for tag in ["base", "grpo"]:
            tag_dir = os.path.join(model_dir, tag)
            if not os.path.isdir(tag_dir):
                continue

            for seed_dir in os.listdir(tag_dir):
                if not seed_dir.startswith("seed_"):
                    continue
                run_dir = os.path.join(tag_dir, seed_dir)
                if not os.path.isdir(run_dir):
                    continue

                for fn in os.listdir(run_dir):
                    m = GT_RE.match(fn)
                    if not m:
                        continue
                    idx = m.group("idx")
                    if idx not in gt_map:
                        gt_map[idx] = os.path.join(run_dir, fn)
                # early exit if we already found lots; but safe to just continue
    return gt_map


# ============================
# Expensive GT cache building
# ============================
def build_tiles_from_gt(gdf_gt: gpd.GeoDataFrame):
    bbox_poly = M.bounding_box_from_gdf(gdf_gt)

    # osmnx API varies by version
    if hasattr(ox, "graph_from_polygon"):
        G_roads = ox.graph_from_polygon(bbox_poly, simplify=True, retain_all=True)
    else:
        G_roads = ox.graph.graph_from_polygon(bbox_poly, simplify=True, retain_all=True)

    # Try voronoi; if it fails or returns empty, fall back to 1 tile (the bbox)
    try:
        tile_gdf = M.create_voronoi_diagram(G_roads, bbox_poly)

        # Some failures return an object but with no usable geometry
        if tile_gdf is None or len(tile_gdf) == 0:
            raise ValueError("Empty voronoi tile_gdf")

        if "geometry" not in tile_gdf.columns:
            raise ValueError("tile_gdf has no geometry column")

        # Ensure geometry exists and is not entirely empty
        if tile_gdf.geometry.isna().all() or tile_gdf.geometry.is_empty.all():
            raise ValueError("tile_gdf geometry empty")

        return tile_gdf

    except Exception as e:
        # Fallback: one polygon tile = bbox
        # This keeps evaluation running and still produces graph-level metrics.
        tile_gdf = gpd.GeoDataFrame({"geometry": [bbox_poly]}, crs=M.PROJ)
        return tile_gdf

def compute_gt_tile_measures(tile_gdf: gpd.GeoDataFrame, gdf_gt: gpd.GeoDataFrame) -> pd.DataFrame:
    gt_measures = [M.get_measures_from_polygon(geom, gdf_gt, gdf_gt, "gt") for geom in tile_gdf.geometry]
    return pd.DataFrame(gt_measures)

def load_or_build_gt_cache_for_idx(idx: str, gt_path: str, cache_dir: str, verbose=False):
    """
    Cache per idx:
      - gdf_gt (projected)
      - tile_gdf
      - gt_tile_df
      - gt_buf
    """
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{idx}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"[cache] building idx={idx} from {gt_path}")

    gdf_gt = read_geojson(gt_path).to_crs(M.PROJ)

    tile_gdf = build_tiles_from_gt(gdf_gt)
    gt_tile_df = compute_gt_tile_measures(tile_gdf, gdf_gt)
    gt_buf = buffered_union_geom(gdf_gt)

    cache_obj = {
        "idx": str(idx),
        "gdf_gt": gdf_gt,
        "tile_gdf": tile_gdf,
        "gt_tile_df": gt_tile_df,
        "gt_buf": gt_buf,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cache_obj


# ============================
# Evaluation on predictions
# ============================
def compute_pred_tile_measures(tile_gdf: gpd.GeoDataFrame, gdf_pred: gpd.GeoDataFrame, gdf_gt: gpd.GeoDataFrame) -> pd.DataFrame:
    pred_measures = [M.get_measures_from_polygon(geom, gdf_pred, gdf_gt, "pred") for geom in tile_gdf.geometry]
    return pd.DataFrame(pred_measures)

def aggregate_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_val: float) -> dict:
    return {
        "avg_degree": float(M.compute_avg(pred_df, "deg_centrality_avg")),
        "avg_betweenness": float(M.compute_avg(pred_df, "bet_centrality_avg")),
        "avg_num_components": float(M.compute_avg(pred_df, "num_connect_comp_avg")),
        "iou": float(iou_val),
        "avg_f1": float(M.compute_avg(pred_df, "f1")),
        "traversability": float(M.compute_tra_avg(pred_df)),
        "traversability_similarity": float(M.compute_tra_jaccard(pred_df, gt_df)),
    }

def evaluate_one_run(run_dir: str, split_indices: list, gt_map: dict, cache_dir: str, *, verbose=False):
    """
    run_dir: ../predictions/<model>/<base|grpo>/seed_<seed>/
    returns dict with averaged metrics across graphs.
    """
    seed = seed_from_dir(run_dir)
    per_graph = []
    gt_cache_mem = {}

    for idx in tqdm(split_indices, desc=os.path.basename(os.path.dirname(run_dir)) + "/" + os.path.basename(run_dir)):
        if idx not in gt_map:
            if verbose:
                print(f"[WARN] missing GT mapping for idx={idx}")
            continue

        if idx not in gt_cache_mem:
            gt_cache_mem[idx] = load_or_build_gt_cache_for_idx(idx, gt_map[idx], cache_dir, verbose=verbose)
        gt_cache = gt_cache_mem[idx]

        pred_path = os.path.join(run_dir, f"{idx}_seed{seed}_pred.geojson")
        if not os.path.exists(pred_path):
            if verbose:
                print(f"[WARN] missing pred for idx={idx}: {pred_path}")
            continue

        gdf_pred = read_geojson(pred_path).to_crs(M.PROJ)

        pred_buf = buffered_union_geom(gdf_pred)
        iou_val = buffered_iou_from_unions(gt_cache["gt_buf"], pred_buf)

        pred_df = compute_pred_tile_measures(gt_cache["tile_gdf"], gdf_pred, gt_cache["gdf_gt"])
        gt_df = gt_cache["gt_tile_df"]

        per_graph.append(aggregate_metrics(pred_df, gt_df, iou_val))

    if not per_graph:
        return {
            "avg_degree": np.nan,
            "avg_betweenness": np.nan,
            "avg_num_components": np.nan,
            "iou": np.nan,
            "avg_f1": np.nan,
            "traversability": np.nan,
            "traversability_similarity": np.nan,
            "n_graphs": 0,
        }

    out = {k: float(np.mean([d[k] for d in per_graph])) for k in per_graph[0].keys()}
    out["n_graphs"] = int(len(per_graph))
    return out


# ============================
# Main
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datasets = [
        {
            "name": "generation_full_data",
            "pred_root": "/data/models/bin/2026_pathway_generation/predictions/",
            "split": "/data/aiclinic/pathways/generation_full_data/test.txt",
            "out_csv": "/data/models/bin/2026_pathway_generation/predictions/evaluation_results.csv",
        },
        {
            "name": "pathwaybench_seattle",
            "pred_root": "/data/models/bin/2026_pathway_generation/predictions_seattle/",
            "split": "/data/aiclinic/pathways/pathwaybench_seattle/test.txt",
            "out_csv": "/data/models/bin/2026_pathway_generation/predictions_seattle/evaluation_results.csv",
        },
        {
            "name": "pathwaybench_portland",
            "pred_root": "/data/models/bin/2026_pathway_generation/predictions_portland/",
            "split": "/data/aiclinic/pathways/pathwaybench_portland/test.txt",
            "out_csv": "/data/models/bin/2026_pathway_generation/predictions_portland/evaluation_results.csv",
        },
    ]

    seeds = 1

    for ds in datasets:
        ds_name = ds["name"]
        pred_root = ds["pred_root"]
        split_path = ds["split"]
        out_csv = ds["out_csv"]

        if not os.path.isdir(pred_root):
            print(f"[SKIP] pred_root not found: {pred_root}")
            continue
        if not os.path.exists(split_path):
            print(f"[SKIP] split file not found: {split_path}")
            continue

        # Load indices
        with open(split_path, "r") as f:
            split_indices = [line.strip() for line in f if line.strip()]

        # Cache dir is per dataset/pred_root
        cache_dir = os.path.join(pred_root, "_gt_cache")
        ensure_dir(cache_dir)

        # One-time GT discovery per dataset
        gt_map = build_gt_index(pred_root)
        if args.verbose:
            print(f"\n[{ds_name}] gt_map found GT for {len(gt_map)} indices")
            print(f"[{ds_name}] cache_dir: {cache_dir}")

        print("\n" + "=" * 80)
        print(f"Evaluating dataset: {ds_name}")
        print(f"  pred_root: {pred_root}")
        print(f"  split:     {split_path}")
        print(f"  out_csv:   {out_csv}")
        print("=" * 80)

        rows = []

        for model_name in sorted(os.listdir(pred_root)):
            if model_name.startswith("_"):
                continue

            model_dir = os.path.join(pred_root, model_name)
            if not os.path.isdir(model_dir):
                continue

            for tag in ["base", "grpo"]:
                tag_dir = os.path.join(model_dir, tag)
                if not os.path.isdir(tag_dir):
                    continue

                for seed in range(seeds):
                    run_dir = os.path.join(tag_dir, f"seed_{seed}")
                    if not os.path.isdir(run_dir):
                        continue

                    print(f"\n=== [{ds_name}] Evaluating {model_name} [{tag}] seed={seed} ===")
                    metrics_out = evaluate_one_run(
                        run_dir,
                        split_indices,
                        gt_map=gt_map,
                        cache_dir=cache_dir,
                        verbose=args.verbose,
                    )

                    print(f"Avg degree:                 {metrics_out['avg_degree']}")
                    print(f"Avg betweenness:            {metrics_out['avg_betweenness']}")
                    print(f"Avg # connected components: {metrics_out['avg_num_components']}")
                    print(f"Avg IoU:                    {metrics_out['iou']}")
                    print(f"Avg F1:                     {metrics_out['avg_f1']}")
                    print(f"Traversability:             {metrics_out['traversability']}")
                    print(f"Traversability similarity:  {metrics_out['traversability_similarity']}")
                    print(f"Graphs evaluated:           {metrics_out['n_graphs']}")

                    rows.append({
                        "dataset": ds_name,
                        "pred_root": pred_root,
                        "model": model_name,
                        "tag": tag,
                        "seed": seed,
                        **metrics_out,
                    })

        # Write dataset CSV (only this dataset)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\n[{ds_name}] Saved: {out_csv}")
        print(f"[{ds_name}] GT cache stored in: {cache_dir}")

    print("\nDone.")