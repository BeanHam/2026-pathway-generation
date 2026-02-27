import os
import re
import pickle
import argparse
import warnings

import geopandas as gpd
import shapely
import osmnx as ox

from tqdm import tqdm
from shapely.ops import unary_union

import metrics as M  # DO NOT MODIFY THIS FILE

warnings.filterwarnings("ignore")


# ============================
# Helpers
# ============================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def union_geoms(geoms):
    geoms = list(geoms)
    if not geoms:
        return None
    if hasattr(shapely, "union_all"):
        return shapely.union_all(geoms)
    return unary_union(geoms)

def read_geojson(path: str, default_crs: str = "EPSG:4326"):
    """
    If CRS is missing, assume EPSG:4326 (lon/lat). Your GT/preds are lon/lat.
    """
    gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=default_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs, allow_override=True)
    return gdf

def buffered_union_geom_in_proj(gdf_ll: gpd.GeoDataFrame):
    """
    Buffer union in M.PROJ units (meters if epsg:26910).
    """
    if gdf_ll is None or len(gdf_ll) == 0:
        return None

    if gdf_ll.crs is None:
        gdf_ll = gdf_ll.set_crs("EPSG:4326", allow_override=True)

    gdf = gdf_ll.to_crs(M.PROJ)

    bufs = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        b = geom.buffer(M.BUFF_DIS)
        if b is not None and not b.is_empty:
            bufs.append(b)

    return union_geoms(bufs)


# ============================
# Find one GT per idx
# ============================
GT_RE = re.compile(r"^(?P<idx>.+)_seed(?P<seed>\d+)_gt\.geojson$")

def build_gt_index(pred_root: str):
    """
    idx -> any GT geojson path (GT is identical across runs).
    """
    gt_map = {}
    if not os.path.isdir(pred_root):
        return gt_map

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

    return gt_map


# ============================
# Cache builders
# ============================
def build_tiles_from_gt(gdf_gt_ll: gpd.GeoDataFrame, *, verbose: bool = False):
    """
    Build Voronoi tiles for one idx.
    Returns:
      - tile_gdf (GeoDataFrame in M.PROJ) on success
      - None if OSM has no graph nodes or tiling fails (caller should skip idx)
    """
    if gdf_gt_ll.crs is None:
        gdf_gt_ll = gdf_gt_ll.set_crs("EPSG:4326", allow_override=True)
    gdf_ll = gdf_gt_ll.to_crs("EPSG:4326")

    bbox_ll = M.bounding_box_from_gdf(gdf_ll)
    if bbox_ll is not None and hasattr(bbox_ll, "is_valid") and not bbox_ll.is_valid:
        bbox_ll = bbox_ll.buffer(0)

    # 1) Query OSMnx; if no nodes found, skip
    try:
        if hasattr(ox, "graph_from_polygon"):
            G_roads = ox.graph_from_polygon(bbox_ll, simplify=True, retain_all=True)
        else:
            G_roads = ox.graph.graph_from_polygon(bbox_ll, simplify=True, retain_all=True)
    except ValueError as e:
        # OSMnx uses ValueError for "Found no graph nodes within polygon"
        if verbose:
            print(f"[SKIP tiles] no OSM nodes in polygon: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[SKIP tiles] OSMnx query failed: {type(e).__name__}: {e}")
        return None

    # 2) Create Voronoi tiles; if empty, skip
    try:
        tile_gdf = M.create_voronoi_diagram(G_roads, bbox_ll)

        if tile_gdf is None or len(tile_gdf) == 0:
            if verbose:
                print("[SKIP tiles] empty tile_gdf")
            return None
        if "geometry" not in tile_gdf.columns:
            if verbose:
                print("[SKIP tiles] tile_gdf missing geometry column")
            return None
        if tile_gdf.geometry.isna().all() or tile_gdf.geometry.is_empty.all():
            if verbose:
                print("[SKIP tiles] tile_gdf geometry empty")
            return None

        return tile_gdf

    except Exception as e:
        if verbose:
            print(f"[SKIP tiles] Voronoi failed: {type(e).__name__}: {e}")
        return None

def compute_gt_tile_measures(tile_gdf: gpd.GeoDataFrame, gdf_gt_ll: gpd.GeoDataFrame):
    return [M.get_measures_from_polygon(geom, gdf_gt_ll, gdf_gt_ll, "gt") for geom in tile_gdf.geometry]

def load_or_build_gt_cache_for_idx(idx: str, gt_path: str, cache_dir: str, verbose=False):
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{idx}.pkl")

    if os.path.exists(cache_path):
        return True  # already cached

    gdf_gt_ll = read_geojson(gt_path, default_crs="EPSG:4326")

    tile_gdf = build_tiles_from_gt(gdf_gt_ll, verbose=verbose)
    if tile_gdf is None:
        if verbose:
            print(f"[SKIP idx] {idx}: could not build tiles (no cache written)")
        return False

    gt_tile_measures = compute_gt_tile_measures(tile_gdf, gdf_gt_ll)
    gt_tile_df = gpd.pd.DataFrame(gt_tile_measures)

    gt_buf = buffered_union_geom_in_proj(gdf_gt_ll)

    cache_obj = {
        "idx": str(idx),
        "gdf_gt": gdf_gt_ll,
        "tile_gdf": tile_gdf,
        "gt_tile_df": gt_tile_df,
        "gt_buf": gt_buf,
    }

    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    return True


# ============================
# Main: cache-only across datasets
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
        },
        {
            "name": "pathwaybench_seattle",
            "pred_root": "/data/models/bin/2026_pathway_generation/predictions_seattle/",
            "split": "/data/aiclinic/pathways/pathwaybench_seattle/test.txt",
        },
        {
            "name": "pathwaybench_dc",
            "pred_root": "/data/models/bin/2026_pathway_generation/predictions_dc/",
            "split": "/data/aiclinic/pathways/pathwaybench_dc/test.txt",
        },
    ]

    for ds in datasets:
        pred_root = ds["pred_root"]
        split_path = ds["split"]
        ds_name = ds["name"]

        if not os.path.isdir(pred_root):
            print(f"[SKIP] pred_root not found: {pred_root}")
            continue
        if not os.path.exists(split_path):
            print(f"[SKIP] split file not found: {split_path}")
            continue

        with open(split_path, "r") as f:
            split_indices = [line.strip() for line in f if line.strip()]

        cache_dir = os.path.join(pred_root, f"_gt_cache_{M.PROJ}_{M.BUFF_DIS}")
        ensure_dir(cache_dir)

        gt_map = build_gt_index(pred_root)
        print(f"\n[{ds_name}] GT found for {len(gt_map)} indices. Caching to: {cache_dir}")

        for idx in tqdm(split_indices, desc=f"{ds_name} cache"):
            gt_path = gt_map.get(idx)
            if gt_path is None:
                continue
            load_or_build_gt_cache_for_idx(idx, gt_path, cache_dir, verbose=args.verbose)

    print("\nDone building GT caches for all datasets.")