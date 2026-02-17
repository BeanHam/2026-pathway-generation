import traceback
import itertools
import numpy as np
import shapely.ops
if not hasattr(shapely.ops, 'cascaded_union'):
    shapely.ops.cascaded_union = shapely.ops.unary_union

import networkx as nx
import geopandas as gpd
import geonetworkx as gnx

from statistics import mean, stdev
from scipy.spatial import ConvexHull
from shapely.ops import unary_union
from shapely.ops import voronoi_diagram
from shapely.geometry import Point, LineString, MultiLineString, Polygon, GeometryCollection

# PROJ = 'epsg:4326'
# PRES = 1e-5
# E_THRES = 2.5e-5
# BUFF_DIS = 2e-5

PROJ = 'epsg:26910'
PRES = 1e-5
E_THRES = 5
BUFF_DIS = 4

def bounding_box_from_gdf(gdf):
    # Get the bounding box coordinates directly from the GeoDataFrame
    min_x, min_y, max_x, max_y = gdf.total_bounds

    # Create a Polygon from the bounding box coordinates
    bounding_box_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    return bounding_box_polygon

def create_voronoi_diagram(G_roads_simplified, bounds):
    # first thin the nodes 
    gdf_roads_simplified = gnx.graph_edges_to_gdf(G_roads_simplified)
    voronoi = voronoi_diagram(gdf_roads_simplified.boundary.unary_union, envelope = bounds)
    voronoi_gdf = gpd.GeoDataFrame({"geometry": voronoi.geoms})
    voronoi_gdf = voronoi_gdf.set_crs(gdf_roads_simplified.crs)
    voronoi_gdf_clipped = gpd.clip(voronoi_gdf, bounds)
    voronoi_gdf_clipped = voronoi_gdf_clipped.to_crs(PROJ)
    
    return voronoi_gdf_clipped

def add_edges_from_linestring(graph, linestring, edge_attrs):
    """ Add edges to the NetworkX graph from a LineString. """
    points = list(linestring.coords)
    #points = [(round(x, 0), round(y, 0)) for x, y in points] #round to meter
    for start, end in zip(points[:-1], points[1:]):
        graph.add_edge(start, end, **edge_attrs)


def graph_from_gdf(gdf):
    # Initialize an empty graph
    G = nx.Graph()

    # Iterate through each row in the GeoDataFrame
    for index, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            add_edges_from_linestring(G, geom, row.to_dict())
        elif isinstance(geom, MultiLineString):
            for linestring in geom.geoms:
                add_edges_from_linestring(G, linestring, row.to_dict())
    return G


def hull_connected_paths(G):
    # G is a NetworkX graph, with nodes being tuples of (longitude, latitude)

    pos = {node: node for node in G.nodes()}

    points = np.array([node for node in G.nodes()])

    # Calculate the convex hull
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]
    hull_nodes = [tuple(point) for point in hull_vertices] 

    # find all parirs
    hull_vertices_pairs = list(itertools.combinations(hull_nodes, 2))

    n_total = len(hull_nodes)
    n_connected = 0
    c_pairs = list()
    for pair in hull_vertices_pairs:
        is_connected = nx.has_path(G, pair[0], pair[1])
        if is_connected:
            n_connected += 1
            if pair[0] not in c_pairs:
                c_pairs.append(pair[0])
            if pair[1] not in c_pairs:
                c_pairs.append(pair[1])
    #print(f'total hull nodes: {len(hull_nodes)}, connected pair hull nodes: {n_connected}')

    """
    # Plot the graph
    pos = {tuple(node): node for node in G.nodes()}
    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=5)  # Draw nodes
    nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in hull_nodes}, nodelist = hull_nodes, node_color='red', node_size=10)

    # Optionally, draw the convex hull as a polygon if desired
    #for simplex in hull.simplices:
    #    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=2)

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in c_pairs}, nodelist = c_pairs, node_color='green', node_size=10)

    plt.savefig('tile2net_test_graph_3.png')
    exit()
    """

    return n_total, n_connected


def clip_gdf(gdf, poly):
    P = poly

    # Clip the GeoDataFrame
    gdf_clipped = gpd.clip(gdf, P)

    # Prepare to store intersection points
    intersection_points = []

    # Loop through the clipped geometries to find intersection points with the polygon boundary
    for geometry in gdf_clipped.geometry:
        intersection = geometry.intersection(P.boundary)
        if isinstance(intersection, Point):
            intersection_points.append(intersection)
        elif isinstance(intersection, LineString):
            # No intersection points if the intersection is a LineString
            continue
        else:
            # If multiple points (or other geometries), handle appropriately
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    intersection_points.append(geom)

    return gdf_clipped, intersection_points


def group_pts(pts, poly):
    P = poly
    intersection_points = pts
    # Get polygon boundary as a list of line segments
    boundary = list(P.boundary.coords)
    segments = [LineString([boundary[i], boundary[i + 1]]) for i in range(len(boundary) - 1)]

    # Dictionary to hold points grouped by line segments
    segment_point_map = {index: [] for index in range(len(segments))}

    # Group points by which line segment they fall on
    for point in intersection_points:
        for idx, segment in enumerate(segments):
            if segment.distance(point) < PRES:  # Small threshold for precision issues
                segment_point_map[idx].append((point.x, point.y))
                break
    return segment_point_map


def group_G_pts(G, poly):
    P = poly
    
    node_pts = [Point(x) for x in G.nodes()]
    # Get polygon boundary as a list of line segments
    boundary = list(P.boundary.coords)
    segments = [LineString([boundary[i], boundary[i + 1]]) for i in range(len(boundary) - 1)]

    # Dictionary to hold points grouped by line segments
    segment_point_map = {index: [] for index in range(len(segments))}

    # Group points by which line segment they fall on
    for point in node_pts:
        for idx, segment in enumerate(segments):
            if segment.distance(point) < PRES:  # Small threshold for precision issues
                segment_point_map[idx].append((point.x, point.y))
                break
    return segment_point_map


def edges_are_connected(G, e1_pts, e2_pts):
    for pt1 in e1_pts:
        for pt2 in e2_pts: 
            # node self to self should be count at connected
            if pt1 != pt2:
                if nx.has_path(G, pt1, pt2):
                    return True
    return False 


def tile_tra_score(G, polygon):
    # compute number of connected edge pairs

    # assign each point to an polygon line
    pts_line_map = group_G_pts(G, polygon) 
    boundary_nodes = [item for sublist in pts_line_map.values() for item in sublist]

    # find all pair of edges
    #edge_pairs = list(itertools.combinations(pts_line_map.keys(), 2))
    edge_pairs = list(itertools.combinations_with_replacement(pts_line_map.keys(), 2))

    n_total = len(edge_pairs)
    n_connected = 0
    connected_pairs = list()
    for pair in edge_pairs:
        is_connected = edges_are_connected(G, pts_line_map[pair[0]], pts_line_map[pair[1]])
        if is_connected:
            #print(f'{pair} is connected')
            connected_pairs.append(pair)
            n_connected += 1

    #print(f'number of boundary pts {len(boundary_nodes)}')
    #print(f'{len(G.edges())}: {n_total} pair of edges, {n_connected} are connected')

    """
    # Plot the graph
    pos = {tuple(node): node for node in G.nodes()}

    plt.figure(figsize=(20, 15))  # Optional: Adjust figure size

    # first plot the polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='green', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
    # Annotate each edge with its index
    for i, (start, end) in enumerate(zip(polygon.exterior.coords[:-1], polygon.exterior.coords[1:])):
        # Find the midpoint of each edge
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        # Draw the index number at the midpoint
        plt.text(midpoint[0], midpoint[1], str(i), color='green', fontsize=12, ha='center')

    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=5)  # Draw nodes
    nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in boundary_nodes}, nodelist = boundary_nodes, node_color='red', node_size=20)

    """

    return n_total, n_connected, connected_pairs 


def _flatten_lines(geom):
    """Return a list of LineString components from any geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_flatten_lines(g))
        return out
    # ignore Points/Polygons/etc
    return []

def normalize_lines_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode to LineStrings only so geopandas.overlay won't complain."""
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame({"geometry": []}, crs=getattr(gdf, "crs", None))

    rows = []
    cols = list(gdf.columns)

    for _, row in gdf.iterrows():
        geoms = _flatten_lines(row.geometry)
        if not geoms:
            continue
        for ln in geoms:
            new_row = row.copy()
            new_row.geometry = ln
            rows.append(new_row)

    if not rows:
        return gpd.GeoDataFrame({c: [] for c in cols}, crs=gdf.crs)

    out = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    return out

def compute_f1(pred, gt):
    
    pred_sw = normalize_lines_gdf(pred)
    gt_sw   = normalize_lines_gdf(gt)
    pred_sw = pred_sw[pred_sw.geometry.notnull() & ~pred_sw.geometry.is_empty].copy()
    gt_sw   = gt_sw[gt_sw.geometry.notnull() & ~gt_sw.geometry.is_empty].copy()
    
    num_splits = 5
    tp = 0
    fp = 0

    for _, pred_it in pred_sw.iterrows():
        try:
            shape_geo = pred_it["geometry"]
            shape_geo_dia = shape_geo.buffer(BUFF_DIS)

            pred_copy = pred_it.to_frame().T.reset_index(drop=True)
            pred_copy["geometry"] = shape_geo_dia
            pred_copy = gpd.GeoDataFrame(pred_copy, geometry="geometry", crs=PROJ)

            inter = gt_sw.overlay(pred_copy, keep_geom_type=True, how="intersection")

            pred_it_pts = [shape_geo.interpolate((i/num_splits), normalized=True) for i in range(1, num_splits)]
            pred_it_pts_gdf = gpd.GeoDataFrame({"geometry": pred_it_pts}, crs=pred_copy.crs)

            if not inter.empty:
                distance_matched = pred_it_pts_gdf.sjoin_nearest(inter, distance_col="distances", how="inner")
                d_filter = [x for x in distance_matched["distances"].tolist() if x <= E_THRES]
                if len(d_filter) != 0:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1

        except Exception:
            traceback.print_exc()
            continue

    return tp, fp

def get_stats(polygon, G, gdf, gdf_gt):
    stats = {}
    undirected_g = nx.Graph(G)
    
    # betweenness
    try:
        bet = nx.betweenness_centrality(undirected_g, normalized = True, endpoints=False)
        stats["bet_centrality_avg"] = mean(bet.values())
        stats["bet_stdev"] = stdev(bet.values())
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting betweenness value")
        stats["bet_centrality_avg"] = -99.99
        stats["bet_stdev"] = -99.99

    # eigen
    try:
        eigen = nx.eigenvector_centrality(undirected_g, max_iter=1000)
        stats["eig_centrality_avg"] = mean(eigen.values())
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting eigen value")
        stats["eig_centrality_avg"] = -99.99

    # degree
    try:
        deg = nx.degree_centrality(undirected_g)
        stats["deg_centrality_avg"] = mean(deg.values())
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting degree cventrality")
        stats["deg_centrality_avg"] = -99.99

    # number of connected components
    try:
        noc = nx.number_connected_components(undirected_g)
        stats["num_connect_comp_avg"] = noc
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected components")
        #traceback.print_exc()
        stats["num_connect_comp_avg"] = -99.99

    # node connectivity 
    try:
        conn = nx.average_node_connectivity(undirected_g)
        stats["node_connect_avg"] = conn
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting node connectivity")
        #traceback.print_exc()
        stats["node_connect_avg"] = -99.99

    # node-to-node connected paths
    try:
        _, n_pahts = hull_connected_paths(undirected_g)
        stats["n_connect_paths"] = n_pahts
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected paths")
        #traceback.print_exc()
        stats["n_connect_paths"] = -99.99

    # edge-to-edge connected paths
    try:
        n_total, n_connected, connected_pairs = tile_tra_score(undirected_g, polygon)
        stats["n_total_edges"] = n_total
        stats["n_connect_edges"] = n_connected
        connected_pairs_str = ' '.join([f"({t[0]},{t[1]})" for t in connected_pairs])
        stats['connected_pairs'] = connected_pairs_str
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected edge pairs")
        traceback.print_exc()
        stats["n_total_edges"] = -99.99
        stats["n_connect_edges"] = -99.99
        stats['connected_pairs'] = "-99.99"

    # f1 score
    try:
        tp, fp = compute_f1(gdf, gdf_gt)
        tp, fn = compute_f1(gdf_gt, gdf)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision + recall)
        stats["precision"] = precision
        stats["recall"] = recall
        stats["f1"] = f1
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting f1 score")
        #traceback.print_exc()
        stats["precision"] = -99.99
        stats["recall"] = -99.99
        stats["f1"] = -99.99

    return stats


def get_measures_from_polygon(polygon, gdf, gdf_gt, name):
    gdf = gdf.to_crs(PROJ)
    gdf_gt = gdf_gt.to_crs(PROJ)

    # crop gdf to the polygon
    #cropped_gdf = gdf
    cropped_gdf = gpd.clip(gdf, polygon)
    cropped_gdf_gt = gpd.clip(gdf_gt, polygon)

    if name == 'pred':
        G = graph_from_gdf(cropped_gdf) 
    elif name == 'gt':
        G = graph_from_gdf(cropped_gdf_gt)         

    stats = get_stats(polygon, G, cropped_gdf, cropped_gdf_gt)    
    return stats


def compute_global_stats(gdf, filepath):
    gdf = gpd.read_file(filepath)
    G = graph_from_gdf(gdf)

    # (1) Total number of nodes
    total_nodes = G.number_of_nodes()

    # (2) Total number of edges
    total_edges = G.number_of_edges()

    # (3) Average degree of nodes
    average_degree = sum(dict(G.degree()).values()) / total_nodes

    print(f'computing global stats for {filepath}')
    print(f"Total number of nodes: {total_nodes}")
    print(f"Total number of edges: {total_edges}")
    print(f"Average degree of nodes: {average_degree}")


def str_2_set(data_str):
    # Remove parentheses and split by space
    tuple_strings = data_str.replace('(', '').replace(')', '').split()

    # Convert each string to a tuple
    data = [tuple(map(int, t.split(','))) for t in tuple_strings]

    return set(data)


def remove_na_str(data_set_1, data_set_2):
    assert len(data_set_1) == len(data_set_2)

    data_set_1_new = list()
    data_set_2_new = list()

    for i in range(len(data_set_1)):
        if data_set_1[i] == '-99.99' or data_set_2[i] == '-99.99':
            pass
        else:
            data_set_1_new.append(data_set_1[i])
            data_set_2_new.append(data_set_2[i])

    return np.array(data_set_1_new), np.array(data_set_2_new)


def remove_na_data(data_set_1):
    data_set_1_new = list()

    for i in range(len(data_set_1)):
        if data_set_1[i] == -99.99:
            pass
        else:
            data_set_1_new.append(data_set_1[i])
    return np.array(data_set_1_new)


def compute_avg(gdf, metric):
    data = gdf[metric]
    data = remove_na_data(data)
    return np.average(np.array(data))


def compute_tra_avg(gdf):
    t = gdf['n_total_edges']
    c = gdf['n_connect_edges']
    t = remove_na_data(t)
    c = remove_na_data(c)
    return np.average(np.array(c/t))


def compute_tra_jaccard(gdf1, gdf2):
    t1 = gdf1['connected_pairs'] # pred
    t2 = gdf2['connected_pairs'] # gt

    t1 = list(t1)
    t2 = list(t2)

    t1, t2 = remove_na_str(t1, t2)

    iou_l = list()

    for i in range(len(t1)):
        set1 = str_2_set(t1[i])
        set2 = str_2_set(t2[i])

        inter = set1 & set2
        union = set1 | set2
        if len(union) > 0:
            iou = len(inter)/len(union)
            iou_l.append(iou)

    r = np.average(np.array(iou_l))

    return r