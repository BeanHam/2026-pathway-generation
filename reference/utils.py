import os
import cv2
import torch
import pickle
import random
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=256)
def _load_graph_and_image(path):
    with open(os.path.join(path, "graph.pickle"), "rb") as f:
        G = pickle.load(f)
    img_bgr = cv2.imread(os.path.join(path, "satellite.png"), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(os.path.join(path, "satellite.png"))
    satellite_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return G, satellite_img

def should_freeze_bn(model_name: str) -> bool:
    return model_name.lower().startswith("resnet")

def freeze_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()                 # use running stats
        m.requires_grad_(False)  # optional: freeze gamma/beta too

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

# ==========================================
# Cropping Image
# ==========================================
def crop_image(img_arr, center, crop_size=0.5, output_size=(256, 256)):
    """
    Crops image (H, W, C) around center (x, y). 
    Assumes img_arr is SQUARE (H=W) to maintain aspect ratio consistency.
    """
    H, W = img_arr.shape[:2]
    cx_norm, cy_norm = center
    
    # 1. Convert Center to Pixels (WITH Y-INVERSION)
    # Geospatial (0,0) is Bottom-Left. Image (0,0) is Top-Left.
    px_cx = int(cx_norm * W)
    px_cy = int((1 - cy_norm) * H)
    
    # 2. Calculate Radius (half of the crop window)
    # Since W=H, this covers the same normalized distance in X and Y.
    # crop_size 0.5 -> covers 0.5 normalized units (0.25 radius).
    # This guarantees 'dist_max=0.2' is visible.
    px_half = int((crop_size * W) / 2)
    
    x1, y1 = px_cx - px_half, px_cy - px_half
    x2, y2 = px_cx + px_half, px_cy + px_half
    
    # 3. Padding Logic (Handle boundaries)
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
    
    x1_v, y1_v = max(0, x1), max(0, y1)
    x2_v, y2_v = min(W, x2), min(H, y2)
    
    valid_crop = img_arr[y1_v:y2_v, x1_v:x2_v]
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        pads = ((pad_t, pad_b), (pad_l, pad_r))
        if len(img_arr.shape) == 3: 
            pads += ((0, 0),) 
        valid_crop = np.pad(valid_crop, pads, mode='constant', constant_values=0)
            
    return cv2.resize(valid_crop, output_size, interpolation=cv2.INTER_LINEAR)
    
    
# ==========================================
# Label Encoder
# ==========================================
class IterativeEncoder:
    def __init__(self, config):
        self.num_angles = config['num_angles']
        self.stop_idx = config['stop_class_idx']
        self.dist_bins = np.array([
            0.0, 0.02, 0.04, 0.06, 0.08, 
            0.1, 0.12, 0.14, 0.16, 0.18, 0.2
        ])        
        self.num_dists = len(self.dist_bins) - 1  # Should be 10
        self.angle_step = 360.0 / self.num_angles
        self.angle_offset = -(self.angle_step / 2.0)

    def get_action_label(self, u, v):
        # 1. Deltas (Normalized Geospatial)
        # u is start, v is end.
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        
        # 2. Angle Calculation
        deg = np.degrees(np.arctan2(dy, dx)) % 360
        deg_shift = (deg - self.angle_offset) % 360
        a_idx = int(deg_shift / self.angle_step) % self.num_angles
        
        # 3. Distance
        dist = np.sqrt(dx**2 + dy**2)
        d_idx = np.digitize(dist, self.dist_bins) - 1
        d_idx = max(0, min(d_idx, self.num_dists - 1))
        
        return (a_idx * self.num_dists) + d_idx
    
    def decode_action(self, action_idx):
        """
        Converts class index back to (dx, dy) using Bin Midpoints.
        """
        if action_idx == self.stop_idx:
            return None
            
        # 1. Decompose Index
        a_idx = action_idx // self.num_dists
        d_idx = action_idx % self.num_dists
        
        # 2. Recover Angle
        angle_deg = (a_idx * self.angle_step) + self.angle_offset + (self.angle_step / 2.0)
        
        # 3. Recover Distance (Use Midpoint of the specific bin)
        bin_min = self.dist_bins[d_idx]
        bin_max = self.dist_bins[d_idx+1]
        dist_val = (bin_min + bin_max) / 2.0
        
        # 4. Convert to Deltas
        angle_rad = np.radians(angle_deg)
        dx = dist_val * np.cos(angle_rad)
        dy = dist_val * np.sin(angle_rad)
        
        return dx, dy
    
# ==========================================
# Train Each Graph
# ==========================================
def process_graph(folder_idx,
    model,
    encoder,
    CONFIG,
    mode="train",
    criterion=None,
    optimizer=None,
    teacher_noise=0.0,
    mean=None,
    std=None
):
    """
    Teacher-forced graph rollout over GT graph:
      - global_canvas stores generated edges so far (always GT edges in canonical order).
      - at each expanded node u (optionally noised position u_pos), we iteratively supervise the next
        remaining neighbor action until STOP.
    """
    
    assert criterion is not None, "criterion must be passed (e.g., nn.CrossEntropyLoss())"
    assert mean is not None and std is not None, "mean/std must be passed"
    if mode == "train":
        assert optimizer is not None
        
    # --- Load Data ---
    path = os.path.join(CONFIG["data_root"], str(folder_idx))
    try:
        G, satellite_img = _load_graph_and_image(path)
        H, W = satellite_img.shape[:2]
    except Exception as e:
        print(f"[LOAD FAIL] {folder_idx}: {e}", flush=True)
        return 0.0

    if len(G.nodes) == 0:
        return 0.0
    
    # --- State ---
    global_canvas   = np.zeros((H, W), dtype=np.uint8)
    generated_edges = set()   # edges that are already generated/drawn (drives valid actions + STOP)
    expanded_nodes  = set()   # nodes whose outgoing edges we already processed
    queued_nodes    = set()   # nodes already pushed to stack (prevents duplicates)

    all_nodes = list(G.nodes)
    start_node = random.choice(all_nodes)
    stack = [(start_node, start_node)]
    queued_nodes.add(start_node)

    # --- Torch setup ---
    if mode == "train":
        model.train()
        if should_freeze_bn(CONFIG["model"]):
            model.apply(freeze_bn)
    else:
        model.eval()

    device = CONFIG["device"]
    accum_nodes = CONFIG["accum_nodes"]
    node_counter = 0
    total_loss = 0.0
    num_expanded = 0
    
    with torch.set_grad_enabled(mode == "train"):
        while num_expanded < CONFIG["max_steps"]:
            # If stack empties but we still want to cover the whole graph, restart from an unexpanded node.
            if not stack:
                remaining = [n for n in G.nodes if n not in expanded_nodes]
                if not remaining:
                    break
                new_seed = random.choice(remaining)
                stack.append((new_seed, new_seed))
                queued_nodes.add(new_seed)

            u_node, u_pos = stack.pop()
            if teacher_noise > 0 and mode == "train":
                u_pos = apply_noise(u_node, scale=teacher_noise)
            if u_node in expanded_nodes:
                continue
            expanded_nodes.add(u_node)
            num_expanded += 1

            # --- FIND & SORT NEIGHBORS (CANONICAL) ---
            # Remaining GT neighbors (edges not generated yet)
            neighbors = list(G.neighbors(u_node))
            remaining_neighbors = [n for n in neighbors if frozenset({u_node, n}) not in generated_edges]
            canon = sorted(
                remaining_neighbors,
                key=lambda n: (encoder.get_action_label(u_node, n), float(n[0]), float(n[1])),
            )
            
            # Crop around DRIFTED pos
            px_u = (int(u_pos[0]*W), int((1-u_pos[1])*H)) 
            step_losses = []
            remaining_set = set(canon)
                        
            while True:
                # Expert action at this state:
                if not remaining_set:
                    expert_target = "STOP"
                    expert_label = CONFIG["stop_class_idx"]
                else:
                    # canonical smallest among remaining
                    expert_target = next(n for n in canon if n in remaining_set)
                    expert_label = encoder.get_action_label(u_pos, expert_target)

                # --- Build input from CURRENT canvas state ---                
                canvas_with_dot = global_canvas.copy()
                cv2.circle(canvas_with_dot, px_u, radius=10, color=255, thickness=-1, lineType=cv2.LINE_AA)
                sat_crop = crop_image(satellite_img, u_pos, CONFIG["crop_size"], CONFIG["output_size"])
                can_crop = crop_image(canvas_with_dot, u_pos, CONFIG["crop_size"], CONFIG["output_size"])
                
                inp = np.concatenate([sat_crop, can_crop[..., None]], axis=2)  # (H,W,4)
                x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1,4,256,256)
                x = x.to(device)
                x = (x - mean) / std
                y = torch.tensor([expert_label], device=device, dtype=torch.long)

                logits = model(x)  # (1,361)
                loss_step = criterion(logits, y)
                step_losses.append(loss_step)

                # STOP ends the per-node sequence
                if expert_target == "STOP":
                    break
                
                # --- Apply generation update (teacher-forced GT edge) ---
                v_node = expert_target
                px_v = (int(v_node[0] * W), int((1 - v_node[1]) * H))
                cv2.line(global_canvas, px_u, px_v, color=255, thickness=2, lineType=cv2.LINE_AA)                   
                generated_edges.add(frozenset({u_node, v_node}))
                if v_node in remaining_set:
                    remaining_set.remove(v_node)                                        

                # --- PUSH NEIGHBOR FOR FUTURE EXPANSION (restores traversal) ---
                if v_node not in expanded_nodes and v_node not in queued_nodes:
                    # keep child position anchored at GT node; noise applied when node is popped
                    stack.append((v_node, v_node))
                    queued_nodes.add(v_node)
            
            # --- Node-level backward / logging ---
            node_loss = torch.stack(step_losses).mean()
            if mode == "train":
                (node_loss / accum_nodes).backward()
                node_counter += 1
                if node_counter % accum_nodes == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            total_loss += float(node_loss.detach().item())
            
        # Flush remaining grads
        if mode == "train" and node_counter % accum_nodes != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_expanded, 1)