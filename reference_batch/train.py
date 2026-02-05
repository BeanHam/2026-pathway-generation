import os
import time
import json
import torch
import random
import pickle
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.amp import autocast
from model import PathwayModel
from utils import (
    set_seed, should_freeze_bn, freeze_bn,
    IterativeEncoder, GraphRunner
)
from config import CONFIG

torch.backends.cudnn.benchmark = True


def atomic_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def validate_batched(
    model,
    val_indices,
    spawn_runner,     # function: idx -> GraphRunner(mode="val")
    device,
    mean,
    std,
    batch_graphs_val=32,
):
    """
    Batched validation matching your old semantics:
      - step loss: CE per step
      - node loss: mean(step losses) until STOP
      - graph loss: mean(node losses)
      - returned: mean(graph losses)
    """
    model.eval()
    use_amp = device.startswith("cuda")

    graph_sum = {idx: 0.0 for idx in val_indices}
    graph_cnt = {idx: 0 for idx in val_indices}

    ptr = 0
    runners = []
    runner_ids = []
    B = min(batch_graphs_val, len(val_indices))

    for _ in range(B):
        idx = val_indices[ptr]
        runners.append(spawn_runner(idx, mode="val"))
        runner_ids.append(idx)
        ptr += 1

    with torch.no_grad():
        while runners:
            xs, ys = [], []
            sample_runners = []
            sample_ids = []
            next_runners = []
            next_ids = []

            # one step per active runner
            for r, rid in zip(runners, runner_ids):
                ex = r.make_step_example()
                if ex is None:
                    # finished -> replace with next val graph if available
                    if ptr < len(val_indices):
                        idx = val_indices[ptr]
                        next_runners.append(spawn_runner(idx, mode="val"))
                        next_ids.append(idx)
                        ptr += 1
                    continue

                x_np, y_int, _is_stop = ex
                xs.append(x_np)
                ys.append(y_int)
                sample_runners.append(r)
                sample_ids.append(rid)
                next_runners.append(r)
                next_ids.append(rid)

            runners, runner_ids = next_runners, next_ids
            if not xs:
                continue

            x = torch.from_numpy(np.stack(xs, axis=0)).to(device)   # (N,4,H,W)
            y = torch.tensor(ys, device=device, dtype=torch.long)
            x = (x - mean) / std

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss_per = F.cross_entropy(logits, y, reduction="none")  # (N,)

            # feed step losses to the correct runner
            for k, r_k in enumerate(sample_runners):
                node_loss = r_k.record_step_loss(loss_per[k])
                if node_loss is not None:
                    rid = sample_ids[k]
                    graph_sum[rid] += float(node_loss.detach().item())
                    graph_cnt[rid] += 1

    per_graph = []
    for idx in val_indices:
        per_graph.append(graph_sum[idx] / graph_cnt[idx] if graph_cnt[idx] > 0 else 0.0)

    return float(np.mean(per_graph))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_graphs", type=int, default=32, help="graphs in flight (one step per graph per tick)")
    parser.add_argument("--batch_graphs_val", type=int, default=32, help="val graphs in flight")
    args = parser.parse_args()

    if args.model is not None:
        CONFIG["model"] = args.model
    if CONFIG["model"].startswith("vit"):
        CONFIG["output_size"] = (224, 224)
    CONFIG["device"] = args.device

    device = CONFIG["device"]
    set_seed(100)

    with open(CONFIG["train_index"], "r") as f:
        train_indices = [line.strip() for line in f]
    with open(CONFIG["val_index"], "r") as f:
        val_indices = [line.strip() for line in f]

    run_name = f"{CONFIG['model']}_batched_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(CONFIG["save_path"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    loss_path = os.path.join(run_dir, "loss_log.pkl")
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({**CONFIG, "batch_graphs": args.batch_graphs, "batch_graphs_val": args.batch_graphs_val}, f, indent=2)

    print(f"Run: {run_name}")
    print(f"Dataset: {len(train_indices)} Training, {len(val_indices)} Validation")
    print(f"Model: {CONFIG['model']} | Classes: {CONFIG['total_classes']}")
    print(f"Batch graphs (train): {args.batch_graphs} | Batch graphs (val): {args.batch_graphs_val}")

    model = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=CONFIG["model"]).to(device)
    if should_freeze_bn(CONFIG["model"]):
        model.apply(freeze_bn)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith("cuda"))

    encoder = IterativeEncoder(CONFIG)
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)

    val_every = int(CONFIG.get("val_every", 200))      # every N graphs STARTED
    accum_nodes = int(CONFIG["accum_nodes"])           # step optimizer after this many COMPLETED NODES

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    global_step = 0

    B = int(args.batch_graphs)

    def spawn_runner(idx, mode):
        noise = CONFIG["noise"] if mode == "train" else 0.0
        return GraphRunner(idx, encoder, CONFIG, mode=mode, teacher_noise=noise)

    # -------------------------
    # Training epochs
    # -------------------------
    for epoch in range(int(CONFIG["num_epochs"])):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        random.shuffle(train_indices)

        model.train()
        if should_freeze_bn(CONFIG["model"]):
            model.apply(freeze_bn)

        # start B runners
        ptr = 0
        runners = []
        for _ in range(min(B, len(train_indices))):
            runners.append(spawn_runner(train_indices[ptr], mode="train"))
            ptr += 1

        graphs_started = len(runners)
        pbar = tqdm(total=len(train_indices), desc="Graphs started")
        pbar.update(graphs_started)

        train_node_count = 0
        train_node_loss_sum = 0.0
        nodes_since_step = 0
        use_amp = device.startswith("cuda")

        while runners:
            xs, ys = [], []
            sample_runners = []   # aligns with xs/loss_per
            next_runners = []

            # one step per active runner
            for r in runners:
                ex = r.make_step_example()
                if ex is None:
                    # finished -> replace with next train graph if available
                    if ptr < len(train_indices):
                        new_r = spawn_runner(train_indices[ptr], mode="train")
                        ptr += 1
                        graphs_started += 1
                        pbar.update(1)
                        next_runners.append(new_r)
                    continue

                x_np, y_int, _is_stop = ex
                xs.append(x_np)
                ys.append(y_int)
                sample_runners.append(r)
                next_runners.append(r)

            runners = next_runners
            if not xs:
                continue

            x = torch.from_numpy(np.stack(xs, axis=0)).to(device)   # (N,4,H,W)
            y = torch.tensor(ys, device=device, dtype=torch.long)
            x = (x - mean) / std

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss_per = F.cross_entropy(logits, y, reduction="none")  # (N,)

            # accumulate node losses and backprop per completed node
            for k, r_k in enumerate(sample_runners):
                node_loss = r_k.record_step_loss(loss_per[k])
                if node_loss is None:
                    continue

                nodes_since_step += 1
                train_node_count += 1
                nl_val = float(node_loss.detach().item())
                train_node_loss_sum += nl_val
                history["train"].append((epoch, global_step, nl_val))

                if use_amp:
                    scaler.scale(node_loss / accum_nodes).backward()
                else:
                    (node_loss / accum_nodes).backward()

                if nodes_since_step % accum_nodes == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # periodic validation trigger
            if (graphs_started % val_every) == 0 and graphs_started > 0:
                # flush partial accumulation before eval so you don't carry grads across eval
                if nodes_since_step % accum_nodes != 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    nodes_since_step = 0

                avg_val = validate_batched(
                    model=model,
                    val_indices=val_indices,
                    spawn_runner=spawn_runner,
                    device=device,
                    mean=mean,
                    std=std,
                    batch_graphs_val=int(args.batch_graphs_val),
                )

                history["val"].append((epoch, global_step, avg_val))
                print(f"\n[VAL @ epoch {epoch+1}, graphs_started {graphs_started}, global_step {global_step}] loss={avg_val:.4f}")

                atomic_pickle_dump(history, loss_path)
                torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
                    print("  [+] New Best Model! Saved.")

                model.train()
                if should_freeze_bn(CONFIG["model"]):
                    model.apply(freeze_bn)

        pbar.close()

        # end-epoch flush if needed
        if nodes_since_step % accum_nodes != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        atomic_pickle_dump(history, loss_path)
        torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))

        if train_node_count > 0:
            print(f"Epoch {epoch+1}: avg node loss = {train_node_loss_sum/train_node_count:.4f} over {train_node_count} nodes")

    print("\nTraining Complete.")
