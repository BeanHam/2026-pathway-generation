import os
import time
import json
import pickle
import random
import argparse
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from model import PathwayModel
from utils import (
    set_seed,
    should_freeze_bn, freeze_bn,
    IterativeEncoder,
    process_graph_graphloss,
)
from config import CONFIG

torch.backends.cudnn.benchmark = True

def atomic_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return local_rank, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

def pad_to_world_size(items, world_size):
    r = len(items) % world_size
    if r == 0:
        return items
    pad = world_size - r
    return items + items[:pad]

def shard_list(items, rank, world_size):
    return items[rank::world_size]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--accum_graphs", type=int, default=1, help="gradient accumulation in graphs")
    args = parser.parse_args()

    local_rank, rank, world_size = ddp_setup()

    if args.model is not None:
        CONFIG["model"] = args.model
    if CONFIG["model"].startswith("vit"):
        CONFIG["output_size"] = (224, 224)

    device = f"cuda:{local_rank}"
    CONFIG["device"] = device

    set_seed(100 + rank)

    with open(CONFIG["train_index"], "r") as f:
        train_all = [line.strip() for line in f]
    with open(CONFIG["val_index"], "r") as f:
        val_indices = [line.strip() for line in f]

    if rank == 0:
        run_name = f"{CONFIG['model']}_ddp_graph_{time.strftime('%Y%m%d_%H%M%S')}_cont"
        run_dir = os.path.join(CONFIG["save_path"], run_name)
        os.makedirs(run_dir, exist_ok=True)
        loss_path = os.path.join(run_dir, "loss_log.pkl")
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump({**CONFIG, "world_size": world_size, "accum_graphs": args.accum_graphs}, f, indent=2)
        print(f"Run: {run_name}")
    else:
        run_dir = None
        loss_path = None

    model = PathwayModel(num_classes=CONFIG["total_classes"], backbone_name=CONFIG["model"]).to(device)
    
    # --------------------------------------------------------------------------------------------------------
    log = '/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260206_130501/'
    model.load_state_dict(torch.load(log+"model_best.pth", map_location=device))
    loss = np.load(log+'loss_log.pkl', allow_pickle=True)
    val = np.stack(loss['val'])[:,2]
    best_val = np.min(val)
    # --------------------------------------------------------------------------------------------------------
    
    if should_freeze_bn(CONFIG["model"]):
        model.apply(freeze_bn)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    encoder = IterativeEncoder(CONFIG)

    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=device).view(1, 4, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=device).view(1, 4, 1, 1)

    history = {"train": [], "val": []}
    global_step = 0

    accum_graphs = max(1, int(args.accum_graphs))

    for epoch in range(int(CONFIG["num_epochs"])):
        rng = random.Random(12345 + epoch)
        shuffled = train_all[:]
        rng.shuffle(shuffled)
        shuffled = pad_to_world_size(shuffled, world_size)
        train_indices = shard_list(shuffled, rank, world_size)

        if rank == 0:
            pbar = tqdm(total=len(train_indices), desc=f"Epoch {epoch+1} rank0")
        else:
            pbar = None

        model.train()
        if should_freeze_bn(CONFIG["model"]):
            model.module.apply(freeze_bn)

        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0

        for i, idx in enumerate(train_indices, start=1):
            loss = process_graph_graphloss(
                idx, model, encoder, CONFIG,
                mode="train",
                criterion=criterion,
                teacher_noise=CONFIG["noise"],
                mean=mean, std=std,
            )  # tensor

            (loss / accum_graphs).backward()
            train_sum += float(loss.detach().item())

            if (i % accum_graphs) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if rank == 0 and pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"train(avg)": f"{train_sum/i:.3f}"})
                history["train"].append((epoch, global_step, float(loss.detach().item())))
                global_step += 1

        if (len(train_indices) % accum_graphs) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if rank == 0 and pbar is not None:
            pbar.close()

        dist.barrier(device_ids=[local_rank])

        if rank == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for v_idx in val_indices:
                    v = process_graph_graphloss(
                        v_idx, model, encoder, CONFIG,
                        mode="val",
                        criterion=criterion,
                        teacher_noise=0.0,
                        mean=mean, std=std,
                    )
                    val_loss += float(v)
                avg_val = val_loss / max(len(val_indices), 1)

            history["val"].append((epoch, global_step, float(avg_val)))
            print(f"\n[VAL @ epoch {epoch+1}] loss={avg_val:.4f}")

            atomic_pickle_dump(history, loss_path)
            torch.save(model.module.state_dict(), os.path.join(run_dir, "model_last.pth"))
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.module.state_dict(), os.path.join(run_dir, "model_best.pth"))
                print("  [+] New Best Model! Saved.")

        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("Training complete.")
    ddp_cleanup()