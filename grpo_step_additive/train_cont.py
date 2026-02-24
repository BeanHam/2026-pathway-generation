import os
import json
import argparse
import torch
import torch.distributed as dist
from datetime import timedelta

from utils import grpo_train_main_cont
from config import CONFIG, REFERENCE_DIRS

torch.backends.cudnn.benchmark = True

def ddp_setup():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default="last", choices=["last", "best"])
    parser.add_argument("--extra_epochs", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.run_dir
    run_cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(run_cfg_path)

    # detect DDP
    use_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank, rank, world_size = ddp_setup()
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = args.device

    # load saved run config
    with open(run_cfg_path, "r") as f:
        saved = json.load(f)

    # merge into CONFIG (saved run config wins)
    CONFIG.update(saved)
    CONFIG["device"] = device

    # ensure topk / stop_prob_thresh / max_expand_per_node are present
    # (if your run_config already stored them, this does nothing)
    reference_path = REFERENCE_DIRS[CONFIG["model"]]
    inf_path = os.path.join(reference_path, "inference_config.json")
    if os.path.exists(inf_path):
        with open(inf_path, "r") as f:
            inference_config = json.load(f)
        CONFIG["topk"] = inference_config["topk"]
        CONFIG["restarts_per_graph"] = inference_config.get("restarts_per_graph", 1)
        CONFIG["stop_prob_thresh"] = inference_config["stop_prob_thresh"]
        CONFIG["max_expand_per_node"] = inference_config["max_expand_per_node"]

    # bump total epochs if requested
    if args.extra_epochs and args.extra_epochs > 0:
        CONFIG["num_epochs"] = int(CONFIG["num_epochs"]) + int(args.extra_epochs)

    grpo_train_main_cont(
        CONFIG,
        run_dir=run_dir,
        resume_which=args.resume,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )

    if use_ddp:
        dist.barrier()
        ddp_cleanup()
        
        
        
# torchrun --nproc_per_node=4 --master_port=29501 train_cont.py --run_dir /data/models/bin/2026_pathway_generation/grpo_logs_steps_additive/resnet18_20260223_114320/