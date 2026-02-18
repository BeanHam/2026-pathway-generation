import os
import json
import torch
import argparse
import torch.distributed as dist
from datetime import timedelta
from utils import *
from model import *
from config import *

torch.backends.cudnn.benchmark = True

def ddp_setup():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--fp_lambda", type=float, default=0.0)
    args = parser.parse_args()
    
    # detect DDP
    use_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank, rank, world_size = ddp_setup()
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = args.device

    # config
    CONFIG["model"] = args.model
    if CONFIG["model"].startswith("vit"):
        CONFIG["output_size"] = (224, 224)
    CONFIG["device"] = device
    CONFIG["fp_lambda"] = float(args.fp_lambda)

    # checkpoint paths
    reference_path = REFERENCE_DIRS[args.model]
    CONFIG["init_ckpt"] = os.path.join(reference_path, "model_best.pth")
    with open(os.path.join(reference_path, "inference_config.json"), "r") as f:
        inference_config = json.load(f)
    
    CONFIG["topk"]                = inference_config["topk"]
    CONFIG["restarts_per_graph"]  = inference_config["restarts_per_graph"]
    CONFIG["stop_prob_thresh"]    = inference_config["stop_prob_thresh"]
    CONFIG["max_expand_per_node"] = inference_config["max_expand_per_node"]

    # train
    grpo_train_main_cont(CONFIG, ddp_rank=rank, ddp_world_size=world_size)

    if use_ddp:
        dist.barrier()
        ddp_cleanup()