import os
import time
import json
import pickle
import argparse
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import CONFIG
from model import PathwaySegmentationModel
from utils import (
    atomic_pickle_dump,
    compute_iou,
    PathwaySegmentationDataset,
    set_seed,
)

torch.backends.cudnn.benchmark = True

def ddp_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    return local_rank, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    local_rank, rank, world_size = ddp_setup()
    if args.model is not None:
        CONFIG['model'] = args.model
    device = f'cuda:{local_rank}'
    CONFIG['device'] = device

    set_seed(100 + rank)

    train_ds = PathwaySegmentationDataset(
        data_root=CONFIG['data_root'],
        index_path=CONFIG['train_index'],
        output_size=CONFIG['output_size'],
        mask_line_width=CONFIG['mask_line_width'],
    )
    val_ds = PathwaySegmentationDataset(
        data_root=CONFIG['data_root'],
        index_path=CONFIG['val_index'],
        output_size=CONFIG['output_size'],
        mask_line_width=CONFIG['mask_line_width'],
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG['batch_size']),
        sampler=train_sampler,
        num_workers=int(CONFIG['num_workers']),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(CONFIG['num_workers']) > 0,
    )

    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(CONFIG['batch_size']),
            shuffle=False,
            num_workers=max(0, int(CONFIG['num_workers']) // 2),
            pin_memory=True,
            drop_last=False,
            persistent_workers=max(0, int(CONFIG['num_workers']) // 2) > 0,
        )

        run_name = f"{CONFIG['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(CONFIG['save_path'], run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump({**CONFIG, 'world_size': world_size}, f, indent=2)
        loss_path = os.path.join(run_dir, 'loss_log.pkl')
        print(f'Run: {run_name}')
    else:
        val_loader = None
        run_dir = None
        loss_path = None

    model = PathwaySegmentationModel(
        backbone_name=CONFIG['model'],
        pretrained=bool(CONFIG['pretrained']),
        decoder_dim=int(CONFIG['decoder_dim']),
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    pos_weight = torch.tensor([float(CONFIG['pos_weight'])], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {'train': [], 'val': []}
    best_val = float('inf')
    global_step = 0

    for epoch in range(int(CONFIG['num_epochs'])):
        train_sampler.set_epoch(epoch)

        model.train()
        train_sum = 0.0
        train_count = 0

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1} rank0')
        else:
            pbar = None

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['grad_clip'])
            optimizer.step()

            train_sum += float(loss.detach().item())
            train_count += 1

            if rank == 0 and pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'train(avg)': f'{train_sum / max(train_count, 1):.4f}'})
                history['train'].append((epoch, global_step, float(loss.detach().item())))
                global_step += 1

        if rank == 0 and pbar is not None:
            pbar.close()

        dist.barrier(device_ids=[local_rank])

        if rank == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_iou = 0.0
                val_count = 0
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = model.module(images)
                    loss = criterion(logits, masks)
                    val_loss += float(loss.detach().item())
                    val_iou += compute_iou(logits, masks, threshold=0.5)
                    val_count += 1

                avg_val = val_loss / max(val_count, 1)
                avg_iou = val_iou / max(val_count, 1)

            history['val'].append((epoch, global_step, float(avg_val), float(avg_iou)))
            print(f'\n[VAL @ epoch {epoch+1}] loss={avg_val:.4f} iou={avg_iou:.4f}')

            atomic_pickle_dump(history, loss_path)
            torch.save(model.module.state_dict(), os.path.join(run_dir, 'model_last.pth'))
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.module.state_dict(), os.path.join(run_dir, 'model_best.pth'))
                print('  [+] New Best Model! Saved.')

        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print('Training complete.')
    ddp_cleanup()
