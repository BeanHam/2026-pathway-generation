import os
import time
import json
import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import *
from utils import *
from config import *

if __name__ == "__main__":
    
    #-------------------
    # parameters
    #-------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if args.model is not None:
        CONFIG["model"] = args.model
    if CONFIG["model"].startswith("vit"):
        CONFIG["output_size"] = (224, 224)
    CONFIG['device'] = args.device
    
    # ------------------------
    # Load Data Indices
    # ------------------------
    set_seed(100)
    with open(CONFIG['train_index'], 'r') as f:
        train_indices = [line.strip() for line in f]
    with open(CONFIG['val_index'], 'r') as f:
        val_indices = [line.strip() for line in f]  
                     
    run_name = f"{CONFIG['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(CONFIG['save_path'], run_name)
    loss_path = os.path.join(run_dir, "loss_log.pkl")
    tmp_path = loss_path + ".tmp"
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Run: {run_name}")
    print(f"Dataset: {len(train_indices)} Training, {len(val_indices)} Validation")
    print(f"Model: {CONFIG['model']} | Classes: {CONFIG['total_classes']}")
        
    # ------------------------
    # Load Model & Optimizer
    # ------------------------
    model  = PathwayModel(num_classes=CONFIG['total_classes'], backbone_name=CONFIG["model"]).to(CONFIG['device'])
    if should_freeze_bn(CONFIG["model"]):
        model.apply(freeze_bn)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    encoder   = IterativeEncoder(CONFIG)
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=CONFIG["device"]).view(1,4,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=CONFIG["device"]).view(1,4,1,1)
    
    # ------------------------
    # Training Loop
    # ------------------------
    global_step = 0
    total_epochs = CONFIG['num_epochs']
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    for epoch in range(total_epochs):
        
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        model.train()
        if should_freeze_bn(CONFIG["model"]):
            model.apply(freeze_bn)
                                    
        # --- TRAIN ---
        train_loss = 0.0
        pbar = tqdm(train_indices, desc="Training")
        for i, idx in enumerate(pbar, start=1):            
            l_train = process_graph(
                idx,
                model, 
                encoder, 
                CONFIG,
                mode = 'train', 
                criterion = criterion,
                optimizer = optimizer, 
                teacher_noise = CONFIG['noise'],
                mean=mean, 
                std=std
            )
            history['train'].append((epoch, global_step, l_train))
            train_loss += l_train
            pbar.set_postfix({'Train': f"{train_loss/i:.3f}"})        
            global_step += 1
        print(f"  Train Avg: {train_loss / max(len(train_indices), 1):.4f}")
        
        # --- VALIDATION (once per epoch) ---
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for v_idx in val_indices:
                l_val = process_graph(
                    v_idx,
                    model,
                    encoder,
                    CONFIG,
                    mode='val',
                    criterion=criterion,
                    optimizer=None,
                    teacher_noise=0.0,
                    mean=mean,
                    std=std
                )
                val_loss += l_val
            avg_val_loss = val_loss / max(len(val_indices), 1)

        history['val'].append((epoch, global_step, float(avg_val_loss)))
        print(f"  Val Avg:   {avg_val_loss:.4f}")

        # --- Save logs/checkpoints ---
        with open(tmp_path, "wb") as f:
            pickle.dump(history, f)
        os.replace(tmp_path, loss_path)

        torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
            print("  [+] New Best Model! Saved.")
                                                    
    print("\nTraining Complete.")