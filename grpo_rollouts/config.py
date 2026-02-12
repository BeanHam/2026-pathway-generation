CONFIG = {
    # Path Information
    'data_root':              '/data/aiclinic/pathways/generation_full_data/pairs/',
    'train_index':            '/data/aiclinic/pathways/generation_full_data/train.txt',
    'val_index':              '/data/aiclinic/pathways/generation_full_data/val.txt',
    'save_path':              '/data/models/bin/2026_pathway_generation/grpo_logs_rollouts/',
    'seed':                   0,
    
    # Model Params
    'model':                  "resnet18",
    'num_angles':             36,        
    'num_dists':              10,        
    'stop_class_idx':         360,
    'total_classes':          361,
    
    # Data Gen Parameters
    'crop_size':              0.5,
    'output_size':            (256, 256),
    
    # GRPO Training Hyperparams
    'num_epochs':             10,
    'grad_accum':             8,
    'lr':                     3e-5,
    'weight_decay':           1e-4,        
    'grad_clip':              1.0,
    'kl_beta':                0.05,
    'graphs_per_epoch':       None,
    'val_graphs':             200,
    
    # Sampling Parameters (Fixed)
    'K':                      2,
    'buffer_radius':          0.01,   
    'temperature':            1,
    'max_global_steps':       200,
    
    # Sampling Parameters (Tunable)
    'seed_strategy':          'gt_random',          
    'seeds_per_graph':        1,
}

REFERENCE_DIRS = {
    "resnet18":   "/data/models/bin/2026_pathway_generation/reference_logs/resnet18_ddp_graph_20260209_121921/",
    "resnet50":   "/data/models/bin/2026_pathway_generation/reference_logs/resnet50_ddp_graph_20260209_004201/",
    "swin_tiny":  "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260206_130501/",
}