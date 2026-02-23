CONFIG = {
    # Path Information
    'data_root':              '/data/aiclinic/pathways/generation_full_data/pairs/',
    'train_index':            '/data/aiclinic/pathways/generation_full_data/train.txt',
    'val_index':              '/data/aiclinic/pathways/generation_full_data/val.txt',
    'save_path':              '/data/models/bin/2026_pathway_generation/grpo_logs_steps_immediate/',
    'seed':                   0,
            
    # Step-wise credit assignment
    'step_advantages':        True,
    'step_iou_stride':        1,
    'step_iou_max_edges':     100,

    # Multiplicative Update Hyperparameters
    'local_clip':             3.0,
    'adv_std_floor':          1e-3,
    
    # debug
    'debug_one_graph':        True,
    'debug_print_rollout':    0,
    
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
    'K':                      4,
    'buffer_radius':          0.01,   
    'temperature':            1,
    'max_global_steps':       200,
    
    # Sampling Parameters (Tunable)
    'seed_strategy':          'gt_random',          
    'seeds_per_graph':        1,
}

REFERENCE_DIRS = {
    "swin_tiny":      "/data/models/bin/2026_pathway_generation/reference_logs/swin_tiny_ddp_graph_20260212_123147_cont/",
    "resnet18":       "/data/models/bin/2026_pathway_generation/reference_logs/resnet18_20260217_102812/",
    "vit_small":      "/data/models/bin/2026_pathway_generation/reference_logs/vit_small_20260217_102936/"
}