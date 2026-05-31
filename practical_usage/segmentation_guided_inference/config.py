CONFIG = {
    # Path Information
    'data_root':              '/path/to/DATASET_ROOT/pairs/',
    'train_index':            '/path/to/DATASET_ROOT/train.txt',
    'val_index':              '/path/to/DATASET_ROOT/val.txt',
    'save_path':              '/path/to/save/grpo_global_logs/',
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
    'num_epochs':             20,
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