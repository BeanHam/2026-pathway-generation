CONFIG = {
    # Path Information
    'data_root':              '/path/to/DATASET_ROOT/pairs/',
    'train_index':            '/path/to/DATASET_ROOT/train.txt',
    'val_index':              '/path/to/DATASET_ROOT/val.txt',
    'save_path':              '/path/to/save/reference_logs/',
    
    # Model Params
    'model':                  "resnet18",
    'num_angles':             36,
    'num_dists':              10,
    'stop_class_idx':         360,
    'total_classes':          361,
    
    # Data Gen Parameters
    'noise':                  0.02,
    'crop_size':              0.5,
    'output_size':            (256, 256),
    'max_steps':              200,
    
    # Training Hyperparams
    'num_epochs':             50,
    'accum_nodes':            16,
    'lr':                     1e-4,
    'weight_decay':           1e-4,
    'grad_clip':              1.0,
}