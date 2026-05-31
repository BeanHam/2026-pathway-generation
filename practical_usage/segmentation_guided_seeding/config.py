CONFIG = {
    # Path Information
    'data_root':              '/path/to/DATASET_ROOT/pairs/',
    'train_index':            '/path/to/DATASET_ROOT/train.txt',
    'val_index':              '/path/to/DATASET_ROOT/val.txt',
    'test_index':             '/path/to/DATASET_ROOT/test.txt',
    'save_path':              '/path/to/save/segmentation_logs/',
    
    # Model Params
    'model':                'swin_tiny',
    'pretrained':           True,
    'decoder_dim':          96,

    # Data Parameters
    'output_size':          (896, 896),
    'mask_line_width':      5,

    # Training Hyperparams
    'num_epochs':           50,
    'batch_size':           8,
    'num_workers':          4,
    'lr':                   1e-4,
    'weight_decay':         1e-4,
    'grad_clip':            1.0,
    'pos_weight':           5.0,
    
    # Inference / seeding
    'prob_threshold':       0.5,
    'seed_top_k':           200,
    'seed_min_distance':    50,
}