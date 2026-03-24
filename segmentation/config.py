CONFIG = {
    # Path Information
    'data_root':            '/data/aiclinic/pathways/generation_full_data/pairs/',
    'train_index':          '/data/aiclinic/pathways/generation_full_data/train.txt',
    'val_index':            '/data/aiclinic/pathways/generation_full_data/val.txt',
    'save_path':            '/data/models/bin/2026_pathway_generation/segmentation_logs/',

    # Model Params
    'model':                'swin_tiny',   # {'swin_tiny', 'swin_small'}
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
    'prob_threshold':       0.35,
    'seed_top_k':           10,
    'seed_min_distance':    20,
    'min_component_size':   32,
}