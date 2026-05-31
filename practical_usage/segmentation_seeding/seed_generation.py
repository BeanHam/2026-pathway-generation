import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from config import CONFIG
from model import PathwaySegmentationModel
from utils import atomic_json_dump, PathwayInferenceDataset, select_seed_points

DATASETS = [
    {
        "name":         "DATASET_NAME",
        "data_root":    "/path/to/DATASET_ROOT/pairs/",
        "index_file":   "/path/to/DATASET_ROOT/test.txt",
        "output_json":  "/path/to/DATASET_ROOT/segmentation_seeds.json",
    },
]
CHECKPOINT = '/path/to/save/segmentation_logs/resnet18/model_best.pth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=CONFIG['model'])
    parser.add_argument('--threshold', type=float, default=CONFIG['prob_threshold'])
    parser.add_argument('--k', type=int, default=CONFIG['seed_top_k'])
    parser.add_argument('--min_distance', type=int, default=CONFIG['seed_min_distance'])
    parser.add_argument('--save_prob_png_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PathwaySegmentationModel(backbone_name=args.model,pretrained=False,decoder_dim=int(CONFIG['decoder_dim'])).to(device)
    state = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
        
    for d in DATASETS:
        
        print(f"Generating Seeds on {d['name']}...")
        
        dataset = PathwayInferenceDataset(
            data_root=d['data_root'],
            index_path=d['index_file'],
            output_size=CONFIG['output_size'],
        )
        
        output = {}

        with torch.no_grad():
            for image, sample_id, orig_h, orig_w in tqdm(dataset, desc='Inference'):
                logits = model(image.unsqueeze(0).to(device))
                probs = torch.sigmoid(logits)
                probs = F.interpolate(
                    probs,
                    size=(int(orig_h), int(orig_w)),
                    mode='bilinear',
                    align_corners=False,
                )
                prob_map = probs[0, 0].cpu().detach().numpy()

                seeds = select_seed_points(
                    prob_map,
                    k=args.k,
                    threshold=args.threshold,
                    min_distance=args.min_distance
                )

                output[str(sample_id)] = {
                    'points_px': [[p['x_px'], p['y_px_top_left']] for p in seeds],
                    'points_norm': [[p['x_norm'], p['y_norm']] for p in seeds],
                    'scores': [p['score'] for p in seeds],
                }
                
        atomic_json_dump(output, d['output_json'])
        print(f"Saved: {d['output_json']}")
