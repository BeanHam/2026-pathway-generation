import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from config import CONFIG
from model import PathwaySegmentationModel
from utils import atomic_json_dump, PathwayInferenceDataset, select_seed_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default = '/data/models/bin/2026_pathway_generation/segmentation_logs/swin_tiny_20260324_152911/model_best.pth')    
    parser.add_argument('--model', type=str, default=CONFIG['model'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--min_distance', type=int, default=50)
    parser.add_argument('--save_prob_png_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PathwaySegmentationModel(backbone_name=args.model,pretrained=False,decoder_dim=int(CONFIG['decoder_dim'])).to(device)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    
    datasets = [
        {
            "name":        "generation_full_data",
            "data_root":   "/data/aiclinic/pathways/generation_full_data/pairs/",
            "index_file":  "/data/aiclinic/pathways/generation_full_data/test.txt",
            "output_json": "/data/aiclinic/pathways/generation_full_data/segmentation_seeds.json",
        },
        {
            "name":        "pathwaybench_seattle",
            "data_root":   "/data/aiclinic/pathways/pathwaybench_seattle/pairs/",
            "index_file":  "/data/aiclinic/pathways/pathwaybench_seattle/test.txt",
            "output_json": "/data/aiclinic/pathways/pathwaybench_seattle/segmentation_seeds.json",            
        },
        {
            "name":        "pathwaybench_dc",
            "data_root":   "/data/aiclinic/pathways/pathwaybench_dc/pairs/",
            "index_file":  "/data/aiclinic/pathways/pathwaybench_dc/test.txt",
            "output_json": "/data/aiclinic/pathways/pathwaybench_dc/segmentation_seeds.json",            
        }
    ]
        
    for d in datasets:
        
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
