import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG
from model import PathwaySegmentationModel
from utils import atomic_json_dump, PathwayInferenceDataset, select_seed_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--model', type=str, default=CONFIG['model'])
    parser.add_argument('--threshold', type=float, default=CONFIG['prob_threshold'])
    parser.add_argument('--k', type=int, default=CONFIG['seed_top_k'])
    parser.add_argument('--min_distance', type=int, default=CONFIG['seed_min_distance'])
    parser.add_argument('--min_component_size', type=int, default=CONFIG['min_component_size'])
    parser.add_argument('--save_prob_png_dir', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PathwaySegmentationModel(
        backbone_name=args.model,
        pretrained=False,
        decoder_dim=int(CONFIG['decoder_dim']),
    ).to(device)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = PathwayInferenceDataset(
        data_root=args.data_root,
        index_path=args.index_file,
        output_size=CONFIG['output_size'],
    )

    if args.save_prob_png_dir is not None:
        os.makedirs(args.save_prob_png_dir, exist_ok=True)

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
            prob_map = probs[0, 0].cpu().numpy()

            seeds = select_seed_points(
                prob_map,
                k=args.k,
                threshold=args.threshold,
                min_distance=args.min_distance,
                min_component_size=args.min_component_size,
            )

            output[str(sample_id)] = {
                'points_px': [[p['x_px'], p['y_px_top_left']] for p in seeds],
                'points_norm': [[p['x_norm'], p['y_norm']] for p in seeds],
                'scores': [p['score'] for p in seeds],
            }

            if args.save_prob_png_dir is not None:
                out_png = os.path.join(args.save_prob_png_dir, f'{sample_id}.png')
                cv2.imwrite(out_png, (prob_map * 255.0).clip(0, 255).astype('uint8'))

    atomic_json_dump(output, args.output_json)
    print(f'Saved: {args.output_json}')
