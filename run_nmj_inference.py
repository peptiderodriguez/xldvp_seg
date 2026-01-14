#!/usr/bin/env python3
"""
Run NMJ inference using pre-computed segmentation results.
Uses existing masks/features from run_nmj_segmentation.py output to classify NMJs.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import h5py
from aicspylibczi import CziFile


def percentile_normalize(img, p_low=5, p_high=95):
    """Normalize image using percentiles."""
    img = img.astype(np.float32)
    p_lo = np.percentile(img, p_low)
    p_hi = np.percentile(img, p_high)
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - p_lo) / (p_hi - p_lo)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def load_classifier(model_path, device):
    """Load trained ResNet18 classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded classifier with val accuracy: {checkpoint['best_acc']:.4f}")
    print(f"  Trained on {checkpoint['num_positive']} positive, {checkpoint['num_negative']} negative")

    return model


def get_transform():
    """Get inference transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def extract_crop(tile_rgb, centroid, zoom_factor=7.5, base_size=300):
    """Extract crop centered on centroid with zoom factor."""
    cy, cx = int(centroid[0]), int(centroid[1])
    crop_size = int(base_size * zoom_factor / 7.5)
    half = crop_size // 2

    h, w = tile_rgb.shape[:2]
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    crop = tile_rgb[y1:y2, x1:x2].copy()

    # Normalize
    crop = percentile_normalize(crop)

    # Resize to base_size
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        pil_img = Image.fromarray(crop)
        pil_img = pil_img.resize((base_size, base_size), Image.LANCZOS)
        return pil_img
    return None


def classify_candidates(model, candidates, tile_rgb, transform, device, batch_size=32):
    """Classify candidates using the trained model."""
    if not candidates:
        return []

    # Extract crops
    crops = []
    valid_indices = []
    for i, cand in enumerate(candidates):
        crop = extract_crop(tile_rgb, cand['centroid'])
        if crop is not None:
            crops.append(crop)
            valid_indices.append(i)

    if not crops:
        return []

    # Batch inference
    results = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]

            # Transform and stack
            batch_tensors = torch.stack([transform(c) for c in batch_crops]).to(device)

            # Forward pass
            outputs = model(batch_tensors)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for j, (pred, prob) in enumerate(zip(preds, probs)):
                cand_idx = batch_indices[j]
                cand = candidates[cand_idx]
                results.append({
                    **cand,
                    'is_nmj': pred.item() == 1,
                    'confidence': prob[pred].item(),
                    'prob_nmj': prob[1].item(),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description='Run NMJ inference using pre-computed segmentation')
    parser.add_argument('--czi-path', type=str, required=True, help='Path to CZI file')
    parser.add_argument('--segmentation-dir', type=str, default=None,
                        help='Directory with pre-computed segmentation (tiles folder). Auto-detected if not specified.')
    parser.add_argument('--model-path', type=str,
                        default='/home/dude/nmj_output/nmj_classifier.pth',
                        help='Path to trained classifier')
    parser.add_argument('--output-dir', type=str,
                        default='/home/dude/nmj_output',
                        help='Output directory')
    parser.add_argument('--channel', type=int, default=1, help='Channel index')
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Confidence threshold for positive classification')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load classifier
    print("\nLoading classifier...")
    model = load_classifier(args.model_path, device)
    transform = get_transform()

    # Load CZI
    print("\nLoading CZI file...")
    czi_path = Path(args.czi_path)
    reader = CziFile(str(czi_path))

    bbox = reader.get_mosaic_bounding_box()
    print(f"  Mosaic dimensions: {bbox.w} x {bbox.h}")

    # Get pixel size
    metadata = reader.meta
    pixel_size = 0.22
    try:
        scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if scaling is not None:
            pixel_size = float(scaling.text) * 1e6
    except:
        pass
    print(f"  Pixel size: {pixel_size:.4f} um/px")

    # Find segmentation directory
    slide_name = czi_path.stem
    if args.segmentation_dir:
        seg_dir = Path(args.segmentation_dir)
    else:
        seg_dir = Path(args.output_dir) / slide_name / "tiles"

    if not seg_dir.exists():
        print(f"\nERROR: Segmentation directory not found: {seg_dir}")
        print("Run run_nmj_segmentation.py first to generate segmentation results.")
        return

    # Get list of tiles with detections
    tile_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir() and d.name.startswith('tile_')])
    print(f"\nFound {len(tile_dirs)} tiles with detections")

    # Setup output
    output_dir = Path(args.output_dir) / slide_name / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process tiles
    all_nmjs = []
    total_candidates = 0
    total_nmjs = 0
    tile_size = args.tile_size

    print("\nClassifying pre-detected NMJs...")
    for tile_dir in tqdm(tile_dirs, desc="Tiles"):
        try:
            # Parse tile coordinates from directory name
            parts = tile_dir.name.split('_')
            tile_x = int(parts[1])
            tile_y = int(parts[2])

            # Load features
            features_file = tile_dir / "nmj_features.json"
            if not features_file.exists():
                continue

            with open(features_file) as f:
                features = json.load(f)

            if not features:
                continue

            # Read tile from CZI
            tile_data = reader.read_mosaic(
                region=(tile_x, tile_y, tile_size, tile_size),
                scale_factor=1,
                C=args.channel
            )

            if tile_data is None or tile_data.size == 0:
                continue

            tile_data = np.squeeze(tile_data)
            if tile_data.ndim != 2:
                continue

            # Create RGB for crop extraction
            tile_rgb = np.stack([tile_data] * 3, axis=-1)

            # Convert features to candidate format
            candidates = []
            for feat in features:
                candidates.append({
                    'id': feat['id'],
                    'centroid': feat['centroid'],
                    'area': feat['area'],
                    'skeleton_length': feat['skeleton_length'],
                    'elongation': feat['elongation'],
                    'mean_intensity': feat['mean_intensity'],
                    'eccentricity': feat.get('eccentricity', 0),
                })

            total_candidates += len(candidates)

            # Classify candidates
            results = classify_candidates(model, candidates, tile_rgb, transform, device)

            # Filter by confidence
            for res in results:
                if res['is_nmj'] and res['confidence'] >= args.confidence_threshold:
                    cy, cx = res['centroid']
                    global_y = tile_y + cy
                    global_x = tile_x + cx

                    nmj_info = {
                        'id': f"{tile_x}_{tile_y}_{res['id']}",
                        'tile_x': tile_x,
                        'tile_y': tile_y,
                        'local_centroid': [cy, cx],
                        'global_centroid': [global_y, global_x],
                        'area_px': res['area'],
                        'area_um2': res['area'] * pixel_size * pixel_size,
                        'skeleton_length': res['skeleton_length'],
                        'elongation': res['elongation'],
                        'mean_intensity': res['mean_intensity'],
                        'confidence': res['confidence'],
                        'prob_nmj': res['prob_nmj'],
                    }
                    all_nmjs.append(nmj_info)
                    total_nmjs += 1

        except Exception as e:
            print(f"\nError processing {tile_dir.name}: {e}")
            continue

    print(f"\n" + "="*50)
    print("INFERENCE COMPLETE")
    print("="*50)
    print(f"Total candidates from segmentation: {total_candidates}")
    print(f"Total NMJs classified as positive: {total_nmjs}")
    if total_candidates > 0:
        print(f"Classification rate: {total_nmjs/total_candidates*100:.1f}%")

    # Save results
    results_file = output_dir / "nmj_detections.json"
    with open(results_file, 'w') as f:
        json.dump({
            'slide_name': slide_name,
            'total_tiles_with_detections': len(tile_dirs),
            'total_candidates': total_candidates,
            'total_nmjs': total_nmjs,
            'pixel_size_um': pixel_size,
            'confidence_threshold': args.confidence_threshold,
            'nmjs': all_nmjs,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary stats
    if all_nmjs:
        areas = [n['area_um2'] for n in all_nmjs]
        confidences = [n['confidence'] for n in all_nmjs]
        print(f"\nNMJ Statistics:")
        print(f"  Area range: {min(areas):.1f} - {max(areas):.1f} um^2")
        print(f"  Mean area: {np.mean(areas):.1f} um^2")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")


if __name__ == '__main__':
    main()
