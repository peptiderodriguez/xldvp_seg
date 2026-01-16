#!/usr/bin/env python3
"""
Run NMJ inference using pre-computed segmentation results.
Uses existing masks/features from run_nmj_segmentation.py output to classify NMJs.
"""

import argparse
import gc
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import h5py

# Use segmentation utilities
from segmentation.io.html_export import percentile_normalize
from segmentation.utils.logging import get_logger, setup_logging, log_parameters
from segmentation.io.czi_loader import get_loader, CZILoader

logger = get_logger(__name__)


def load_classifier(model_path, device):
    """Load trained ResNet18 classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded classifier with val accuracy: {checkpoint['best_acc']:.4f}")
    logger.info(f"  Trained on {checkpoint['num_positive']} positive, {checkpoint['num_negative']} negative")

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
    # centroid is [x, y]
    cx, cy = int(centroid[0]), int(centroid[1])
    crop_size = int(base_size * zoom_factor / 7.5)
    half = crop_size // 2

    h, w = tile_rgb.shape[:2]
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    # Validate crop bounds before extracting
    if y2 <= y1 or x2 <= x1:
        return None

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
    num_batches = (len(crops) + batch_size - 1) // batch_size
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

            # Clear GPU memory periodically (every 10 batches) to prevent OOM
            batch_num = i // batch_size
            if batch_num > 0 and batch_num % 10 == 0:
                del batch_tensors, outputs, probs, preds
                torch.cuda.empty_cache()
                gc.collect()

    # Clear GPU memory at end of batch processing
    torch.cuda.empty_cache()
    gc.collect()

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
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM for faster tile extraction (default: True)')
    parser.add_argument('--no-load-to-ram', dest='load_to_ram', action='store_false',
                        help='Disable RAM loading (use less memory but slower)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load classifier
    logger.info("Loading classifier...")
    model = load_classifier(args.model_path, device)
    transform = get_transform()

    # Load CZI using shared loader (RAM-first for better performance)
    logger.info("Loading CZI file...")
    czi_path = Path(args.czi_path)
    loader = CZILoader(czi_path, load_to_ram=args.load_to_ram, channel=args.channel)

    width, height = loader.mosaic_size
    pixel_size = loader.get_pixel_size()
    slide_name = loader.slide_name

    logger.info(f"  Mosaic dimensions: {width} x {height}")
    logger.info(f"  Pixel size: {pixel_size:.4f} um/px")
    logger.info(f"  RAM loading: {args.load_to_ram}")

    # Find segmentation directory
    if args.segmentation_dir:
        seg_dir = Path(args.segmentation_dir)
    else:
        seg_dir = Path(args.output_dir) / slide_name / "tiles"

    if not seg_dir.exists():
        logger.error(f"Segmentation directory not found: {seg_dir}")
        logger.error("Run run_nmj_segmentation.py first to generate segmentation results.")
        return

    # Get list of tiles with detections
    tile_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir() and d.name.startswith('tile_')])
    logger.info(f"Found {len(tile_dirs)} tiles with detections")

    # Setup output
    output_dir = Path(args.output_dir) / slide_name / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process tiles
    all_nmjs = []
    total_candidates = 0
    total_nmjs = 0
    tile_size = args.tile_size

    logger.info("Classifying pre-detected NMJs...")
    for tile_idx, tile_dir in enumerate(tqdm(tile_dirs, desc="Tiles")):
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

            # Read tile using shared loader
            tile_data = loader.get_tile(tile_x, tile_y, tile_size)
            if tile_data is None or tile_data.size == 0:
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
                    # centroid from features is [x, y]
                    cx, cy = res['centroid']
                    global_x = tile_x + cx
                    global_y = tile_y + cy

                    nmj_info = {
                        'id': f"{tile_x}_{tile_y}_{res['id']}",
                        'tile_x': tile_x,
                        'tile_y': tile_y,
                        'local_centroid': [cx, cy],  # [x, y]
                        'global_centroid': [global_x, global_y],  # [x, y]
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

            # Clear GPU memory periodically (every 50 tiles) to prevent OOM on long runs
            if tile_idx > 0 and tile_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing {tile_dir.name}: {e}")
            continue

    # Final GPU memory cleanup after all tiles processed
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("=" * 50)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total candidates from segmentation: {total_candidates}")
    logger.info(f"Total NMJs classified as positive: {total_nmjs}")
    if total_candidates > 0:
        logger.info(f"Classification rate: {total_nmjs/total_candidates*100:.1f}%")

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

    logger.info(f"Results saved to: {results_file}")

    # Summary stats
    if all_nmjs:
        areas = [n['area_um2'] for n in all_nmjs]
        confidences = [n['confidence'] for n in all_nmjs]
        logger.info("NMJ Statistics:")
        logger.info(f"  Area range: {min(areas):.1f} - {max(areas):.1f} um^2")
        logger.info(f"  Mean area: {np.mean(areas):.1f} um^2")
        logger.info(f"  Mean confidence: {np.mean(confidences):.3f}")


if __name__ == '__main__':
    main()
