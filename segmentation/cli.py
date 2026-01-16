#!/usr/bin/env python3
"""
Unified CLI for the cell segmentation pipeline.

Usage:
    segmentation run nmj --czi-path /path/to/slide.czi
    segmentation run mk --input-dir /path/to/slides --pattern "*.czi"
    segmentation export html --results /path/to/detections.json
    segmentation validate /path/to/file.json

Subcommands:
    run         Run detection on slides
    export      Export results to various formats
    validate    Validate JSON files
    info        Show information about files/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

import numpy as np

from segmentation.utils.logging import setup_logging, get_logger, log_parameters
from segmentation.utils.config import DEFAULT_PATHS, get_output_dir
from segmentation.processing.batch import collect_slides, BatchProcessor, create_batch_summary_html
from segmentation.utils.schemas import infer_and_validate


# =============================================================================
# DETECTOR FACTORY
# =============================================================================

def get_detector(
    cell_type: str,
    model_path: Optional[Path] = None,
    confidence_threshold: float = 0.5,
    min_area_um: float = 200,
    max_area_um: float = 2000,
    min_vessel_diameter: float = 10,
    max_vessel_diameter: float = 1000,
    pixel_size_um: float = 0.22,
    **kwargs
) -> Callable[[np.ndarray, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Factory function that returns the appropriate detector based on cell_type.

    Args:
        cell_type: Type of cell to detect (nmj, mk, cell, vessel)
        model_path: Path to NMJ classifier model (required for nmj)
        confidence_threshold: Confidence threshold for NMJ classification
        min_area_um: Minimum cell area in um^2 (for mk/cell)
        max_area_um: Maximum cell area in um^2 (for mk/cell)
        min_vessel_diameter: Minimum vessel diameter in um (for vessel)
        max_vessel_diameter: Maximum vessel diameter in um (for vessel)
        pixel_size_um: Pixel size in micrometers
        **kwargs: Additional cell-type specific parameters

    Returns:
        Detector function that takes (tile_data, **kwargs) and returns detections
    """
    if cell_type == "nmj":
        return _create_nmj_detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            pixel_size_um=pixel_size_um,
            **kwargs
        )
    elif cell_type in ("mk", "cell"):
        return _create_mk_cell_detector(
            cell_type=cell_type,
            min_area_um=min_area_um,
            max_area_um=max_area_um,
            pixel_size_um=pixel_size_um,
            **kwargs
        )
    elif cell_type == "vessel":
        return _create_vessel_detector(
            min_diameter_um=min_vessel_diameter,
            max_diameter_um=max_vessel_diameter,
            pixel_size_um=pixel_size_um,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


def _create_nmj_detector(
    model_path: Optional[Path] = None,
    confidence_threshold: float = 0.5,
    pixel_size_um: float = 0.1725,
    intensity_percentile: float = 97,
    min_area: int = 50,
    min_skeleton_length: int = 20,
    min_elongation: float = 0.8,
    **kwargs
) -> Callable:
    """
    Create NMJ detector with classifier model.

    The detector performs two stages:
    1. Intensity threshold + elongation filter to find candidates
    2. CNN classifier to classify candidates as NMJ or not
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    from skimage.morphology import skeletonize, remove_small_objects, binary_opening, binary_closing, disk
    from skimage.measure import label, regionprops

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load classifier if model path provided
    classifier = None
    if model_path and Path(model_path).exists():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        classifier = model

    # Transform for classifier
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def nmj_detector(tile_data: np.ndarray, **det_kwargs) -> List[Dict[str, Any]]:
        """Detect NMJs in a tile."""
        from segmentation.io.html_export import percentile_normalize

        # Convert to grayscale if needed
        if tile_data.ndim == 3:
            gray = np.mean(tile_data[:, :, :3], axis=2)
        else:
            gray = tile_data.astype(float)

        # Threshold bright regions
        threshold = np.percentile(gray, intensity_percentile)
        bright_mask = gray > threshold

        # Morphological cleanup
        bright_mask = binary_opening(bright_mask, disk(1))
        bright_mask = binary_closing(bright_mask, disk(2))
        bright_mask = remove_small_objects(bright_mask, min_size=min_area)

        # Label connected components
        labeled = label(bright_mask)
        props = regionprops(labeled, intensity_image=gray)

        # Create RGB for crops
        if tile_data.ndim == 2:
            tile_rgb = np.stack([tile_data] * 3, axis=-1)
        else:
            tile_rgb = tile_data

        detections = []
        for prop in props:
            if prop.area < min_area:
                continue

            # Check elongation
            region_mask = labeled == prop.label
            skeleton = skeletonize(region_mask)
            skeleton_length = skeleton.sum()
            elongation = skeleton_length / np.sqrt(prop.area) if prop.area > 0 else 0

            if skeleton_length < min_skeleton_length or elongation < min_elongation:
                continue

            cy, cx = prop.centroid  # regionprops returns (row, col)

            # If we have a classifier, use it
            is_nmj = True
            confidence = 1.0
            prob_nmj = 1.0

            if classifier is not None:
                # Extract crop for classification
                crop_size = 300
                half = crop_size // 2
                h, w = tile_rgb.shape[:2]
                y1 = max(0, int(cy) - half)
                y2 = min(h, int(cy) + half)
                x1 = max(0, int(cx) - half)
                x2 = min(w, int(cx) + half)

                # Validate crop bounds before extracting
                if y2 <= y1 or x2 <= x1:
                    continue

                crop = tile_rgb[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    crop = percentile_normalize(crop)
                    pil_img = Image.fromarray(crop)
                    pil_img = pil_img.resize((crop_size, crop_size), Image.LANCZOS)

                    with torch.no_grad():
                        tensor = transform(pil_img).unsqueeze(0).to(device)
                        outputs = classifier(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        pred = torch.argmax(outputs, dim=1).item()
                        prob_nmj = probs[0, 1].item()
                        is_nmj = pred == 1
                        confidence = probs[0, pred].item()

            # Only include if classified as NMJ with sufficient confidence
            if is_nmj and confidence >= confidence_threshold:
                detections.append({
                    'centroid': [float(cx), float(cy)],  # [x, y]
                    'features': {
                        'area': int(prop.area),
                        'area_um2': float(prop.area * pixel_size_um * pixel_size_um),
                        'skeleton_length': int(skeleton_length),
                        'elongation': float(elongation),
                        'mean_intensity': float(prop.mean_intensity),
                        'eccentricity': float(prop.eccentricity),
                        'confidence': float(confidence),
                        'prob_nmj': float(prob_nmj),
                    }
                })

        return detections

    return nmj_detector


def _create_mk_cell_detector(
    cell_type: str,
    min_area_um: float = 200,
    max_area_um: float = 2000,
    pixel_size_um: float = 0.22,
    **kwargs
) -> Callable:
    """
    Create MK/HSPC detector using the UnifiedSegmenter.

    This loads SAM2 and optionally Cellpose for detection.
    """
    # Import here to avoid circular imports and heavy imports at module level
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from run_segmentation import UnifiedSegmenter

    # Initialize segmenter once (expensive)
    segmenter = None

    def mk_cell_detector(tile_data: np.ndarray, **det_kwargs) -> List[Dict[str, Any]]:
        nonlocal segmenter

        # Lazy initialization
        if segmenter is None:
            load_cellpose = (cell_type == "cell")
            segmenter = UnifiedSegmenter(load_sam2=True, load_cellpose=load_cellpose)

        # Convert to RGB if needed
        if tile_data.ndim == 2:
            tile_rgb = np.stack([tile_data] * 3, axis=-1)
        else:
            tile_rgb = tile_data

        # Convert uint16 to uint8 if needed
        if tile_rgb.dtype == np.uint16:
            tile_rgb = (tile_rgb / 256).astype(np.uint8)

        # Convert area thresholds from um^2 to pixels
        min_area_px = min_area_um / (pixel_size_um ** 2)
        max_area_px = max_area_um / (pixel_size_um ** 2)

        params = {
            'mk_min_area': int(min_area_px),
            'mk_max_area': int(max_area_px),
        }

        # Run detection
        if cell_type == "mk":
            masks, features_list = segmenter.detect_mk(tile_rgb, params)
        else:  # cell
            masks, features_list = segmenter.detect_cell(tile_rgb, params)

        # Convert to detection format
        detections = []
        for feat in features_list:
            center = feat['center']  # Already [x, y]
            features = feat.get('features', {})
            features['area_um2'] = features.get('area', 0) * pixel_size_um * pixel_size_um

            detections.append({
                'centroid': center,
                'features': features,
            })

        return detections

    return mk_cell_detector


def _create_vessel_detector(
    min_diameter_um: float = 10,
    max_diameter_um: float = 1000,
    pixel_size_um: float = 0.22,
    min_wall_thickness_um: float = 2,
    max_aspect_ratio: float = 4.0,
    min_circularity: float = 0.3,
    **kwargs
) -> Callable:
    """
    Create vessel detector using the UnifiedSegmenter.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from run_segmentation import UnifiedSegmenter

    segmenter = None

    def vessel_detector(tile_data: np.ndarray, **det_kwargs) -> List[Dict[str, Any]]:
        nonlocal segmenter

        # Lazy initialization
        if segmenter is None:
            segmenter = UnifiedSegmenter(load_sam2=True, load_cellpose=False)

        # Convert to RGB if needed
        if tile_data.ndim == 2:
            tile_rgb = np.stack([tile_data] * 3, axis=-1)
        else:
            tile_rgb = tile_data

        # Convert uint16 to uint8 if needed
        if tile_rgb.dtype == np.uint16:
            tile_rgb = (tile_rgb / 256).astype(np.uint8)

        params = {
            'min_vessel_diameter_um': min_diameter_um,
            'max_vessel_diameter_um': max_diameter_um,
            'min_wall_thickness_um': min_wall_thickness_um,
            'max_aspect_ratio': max_aspect_ratio,
            'min_circularity': min_circularity,
            'pixel_size_um': pixel_size_um,
        }

        # Run detection
        masks, features_list = segmenter.detect_vessel(tile_rgb, params)

        # Convert to detection format
        detections = []
        for feat in features_list:
            center = feat['center']  # Already [x, y]
            features = feat.get('features', {})

            detections.append({
                'centroid': center,
                'features': features,
            })

        return detections

    return vessel_detector


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog="segmentation",
        description="Unified cell segmentation pipeline for microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run NMJ detection on a single slide
  segmentation run nmj --czi-path /path/to/slide.czi

  # Run MK detection on multiple slides
  segmentation run mk --input-dir /path/to/slides --sample-fraction 0.1

  # Run batch processing from a file list
  segmentation run vessel --batch-file slides.txt --load-to-ram

  # Export results to HTML
  segmentation export html --results /path/to/detections.json

  # Validate a JSON file
  segmentation validate /path/to/config.json
""",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress most output",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === RUN command ===
    run_parser = subparsers.add_parser(
        "run",
        help="Run cell detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_subparsers = run_parser.add_subparsers(dest="cell_type", help="Cell type to detect")

    # Create run subparser for each cell type
    for cell_type in ["nmj", "mk", "cell", "vessel"]:
        cell_parser = run_subparsers.add_parser(
            cell_type,
            help=f"Detect {cell_type.upper()} cells",
        )
        _add_run_arguments(cell_parser, cell_type)

    # === EXPORT command ===
    export_parser = subparsers.add_parser(
        "export",
        help="Export results to various formats",
    )
    export_subparsers = export_parser.add_subparsers(dest="format", help="Export format")

    # HTML export
    html_parser = export_subparsers.add_parser("html", help="Export to HTML annotation interface")
    html_parser.add_argument("--results", "-r", type=Path, required=True, help="Path to detections JSON")
    html_parser.add_argument("--czi-path", type=Path, help="Path to CZI file (for image extraction)")
    html_parser.add_argument("--output-dir", "-o", type=Path, help="Output directory")
    html_parser.add_argument("--samples-per-page", type=int, default=300, help="Samples per page")

    # CSV export
    csv_parser = export_subparsers.add_parser("csv", help="Export coordinates to CSV")
    csv_parser.add_argument("--results", "-r", type=Path, required=True, help="Path to detections JSON")
    csv_parser.add_argument("--output", "-o", type=Path, help="Output CSV file")

    # === VALIDATE command ===
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate JSON files against schemas",
    )
    validate_parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="JSON files to validate",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first validation error",
    )

    # === INFO command ===
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about files or results",
    )
    info_parser.add_argument(
        "path",
        type=Path,
        help="Path to file or directory",
    )

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser, cell_type: str) -> None:
    """Add common run arguments to a cell type parser."""

    # Input options (mutually supportive, not exclusive)
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--czi-path",
        type=Path,
        help="Path to single CZI file",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing CZI files",
    )
    input_group.add_argument(
        "--pattern",
        default="*.czi",
        help="Glob pattern for matching files (default: *.czi)",
    )
    input_group.add_argument(
        "--batch-file",
        type=Path,
        help="Text file with paths (one per line)",
    )
    input_group.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o",
        type=Path,
        help=f"Output directory (default: {get_output_dir(cell_type)})",
    )
    output_group.add_argument(
        "--experiment-name",
        help="Name for this experiment (affects localStorage keys)",
    )
    output_group.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML export",
    )
    output_group.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--channel", "-c",
        type=int,
        default=0 if cell_type != "nmj" else 1,
        help="Channel to process",
    )
    proc_group.add_argument(
        "--tile-size",
        type=int,
        default=3000,
        help="Tile size in pixels (default: 3000)",
    )
    proc_group.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of tiles to sample (default: 1.0 = all)",
    )
    proc_group.add_argument(
        "--load-to-ram",
        action="store_true",
        help="Load slide into RAM for faster processing (recommended for network mounts)",
    )

    # Cell-type specific options
    if cell_type == "nmj":
        nmj_group = parser.add_argument_group("NMJ Options")
        nmj_group.add_argument(
            "--model-path",
            type=Path,
            default=Path(DEFAULT_PATHS["nmj_model_path"]),
            help="Path to NMJ classifier model",
        )
        nmj_group.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Confidence threshold for positive classification",
        )

    elif cell_type in ("mk", "cell"):
        mk_group = parser.add_argument_group("MK/HSPC Options")
        mk_group.add_argument(
            "--min-area-um",
            type=float,
            default=200 if cell_type == "mk" else 50,
            help="Minimum cell area in µm²",
        )
        mk_group.add_argument(
            "--max-area-um",
            type=float,
            default=2000 if cell_type == "mk" else 500,
            help="Maximum cell area in µm²",
        )

    elif cell_type == "vessel":
        vessel_group = parser.add_argument_group("Vessel Options")
        vessel_group.add_argument(
            "--min-diameter",
            type=float,
            default=10,
            help="Minimum vessel diameter in µm",
        )
        vessel_group.add_argument(
            "--max-diameter",
            type=float,
            default=1000,
            help="Maximum vessel diameter in µm",
        )
        vessel_group.add_argument(
            "--cd31-channel",
            type=int,
            help="CD31 channel for validation",
        )


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    logger = get_logger(__name__)

    # Collect slides
    slides = collect_slides(
        input_dir=args.input_dir,
        pattern=args.pattern,
        batch_file=args.batch_file,
        paths=[args.czi_path] if args.czi_path else None,
        recursive=args.recursive,
    )

    if not slides:
        logger.error("No slides found to process")
        return 1

    # Log parameters
    log_parameters(logger, {
        "cell_type": args.cell_type,
        "slides": len(slides),
        "channel": args.channel,
        "tile_size": args.tile_size,
        "sample_fraction": args.sample_fraction,
        "load_to_ram": args.load_to_ram,
        "output_dir": args.output_dir or get_output_dir(args.cell_type),
    })

    # Build detector kwargs based on cell type
    detector_kwargs = _build_detector_kwargs(args)

    # Get pixel size (will be updated per-slide from CZI metadata)
    from segmentation.utils.config import DEFAULT_CONFIG, get_pixel_size
    pixel_size_um = get_pixel_size(DEFAULT_CONFIG, args.cell_type)

    # Create detector
    logger.info(f"Initializing {args.cell_type.upper()} detector...")
    try:
        detector = get_detector(
            cell_type=args.cell_type,
            pixel_size_um=pixel_size_um,
            **detector_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1

    # For single slide, use DetectionPipeline directly
    if len(slides) == 1:
        from segmentation.processing.pipeline import DetectionPipeline

        slide = slides[0]
        logger.info(f"Processing single slide: {slide.name}")

        try:
            with DetectionPipeline(
                czi_path=slide.path,
                cell_type=args.cell_type,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
                channel=args.channel,
                tile_size=args.tile_size,
                load_to_ram=args.load_to_ram,
            ) as pipeline:
                # Update pixel size from actual CZI metadata
                actual_pixel_size = pipeline.pixel_size
                logger.info(f"Pixel size from CZI: {actual_pixel_size:.4f} um/px")

                # Run detection
                detections = pipeline.process_tiles(
                    detector_fn=detector,
                    sample_fraction=args.sample_fraction,
                )

                logger.info(f"Found {len(detections)} {args.cell_type.upper()} detections")

                # Export results
                pipeline.export_results(detections)

                if not args.no_csv:
                    pipeline.export_csv(detections)

                if not args.no_html and detections:
                    # Extract images for HTML export
                    samples = _extract_samples_for_html(
                        detections=detections,
                        pipeline=pipeline,
                        cell_type=args.cell_type
                    )
                    if samples:
                        pipeline.export_html(samples=samples)

                # Save config
                pipeline.save_config()

                logger.info(f"Results saved to: {pipeline.output_dir}")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # For multiple slides, use BatchProcessor
    else:
        processor = BatchProcessor(
            slides=slides,
            cell_type=args.cell_type,
            output_base=args.output_dir,
            channel=args.channel,
            tile_size=args.tile_size,
            sample_fraction=args.sample_fraction,
            load_to_ram=args.load_to_ram,
            experiment_name=args.experiment_name,
        )

        # Run batch processing
        try:
            result = processor.run(
                detector_fn=detector,
                export_html=not args.no_html,
                export_csv=not args.no_csv,
                continue_on_error=True,
            )

            # Create batch summary HTML
            create_batch_summary_html(result)

            logger.info(f"Batch processing complete:")
            logger.info(f"  Completed: {result.completed}/{result.total_slides}")
            logger.info(f"  Total detections: {result.total_detections}")
            logger.info(f"  Output: {result.output_dir}")

            if result.failed > 0:
                logger.warning(f"  Failed: {result.failed} slide(s)")
                return 1

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


def _build_detector_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Build detector keyword arguments from CLI args based on cell type."""
    kwargs = {}

    if args.cell_type == "nmj":
        kwargs['model_path'] = getattr(args, 'model_path', None)
        kwargs['confidence_threshold'] = getattr(args, 'confidence_threshold', 0.5)

    elif args.cell_type in ("mk", "cell"):
        kwargs['min_area_um'] = getattr(args, 'min_area_um', 200 if args.cell_type == "mk" else 50)
        kwargs['max_area_um'] = getattr(args, 'max_area_um', 2000 if args.cell_type == "mk" else 500)

    elif args.cell_type == "vessel":
        kwargs['min_vessel_diameter'] = getattr(args, 'min_diameter', 10)
        kwargs['max_vessel_diameter'] = getattr(args, 'max_diameter', 1000)
        kwargs['cd31_channel'] = getattr(args, 'cd31_channel', None)

    return kwargs


def _extract_samples_for_html(
    detections: List[Dict[str, Any]],
    pipeline,
    cell_type: str,
    crop_size: int = 300
) -> List[Dict[str, Any]]:
    """
    Extract image crops for HTML export.

    Args:
        detections: List of detection dicts with global_center
        pipeline: DetectionPipeline with loader
        cell_type: Cell type for normalization
        crop_size: Size of crop in pixels

    Returns:
        List of sample dicts with image_b64
    """
    from segmentation.io.html_export import percentile_normalize, image_to_base64
    import base64
    from PIL import Image
    from io import BytesIO

    logger = get_logger(__name__)
    samples = []

    # Limit samples to avoid huge HTML files
    MAX_SAMPLES = 5000
    if len(detections) > MAX_SAMPLES:
        logger.warning(f"Limiting HTML export to {MAX_SAMPLES} samples (of {len(detections)})")
        # Sample evenly
        indices = np.linspace(0, len(detections) - 1, MAX_SAMPLES, dtype=int)
        detections = [detections[i] for i in indices]

    logger.info(f"Extracting {len(detections)} crops for HTML export...")

    half = crop_size // 2
    mosaic_x, mosaic_y = pipeline.loader.mosaic_origin
    mosaic_w, mosaic_h = pipeline.loader.mosaic_size

    for det in detections:
        try:
            gx, gy = det['global_center']

            # Calculate crop region in global coordinates
            x1 = int(gx - half)
            y1 = int(gy - half)
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            # Clip to mosaic bounds
            x1_clip = max(x1, mosaic_x)
            y1_clip = max(y1, mosaic_y)
            x2_clip = min(x2, mosaic_x + mosaic_w)
            y2_clip = min(y2, mosaic_y + mosaic_h)

            if x2_clip <= x1_clip or y2_clip <= y1_clip:
                continue

            # Get crop from loader
            crop = pipeline.loader.get_tile(
                x1_clip, y1_clip,
                x2_clip - x1_clip,
                channel=pipeline.channel
            )

            if crop is None or crop.size == 0:
                continue

            # Create RGB if grayscale
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)

            # Normalize and convert to uint8
            crop = percentile_normalize(crop)

            # Convert to base64
            pil_img = Image.fromarray(crop)
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Build sample dict
            features = det.get('features', {})
            sample = {
                'uid': det['uid'],
                'image_b64': img_b64,
                'area_um2': features.get('area_um2', features.get('area', 0) * pipeline.pixel_size ** 2),
            }

            # Add cell-type specific fields
            if cell_type == "nmj":
                sample['elongation'] = features.get('elongation', 0)
                sample['confidence'] = features.get('confidence', 1.0)
            elif cell_type == "vessel":
                sample['outer_diameter_um'] = features.get('outer_diameter_um', 0)
                sample['wall_thickness_um'] = features.get('wall_thickness_mean_um', 0)

            samples.append(sample)

        except Exception as e:
            # Skip samples that fail to extract
            continue

    logger.info(f"Extracted {len(samples)} samples for HTML")
    return samples


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    logger = get_logger(__name__)

    if args.format == "html":
        from segmentation.io.html_export import export_samples_to_html
        import json

        # Load results
        with open(args.results) as f:
            data = json.load(f)

        detections = data.get("detections", data.get("nmjs", []))
        cell_type = data.get("cell_type", "unknown")
        experiment_name = data.get("experiment_name", data.get("slide_name", ""))

        # Filter to samples with images
        samples = [d for d in detections if d.get("image_b64")]

        if not samples:
            logger.warning("No samples with images found. Need to extract from CZI.")
            if not args.czi_path:
                logger.error("--czi-path required to extract images")
                return 1
            # TODO: Extract images from CZI
            logger.error("Image extraction not yet implemented in CLI export")
            return 1

        output_dir = args.output_dir or args.results.parent
        html_dir = export_samples_to_html(
            samples=samples,
            output_dir=output_dir,
            cell_type=cell_type,
            experiment_name=experiment_name,
            samples_per_page=args.samples_per_page,
        )
        logger.info(f"HTML exported to: {html_dir}")

    elif args.format == "csv":
        import json

        with open(args.results) as f:
            data = json.load(f)

        detections = data.get("detections", data.get("nmjs", []))
        pixel_size = data.get("pixel_size_um", 0.22)

        output = args.output or args.results.with_suffix(".csv")

        with open(output, "w") as f:
            f.write("uid,global_x_px,global_y_px,global_x_um,global_y_um\n")
            for d in detections:
                uid = d.get("uid", d.get("id", ""))
                center = d.get("global_center", d.get("global_centroid", [0, 0]))
                gx, gy = center[0], center[1]
                f.write(f"{uid},{gx:.1f},{gy:.1f},{gx*pixel_size:.2f},{gy*pixel_size:.2f}\n")

        logger.info(f"CSV exported to: {output}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    logger = get_logger(__name__)

    errors = 0
    for file_path in args.files:
        try:
            result = infer_and_validate(file_path, raise_on_error=True)
            logger.info(f"✓ {file_path}: Valid")
        except Exception as e:
            logger.error(f"✗ {file_path}: {e}")
            errors += 1
            if args.strict:
                return 1

    if errors:
        logger.error(f"{errors} file(s) failed validation")
        return 1

    logger.info(f"All {len(args.files)} file(s) valid")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    logger = get_logger(__name__)
    import json

    path = args.path

    if path.is_file():
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)

            print(f"File: {path}")
            print(f"Type: JSON")

            if "detections" in data or "nmjs" in data:
                dets = data.get("detections", data.get("nmjs", []))
                print(f"Detection count: {len(dets)}")
                print(f"Slide: {data.get('slide_name', 'unknown')}")
                print(f"Cell type: {data.get('cell_type', 'unknown')}")

            elif "annotations" in data or "positive" in data:
                ann = data.get("annotations", {})
                pos = data.get("positive", [])
                neg = data.get("negative", [])
                total = len(ann) + len(pos) + len(neg)
                print(f"Annotation count: {total}")

        elif path.suffix == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(str(path))
            bbox = czi.get_mosaic_bounding_box()
            print(f"File: {path}")
            print(f"Type: CZI")
            print(f"Dimensions: {bbox.w} x {bbox.h} px")
            print(f"Size: {path.stat().st_size / 1e9:.2f} GB")

    elif path.is_dir():
        # Count files
        czi_count = len(list(path.glob("*.czi")))
        json_count = len(list(path.glob("**/*.json")))
        html_count = len(list(path.glob("**/*.html")))

        print(f"Directory: {path}")
        print(f"CZI files: {czi_count}")
        print(f"JSON files: {json_count}")
        print(f"HTML files: {html_count}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level=level, log_file=args.log_file)

    logger = get_logger(__name__)

    # Dispatch to command handler
    if args.command == "run":
        if not args.cell_type:
            parser.parse_args(["run", "--help"])
            return 1
        return cmd_run(args)

    elif args.command == "export":
        if not args.format:
            parser.parse_args(["export", "--help"])
            return 1
        return cmd_export(args)

    elif args.command == "validate":
        return cmd_validate(args)

    elif args.command == "info":
        return cmd_info(args)

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
