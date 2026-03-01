"""
Unified detection pipeline for cell segmentation.

Provides a common framework for:
- Loading CZI files (on-demand or RAM-cached)
- Processing tiles with configurable detectors
- Saving results in consistent formats
- Generating HTML annotation interfaces

Usage:
    from shared.detection_pipeline import DetectionPipeline

    pipeline = DetectionPipeline(
        czi_path='/path/to/slide.czi',
        cell_type='nmj',
        output_dir='/path/to/output',
        load_to_ram=True
    )

    # Process with custom detector
    detections = pipeline.process_tiles(
        detector_fn=my_detector,
        tiles=tile_list,
        channel=1
    )

    # Export results
    pipeline.export_results(detections)
    pipeline.export_html(detections)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple, Generator
from tqdm import tqdm
from datetime import datetime

# NOTE: CZILoader import moved to __init__ to avoid circular import
# (czi_loader.py -> processing.memory -> processing/__init__.py -> pipeline.py -> czi_loader.py)
from segmentation.utils.config import (
    load_config,
    save_config,
    create_run_config,
    get_output_dir,
)
from segmentation.processing.coordinates import (
    tile_to_global_coords,
    generate_uid,
)
from segmentation.io.html_export import export_samples_to_html
from segmentation.utils.json_utils import NumpyEncoder


class DetectionPipeline:
    """
    Unified pipeline for cell detection across different cell types.

    Handles:
    - CZI loading (with optional RAM caching)
    - Tile iteration
    - Result aggregation
    - JSON/CSV export
    - HTML annotation export
    """

    def __init__(
        self,
        czi_path: str | Path,
        cell_type: str,
        output_dir: Optional[str | Path] = None,
        experiment_name: Optional[str] = None,
        channel: int = 0,
        tile_size: int = 3000,
        load_to_ram: bool = False,
        config: Optional[Dict[str, Any]] = None,
        quiet: bool = False
    ):
        """
        Initialize detection pipeline.

        Args:
            czi_path: Path to CZI file
            cell_type: Type of cell (mk, cell, nmj, vessel)
            output_dir: Output directory (default: auto based on cell_type)
            experiment_name: Name for this experiment (default: slide name)
            channel: Channel to process
            tile_size: Size of tiles for processing
            load_to_ram: Load channel into RAM for faster processing
            config: Optional config dict (loads from output_dir if None)
            quiet: Suppress progress output
        """
        self.czi_path = Path(czi_path)
        self.cell_type = cell_type
        self.channel = channel
        self.tile_size = tile_size
        self.quiet = quiet

        # Setup output directory
        if output_dir is None:
            output_dir = get_output_dir(cell_type) / self.czi_path.stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment name for localStorage isolation
        self.experiment_name = experiment_name or self.czi_path.stem

        # Load or create config
        if config is not None:
            self.config = config
        else:
            self.config = load_config(self.output_dir, cell_type)

        # Initialize CZI loader (local import to avoid circular import)
        from segmentation.io.czi_loader import CZILoader
        self.loader = CZILoader(
            czi_path,
            load_to_ram=load_to_ram,
            channel=channel if load_to_ram else None,
            quiet=quiet
        )

        # Get pixel size
        self.pixel_size = self.loader.get_pixel_size()
        if self.pixel_size is None:
            raise ValueError(
                f"CZI file has no pixel size metadata. "
                f"Provide --pixel-size-um on the command line."
            )

        # Storage for results
        self.detections: List[Dict[str, Any]] = []
        self.processing_stats: Dict[str, Any] = {}

    def generate_tile_grid(
        self,
        overlap: int = 0
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Generate tile coordinates covering the entire mosaic.

        Args:
            overlap: Overlap between tiles in pixels

        Yields:
            (tile_x, tile_y) tuples
        """
        x_start, y_start = self.loader.mosaic_origin
        width, height = self.loader.mosaic_size
        step = self.tile_size - overlap

        for y in range(y_start, y_start + height, step):
            for x in range(x_start, x_start + width, step):
                yield (x, y)

    def process_tiles(
        self,
        detector_fn: Callable[[np.ndarray, Dict[str, Any]], List[Dict[str, Any]]],
        tiles: Optional[List[Tuple[int, int]]] = None,
        sample_fraction: float = 1.0,
        min_detections_per_tile: int = 0,
        **detector_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process tiles with a detector function.

        Args:
            detector_fn: Function that takes (tile_data, kwargs) and returns list of detections
            tiles: List of (tile_x, tile_y) to process (default: all tiles)
            sample_fraction: Fraction of tiles to sample (default: 1.0 = all)
            min_detections_per_tile: Skip tiles with fewer detections
            **detector_kwargs: Additional arguments passed to detector_fn

        Returns:
            List of detection dicts with global coordinates
        """
        if tiles is None:
            tiles = list(self.generate_tile_grid())

        # Sample tiles if requested
        if sample_fraction < 1.0:
            n_sample = max(1, int(len(tiles) * sample_fraction))
            indices = np.random.choice(len(tiles), n_sample, replace=False)
            tiles = [tiles[i] for i in sorted(indices)]

        if not self.quiet:
            print(f"Processing {len(tiles)} tiles...")

        self.detections = []
        tiles_processed = 0
        tiles_with_detections = 0

        iterator = tiles
        if not self.quiet:
            iterator = tqdm(tiles, desc="Processing")

        for tile_x, tile_y in iterator:
            # Get tile data
            tile_data = self.loader.get_tile(
                tile_x, tile_y, self.tile_size,
                channel=self.channel
            )

            if tile_data is None:
                continue

            tiles_processed += 1

            # Run detector
            try:
                tile_detections = detector_fn(tile_data, **detector_kwargs)
            except Exception as e:
                if not self.quiet:
                    print(f"  WARNING: Detector failed on tile ({tile_x}, {tile_y}): {e}")
                continue

            if len(tile_detections) < min_detections_per_tile:
                continue

            tiles_with_detections += 1

            # Convert to global coordinates and add UIDs
            for det in tile_detections:
                # Get local centroid (should be [x, y])
                local_x, local_y = det.get('centroid', det.get('local_centroid', [0, 0]))

                # Convert to global
                global_x, global_y = tile_to_global_coords(local_x, local_y, tile_x, tile_y)

                # Generate UID
                uid = generate_uid(
                    self.loader.slide_name,
                    self.cell_type,
                    global_x,
                    global_y
                )

                # Build detection record
                detection = {
                    'uid': uid,
                    'tile_origin': [tile_x, tile_y],
                    'local_centroid': [local_x, local_y],
                    'global_center': [global_x, global_y],
                    'features': det.get('features', {}),
                    **{k: v for k, v in det.items()
                       if k not in ('centroid', 'local_centroid', 'features')}
                }

                self.detections.append(detection)

        # Store stats
        self.processing_stats = {
            'tiles_total': len(tiles),
            'tiles_processed': tiles_processed,
            'tiles_with_detections': tiles_with_detections,
            'total_detections': len(self.detections),
            'pixel_size_um': self.pixel_size,
            'timestamp': datetime.now().isoformat(),
        }

        if not self.quiet:
            print(f"Found {len(self.detections)} detections in {tiles_with_detections} tiles")

        return self.detections

    def export_results(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export detections to JSON file.

        Args:
            detections: List of detections (default: self.detections)
            filename: Output filename (default: {cell_type}_detections.json)

        Returns:
            Path to saved file
        """
        if detections is None:
            detections = self.detections

        if filename is None:
            filename = f"{self.cell_type}_detections.json"

        output_path = self.output_dir / filename

        data = {
            'slide_name': self.loader.slide_name,
            'cell_type': self.cell_type,
            'experiment_name': self.experiment_name,
            **self.processing_stats,
            'detections': detections,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)

        if not self.quiet:
            print(f"Results saved to: {output_path}")

        return output_path

    def export_csv(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export detection coordinates to CSV.

        Args:
            detections: List of detections (default: self.detections)
            filename: Output filename (default: {cell_type}_coordinates.csv)

        Returns:
            Path to saved file
        """
        if detections is None:
            detections = self.detections

        if filename is None:
            filename = f"{self.cell_type}_coordinates.csv"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            # Header
            f.write("uid,global_x_px,global_y_px,global_x_um,global_y_um\n")

            for det in detections:
                uid = det['uid']
                gx, gy = det['global_center']
                gx_um = gx * self.pixel_size
                gy_um = gy * self.pixel_size
                f.write(f"{uid},{gx:.1f},{gy:.1f},{gx_um:.2f},{gy_um:.2f}\n")

        if not self.quiet:
            print(f"Coordinates saved to: {output_path}")

        return output_path

    def export_html(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        samples_per_page: int = 300
    ) -> Path:
        """
        Export HTML annotation interface.

        Args:
            detections: List of detections (must have 'image_b64' for display)
            samples: Pre-formatted samples with image_b64, area_um2, etc.
            samples_per_page: Number of samples per HTML page

        Returns:
            Path to HTML directory
        """
        if samples is None and detections is not None:
            # Try to use detections directly if they have image data
            samples = []
            for det in detections:
                if 'image_b64' in det:
                    samples.append({
                        'uid': det['uid'],
                        'image_b64': det['image_b64'],
                        'area_um2': det.get('features', {}).get('area', 0) * self.pixel_size ** 2,
                        'elongation': det.get('features', {}).get('elongation', 0),
                    })

        if not samples:
            if not self.quiet:
                print("No samples with images to export")
            return self.output_dir / "html"

        html_dir = export_samples_to_html(
            samples=samples,
            output_dir=self.output_dir,
            cell_type=self.cell_type,
            experiment_name=self.experiment_name,
            samples_per_page=samples_per_page,
        )

        if not self.quiet:
            print(f"HTML exported to: {html_dir}")

        return html_dir

    def save_config(self) -> Path:
        """Save current configuration to output directory."""
        config = create_run_config(
            experiment_name=self.experiment_name,
            cell_type=self.cell_type,
            slide_name=self.loader.slide_name,
            channel=self.channel,
            pixel_size_um=self.pixel_size,
            tile_size=self.tile_size,
            **self.processing_stats
        )
        return save_config(self.output_dir, config)

    def close(self):
        """Release resources."""
        self.loader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_simple_detector(
    threshold_fn: Callable[[np.ndarray], np.ndarray],
    min_area: int = 100,
    max_area: int = 10000
) -> Callable:
    """
    Create a simple detector function from a threshold function.

    Args:
        threshold_fn: Function that takes image and returns binary mask
        min_area: Minimum detection area in pixels
        max_area: Maximum detection area in pixels

    Returns:
        Detector function compatible with DetectionPipeline.process_tiles()
    """
    from skimage.measure import label, regionprops

    def detector(tile_data: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        # Apply threshold
        mask = threshold_fn(tile_data)

        # Label connected components
        labeled = label(mask)
        props = regionprops(labeled, intensity_image=tile_data)

        detections = []
        for prop in props:
            if prop.area < min_area or prop.area > max_area:
                continue

            detections.append({
                'centroid': [float(prop.centroid[1]), float(prop.centroid[0])],  # [x, y]
                'features': {
                    'area': int(prop.area),
                    'eccentricity': float(prop.eccentricity),
                    'solidity': float(prop.solidity),
                    'mean_intensity': float(prop.mean_intensity),
                }
            })

        return detections

    return detector
