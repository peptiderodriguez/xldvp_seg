"""
Batch processing module for handling multiple slides.

Supports:
- Directory with glob pattern
- Text file with paths (one per line)
- Mixed input sources

Usage:
    from shared.batch import BatchProcessor, collect_slides

    # Collect slides from various sources
    slides = collect_slides(
        input_dir="/path/to/slides",
        pattern="*.czi",
        batch_file="slides.txt"
    )

    # Process with BatchProcessor
    processor = BatchProcessor(
        slides=slides,
        cell_type="nmj",
        output_base="/path/to/output"
    )
    processor.run(detector_fn=my_detector)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


from segmentation.utils.logging import get_logger, log_parameters
from segmentation.utils.config import get_output_dir
from segmentation.processing.pipeline import DetectionPipeline

logger = get_logger(__name__)


@dataclass
class SlideInfo:
    """Information about a slide to process."""
    path: Path
    name: str
    output_dir: Optional[Path] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    detections_count: int = 0
    processing_time_seconds: float = 0.0

    @classmethod
    def from_path(cls, path: Union[str, Path], output_base: Optional[Path] = None) -> "SlideInfo":
        """Create SlideInfo from a path."""
        path = Path(path)
        name = path.stem

        output_dir = None
        if output_base:
            output_dir = output_base / name

        return cls(path=path, name=name, output_dir=output_dir)


def collect_slides(
    input_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.czi",
    batch_file: Optional[Union[str, Path]] = None,
    paths: Optional[List[Union[str, Path]]] = None,
    recursive: bool = False,
) -> List[SlideInfo]:
    """
    Collect slides from various input sources.

    Args:
        input_dir: Directory to search for slides
        pattern: Glob pattern for matching files (default: *.czi)
        batch_file: Text file with paths (one per line)
        paths: Explicit list of paths
        recursive: If True, search recursively in input_dir

    Returns:
        List of SlideInfo objects
    """
    collected: Dict[Path, SlideInfo] = {}

    # From explicit paths
    if paths:
        for p in paths:
            path = Path(p).resolve()
            if path.exists() and path not in collected:
                collected[path] = SlideInfo.from_path(path)
                logger.debug(f"Added from paths: {path}")

    # From directory with glob
    if input_dir:
        input_dir = Path(input_dir)
        if input_dir.is_dir():
            glob_method = input_dir.rglob if recursive else input_dir.glob
            for path in glob_method(pattern):
                path = path.resolve()
                if path.is_file() and path not in collected:
                    collected[path] = SlideInfo.from_path(path)
                    logger.debug(f"Added from glob: {path}")

    # From batch file
    if batch_file:
        batch_file = Path(batch_file)
        if batch_file.exists():
            with open(batch_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        path = Path(line).resolve()
                        if path.exists() and path not in collected:
                            collected[path] = SlideInfo.from_path(path)
                            logger.debug(f"Added from batch file: {path}")
                        elif not path.exists():
                            logger.warning(f"File not found (from batch file): {line}")

    slides = list(collected.values())
    logger.info(f"Collected {len(slides)} slides for processing")

    return slides


@dataclass
class BatchResult:
    """Results from batch processing."""
    total_slides: int
    completed: int
    failed: int
    total_detections: int
    total_time_seconds: float
    slides: List[SlideInfo]
    output_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_slides": self.total_slides,
            "completed": self.completed,
            "failed": self.failed,
            "total_detections": self.total_detections,
            "total_time_seconds": self.total_time_seconds,
            "slides": [
                {
                    "name": s.name,
                    "path": str(s.path),
                    "status": s.status,
                    "error": s.error,
                    "detections_count": s.detections_count,
                    "processing_time_seconds": s.processing_time_seconds,
                }
                for s in self.slides
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save batch results to JSON."""
        if path is None:
            path = self.output_dir / "batch_results.json"
        path = Path(path)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

        logger.info(f"Batch results saved to: {path}")
        return path


class BatchProcessor:
    """
    Process multiple slides in batch.

    Supports:
    - Sequential processing (default)
    - Parallel preprocessing with sequential GPU processing
    - Progress tracking and resumption
    """

    def __init__(
        self,
        slides: List[SlideInfo],
        cell_type: str,
        output_base: Optional[Union[str, Path]] = None,
        channel: int = 0,
        tile_size: int = 3000,
        sample_fraction: float = 1.0,
        load_to_ram: bool = False,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize batch processor.

        Args:
            slides: List of SlideInfo objects to process
            cell_type: Type of cell to detect (mk, cell, nmj, vessel)
            output_base: Base output directory (default: auto based on cell_type)
            channel: Channel to process
            tile_size: Tile size for processing
            sample_fraction: Fraction of tiles to sample
            load_to_ram: Load slides into RAM for faster processing
            experiment_name: Name for this batch (default: timestamp)
        """
        self.slides = slides
        self.cell_type = cell_type
        self.channel = channel
        self.tile_size = tile_size
        self.sample_fraction = sample_fraction
        self.load_to_ram = load_to_ram

        # Setup output
        if output_base is None:
            output_base = get_output_dir(cell_type)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Experiment name
        if experiment_name is None:
            experiment_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        # Update slide output dirs
        for slide in self.slides:
            slide.output_dir = self.output_base / slide.name

        logger.info(f"BatchProcessor initialized:")
        logger.info(f"  Slides: {len(self.slides)}")
        logger.info(f"  Cell type: {self.cell_type}")
        logger.info(f"  Output: {self.output_base}")

    def run(
        self,
        detector_fn: Callable,
        export_html: bool = True,
        export_csv: bool = True,
        continue_on_error: bool = True,
        **detector_kwargs
    ) -> BatchResult:
        """
        Run batch processing.

        Args:
            detector_fn: Detection function for DetectionPipeline
            export_html: Export HTML annotation interface
            export_csv: Export CSV coordinates
            continue_on_error: Continue processing other slides on error
            **detector_kwargs: Additional arguments for detector

        Returns:
            BatchResult with processing summary
        """
        import time
        start_time = time.time()

        log_parameters(logger, {
            "slides": len(self.slides),
            "cell_type": self.cell_type,
            "channel": self.channel,
            "tile_size": self.tile_size,
            "sample_fraction": self.sample_fraction,
            "load_to_ram": self.load_to_ram,
            "export_html": export_html,
            "export_csv": export_csv,
        }, title="Batch Processing Parameters")

        total_detections = 0
        completed = 0
        failed = 0

        for i, slide in enumerate(self.slides, 1):
            logger.info(f"Processing slide {i}/{len(self.slides)}: {slide.name}")
            slide.status = "processing"
            slide_start = time.time()

            try:
                with DetectionPipeline(
                    czi_path=slide.path,
                    cell_type=self.cell_type,
                    output_dir=slide.output_dir,
                    experiment_name=self.experiment_name,
                    channel=self.channel,
                    tile_size=self.tile_size,
                    load_to_ram=self.load_to_ram,
                ) as pipeline:
                    # Run detection
                    detections = pipeline.process_tiles(
                        detector_fn=detector_fn,
                        sample_fraction=self.sample_fraction,
                        **detector_kwargs
                    )

                    # Export results
                    pipeline.export_results(detections)

                    if export_csv:
                        pipeline.export_csv(detections)

                    if export_html and detections:
                        pipeline.export_html(detections)

                    # Save config
                    pipeline.save_config()

                    slide.detections_count = len(detections)
                    total_detections += len(detections)

                slide.status = "completed"
                completed += 1
                slide.processing_time_seconds = time.time() - slide_start
                logger.info(f"  Completed: {slide.detections_count} detections in {slide.processing_time_seconds:.1f}s")

            except Exception as e:
                slide.status = "failed"
                slide.error = str(e)
                slide.processing_time_seconds = time.time() - slide_start
                failed += 1
                logger.error(f"  Failed: {e}")

                if not continue_on_error:
                    raise

        total_time = time.time() - start_time

        result = BatchResult(
            total_slides=len(self.slides),
            completed=completed,
            failed=failed,
            total_detections=total_detections,
            total_time_seconds=total_time,
            slides=self.slides,
            output_dir=self.output_base,
        )

        # Save batch results
        result.save()

        logger.info(f"Batch processing complete:")
        logger.info(f"  Completed: {completed}/{len(self.slides)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")

        return result

    def run_parallel_preload(
        self,
        detector_fn: Callable,
        max_slides_in_memory: int = 2,
        **kwargs
    ) -> BatchResult:
        """
        Run batch with parallel preloading.

        Loads next slide while processing current one.

        Args:
            detector_fn: Detection function
            max_slides_in_memory: Max slides to keep in memory
            **kwargs: Additional arguments for run()
        """
        # For now, just run sequentially with load_to_ram
        # Future: implement true parallel preloading
        self.load_to_ram = True
        return self.run(detector_fn, **kwargs)


def create_batch_summary_html(
    result: BatchResult,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create HTML summary of batch processing results.

    Args:
        result: BatchResult from batch processing
        output_path: Where to save HTML (default: output_dir/batch_summary.html)

    Returns:
        Path to created HTML file
    """
    if output_path is None:
        output_path = result.output_dir / "batch_summary.html"
    output_path = Path(output_path)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Batch Processing Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 40px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #00ff88; text-align: center; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-box {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
        .stat-label {{ color: #aaa; margin-top: 5px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ background: #16213e; color: #00ff88; }}
        tr:hover {{ background: #16213e; }}
        .status-completed {{ color: #00ff88; }}
        .status-failed {{ color: #ff4444; }}
        .status-pending {{ color: #ffaa00; }}
        a {{ color: #00aaff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Processing Summary</h1>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{result.completed}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{result.failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{result.total_detections:,}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{result.total_time_seconds/60:.1f}m</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Slide</th>
                    <th>Status</th>
                    <th>Detections</th>
                    <th>Time</th>
                    <th>View</th>
                </tr>
            </thead>
            <tbody>
'''

    for slide in result.slides:
        status_class = f"status-{slide.status}"
        view_link = ""
        if slide.status == "completed" and slide.output_dir:
            html_path = slide.output_dir / "html" / "index.html"
            if html_path.exists():
                view_link = f'<a href="{slide.name}/html/index.html">View &rarr;</a>'

        error_info = f' ({slide.error[:50]}...)' if slide.error else ''

        html += f'''                <tr>
                    <td>{slide.name}</td>
                    <td class="{status_class}">{slide.status}{error_info}</td>
                    <td>{slide.detections_count:,}</td>
                    <td>{slide.processing_time_seconds:.1f}s</td>
                    <td>{view_link}</td>
                </tr>
'''

    html += '''            </tbody>
        </table>
    </div>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Batch summary HTML saved to: {output_path}")
    return output_path
