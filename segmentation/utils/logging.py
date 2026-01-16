"""
Unified logging configuration for the segmentation pipeline.

Usage:
    from shared.logging_config import get_logger, setup_logging

    # Get a logger for your module
    logger = get_logger(__name__)

    # Setup logging at application start
    setup_logging(level="INFO", log_file="/path/to/output/run.log")

    # Use it
    logger.info("Processing started")
    logger.debug("Tile coordinates: %s", tile_coords)
    logger.warning("No detections in tile (%d, %d)", x, y)
    logger.error("Failed to load file: %s", path)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# Custom formatter with colors for terminal output
class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}
_initialized = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    return _loggers[name]


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True,
    colored: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Explicit path to log file
        log_dir: Directory for auto-named log file (uses timestamp)
        console: Whether to output to console
        colored: Whether to use colored output in console
        format_string: Custom format string

    Returns:
        Root logger instance
    """
    global _initialized

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if re-initializing
    if _initialized:
        root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if colored and sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter(format_string))
        else:
            console_handler.setFormatter(logging.Formatter(format_string))

        root_logger.addHandler(console_handler)

    # File handler
    if log_file or log_dir:
        if log_file:
            log_path = Path(log_file)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"segmentation_{timestamp}.log"

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_path}")

    _initialized = True
    return root_logger


def log_parameters(logger: logging.Logger, params: dict, title: str = "Parameters") -> None:
    """
    Log a dictionary of parameters in a formatted way.

    Args:
        logger: Logger instance
        params: Dictionary of parameters to log
        title: Title for the parameter block
    """
    logger.info(f"{'='*50}")
    logger.info(f"{title}")
    logger.info(f"{'='*50}")

    for key, value in params.items():
        if isinstance(value, (list, tuple)) and len(value) > 5:
            logger.info(f"  {key}: [{len(value)} items]")
        elif isinstance(value, dict) and len(value) > 5:
            logger.info(f"  {key}: {{{len(value)} keys}}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"{'='*50}")


def log_processing_start(
    logger: logging.Logger,
    operation: str,
    **kwargs
) -> None:
    """Log the start of a processing operation with parameters."""
    logger.info(f"Starting: {operation}")
    if kwargs:
        for key, value in kwargs.items():
            logger.info(f"  {key}: {value}")


def log_processing_end(
    logger: logging.Logger,
    operation: str,
    duration_seconds: Optional[float] = None,
    **results
) -> None:
    """Log the end of a processing operation with results."""
    if duration_seconds is not None:
        if duration_seconds >= 3600:
            duration_str = f"{duration_seconds/3600:.1f} hours"
        elif duration_seconds >= 60:
            duration_str = f"{duration_seconds/60:.1f} minutes"
        else:
            duration_str = f"{duration_seconds:.1f} seconds"
        logger.info(f"Completed: {operation} in {duration_str}")
    else:
        logger.info(f"Completed: {operation}")

    if results:
        for key, value in results.items():
            logger.info(f"  {key}: {value}")


class ProcessingTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(f"Failed: {self.operation} after {duration:.1f}s - {exc_val}")
        else:
            log_processing_end(self.logger, self.operation, duration)
        return False
