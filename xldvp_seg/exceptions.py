"""Custom exception hierarchy for xldvp_seg.

All exceptions inherit from :class:`XldvpSegError` AND from the closest
standard-library exception (ValueError, RuntimeError, IOError).  This
dual-inheritance ensures that existing ``except ValueError`` handlers
still catch the new types -- no downstream breakage.
"""


class XldvpSegError(Exception):
    """Base exception for all xldvp_seg errors."""


# -- Configuration -----------------------------------------------------------


class ConfigError(XldvpSegError, ValueError):
    """Invalid pipeline configuration."""


class ChannelResolutionError(XldvpSegError, ValueError):
    """Could not resolve a channel specification to a CZI channel index."""


# -- Data I/O ----------------------------------------------------------------


class DataLoadError(XldvpSegError, IOError):
    """Failed to load input data (JSON, HDF5, CZI, classifier)."""


# -- Pipeline runtime --------------------------------------------------------


class DetectionError(XldvpSegError, RuntimeError):
    """Error during the detection pipeline."""


class ClassificationError(XldvpSegError, RuntimeError):
    """Error in classifier train / predict / load."""


class ExportError(XldvpSegError, RuntimeError):
    """Error during data export (HTML, LMD XML, SpatialData)."""
