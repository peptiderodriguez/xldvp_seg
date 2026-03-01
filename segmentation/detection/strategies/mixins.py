"""
Mixin classes for detection strategies.

These mixins provide reusable functionality that can be composed into
different detection strategies without code duplication.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import skew, kurtosis


class MultiChannelFeatureMixin:
    """
    Mixin for extracting per-channel features from multi-channel images.

    This mixin provides methods to extract intensity statistics and
    inter-channel ratios from multi-channel microscopy images. It is
    designed to be independent of specific channel naming conventions,
    accepting a dictionary mapping channel names to their image data.

    Usage:
        class MyStrategy(DetectionStrategy, MultiChannelFeatureMixin):
            def detect(self, tile, models, pixel_size_um, channels=None):
                masks = self.segment(tile, models)
                for mask in masks:
                    features = self.extract_multichannel_features(
                        mask,
                        {'nuclear': ch0_data, 'btx': ch1_data, 'nfl': ch2_data}
                    )

    Channel naming convention (for NMJ analysis):
        - 'nuclear' or 'ch0': Nuclear stain (488nm) - should be LOW in real NMJs
        - 'btx' or 'ch1': Bungarotoxin (647nm) - should be HIGH in real NMJs
        - 'nfl' or 'ch2': Neurofilament (750nm)

    The mixin is designed to work with arbitrary channel names, so it can
    be used for other multi-channel applications beyond NMJ detection.
    """

    def extract_channel_stats(
        self,
        mask: np.ndarray,
        channel_data: np.ndarray,
        channel_name: str,
        _include_zeros: bool = False
    ) -> Dict[str, float]:
        """
        Extract intensity statistics for a single channel within a mask region.

        Computes comprehensive statistics including basic intensity measures,
        percentiles for robust estimation, and distribution features like
        skewness and kurtosis.

        Args:
            mask: Binary mask defining the region of interest (HxW boolean)
            channel_data: 2D array of intensity values for this channel (HxW)
            channel_name: Name prefix for the output features (e.g., 'ch0', 'btx')
            _include_zeros: If True, include zero-intensity pixels in statistics.
                If False (default), exclude zeros to avoid bias from CZI zero-padding
                at tile boundaries. This is a backward-compat parameter; set True
                when you know the tile has no zero-padding.

        Returns:
            Dictionary of features with keys prefixed by channel_name:
                - {channel_name}_mean: Mean intensity
                - {channel_name}_std: Standard deviation
                - {channel_name}_max: Maximum intensity
                - {channel_name}_min: Minimum intensity
                - {channel_name}_median: Median intensity
                - {channel_name}_p5: 5th percentile
                - {channel_name}_p25: 25th percentile (Q1)
                - {channel_name}_p75: 75th percentile (Q3)
                - {channel_name}_p95: 95th percentile
                - {channel_name}_variance: Variance
                - {channel_name}_skewness: Distribution skewness
                - {channel_name}_kurtosis: Distribution kurtosis
                - {channel_name}_iqr: Interquartile range (Q3-Q1)
                - {channel_name}_dynamic_range: Max - Min
                - {channel_name}_cv: Coefficient of variation (std/mean)

            Returns empty dict if mask is empty or shapes don't match.
        """
        if mask.sum() == 0:
            return {}

        # Validate shape match
        if channel_data.shape != mask.shape:
            return {}

        # Get masked pixels (spatial mask only)
        masked_pixels = channel_data[mask].astype(np.float32)
        if not _include_zeros:
            # Exclude zero-intensity pixels to avoid bias from CZI zero-padding
            # at tile boundaries. Legitimate zero-intensity pixels inside tissue
            # are rare and excluding them has negligible effect on statistics.
            masked_pixels = masked_pixels[masked_pixels > 0]
        if len(masked_pixels) == 0:
            return {f'{channel_name}_{stat}': 0.0 for stat in
                    ['mean', 'std', 'max', 'min', 'median', 'p5', 'p25', 'p75', 'p95',
                     'variance', 'skewness', 'kurtosis', 'iqr', 'dynamic_range', 'cv']}

        features = {}
        prefix = channel_name

        # Basic intensity statistics
        features[f'{prefix}_mean'] = float(np.mean(masked_pixels))
        features[f'{prefix}_std'] = float(np.std(masked_pixels))
        features[f'{prefix}_max'] = float(np.max(masked_pixels))
        features[f'{prefix}_min'] = float(np.min(masked_pixels))
        features[f'{prefix}_median'] = float(np.median(masked_pixels))

        # Percentiles for robust statistics
        features[f'{prefix}_p5'] = float(np.percentile(masked_pixels, 5))
        features[f'{prefix}_p25'] = float(np.percentile(masked_pixels, 25))
        features[f'{prefix}_p75'] = float(np.percentile(masked_pixels, 75))
        features[f'{prefix}_p95'] = float(np.percentile(masked_pixels, 95))

        # Distribution features
        features[f'{prefix}_variance'] = float(np.var(masked_pixels))
        features[f'{prefix}_skewness'] = float(self._safe_skewness(masked_pixels))
        features[f'{prefix}_kurtosis'] = float(self._safe_kurtosis(masked_pixels))
        features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
        features[f'{prefix}_dynamic_range'] = features[f'{prefix}_max'] - features[f'{prefix}_min']

        # Coefficient of variation
        if features[f'{prefix}_mean'] > 0:
            features[f'{prefix}_cv'] = features[f'{prefix}_std'] / features[f'{prefix}_mean']
        else:
            features[f'{prefix}_cv'] = 0.0

        return features

    def extract_multichannel_features(
        self,
        mask: np.ndarray,
        channels_dict: Dict[str, np.ndarray],
        primary_channel: Optional[str] = None,
        compute_ratios: bool = True
    ) -> Dict[str, float]:
        """
        Extract features from all channels and compute inter-channel ratios.

        This method processes multiple channels and optionally computes
        diagnostic ratios between them. The ratios are particularly useful
        for distinguishing true biological signals from autofluorescence.

        Args:
            mask: Binary mask defining the region of interest (HxW boolean)
            channels_dict: Dictionary mapping channel names to 2D arrays.
                Keys can be arbitrary strings (e.g., 'nuclear', 'btx', 'nfl')
                or numeric indices as strings ('ch0', 'ch1', 'ch2').
                Values should be HxW arrays matching the mask shape.
            primary_channel: Optional name of the primary signal channel.
                Used for computing channel specificity metrics.
                If None, no specificity metrics are computed.
            compute_ratios: Whether to compute inter-channel ratios (default True).
                Set to False for speed when only per-channel stats are needed.

        Returns:
            Flat dictionary containing:
            - Per-channel features: {channel}_mean, {channel}_std, etc.
              (15 features per channel)
            - Inter-channel ratios (if compute_ratios=True):
              - {ch_a}_{ch_b}_ratio: ratio of means
              - {ch_a}_{ch_b}_diff: difference of means
            - Channel specificity metrics (if primary_channel specified):
              - channel_specificity: primary / max(others)
              - channel_specificity_diff: primary - max(others)

            Returns empty dict if mask is empty.

        Example:
            # For 3-channel imaging with generic channel names
            features = self.extract_multichannel_features(
                mask,
                {
                    'ch0': ch0_data,
                    'ch1': ch1_data,
                    'ch2': ch2_data
                },
                primary_channel='ch1'
            )

            # Access features
            ch1_mean = features['ch1_mean']
            ch0_ch1_ratio = features['ch0_ch1_ratio']
        """
        if mask.sum() == 0:
            return {}

        features = {}
        channel_means = {}

        # Extract per-channel statistics
        for channel_name, channel_data in sorted(channels_dict.items()):
            if channel_data is None:
                continue

            # Extract stats for this channel
            channel_stats = self.extract_channel_stats(mask, channel_data, channel_name)
            features.update(channel_stats)

            # Store mean for ratio calculations
            if f'{channel_name}_mean' in channel_stats:
                channel_means[channel_name] = channel_stats[f'{channel_name}_mean']

        # Compute inter-channel ratios if requested
        if compute_ratios and len(channel_means) >= 2:
            ratio_features = self._compute_channel_ratios(
                channel_means, primary_channel
            )
            features.update(ratio_features)

        return features

    def _compute_channel_ratios(
        self,
        channel_means: Dict[str, float],
        primary_channel: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute inter-channel ratios from mean intensities.

        These ratios are critical for distinguishing biological signals from
        autofluorescence. Real biological signals (e.g., NMJs stained with BTX)
        should have high intensity in the target channel but low intensity in
        non-specific channels like nuclear stain.

        Args:
            channel_means: Dictionary mapping channel names to mean intensities
            primary_channel: Name of the primary signal channel for specificity

        Returns:
            Dictionary of ratio features
        """
        features = {}
        channel_names = sorted(channel_means.keys())

        # Compute pairwise ratios
        for i, ch_a in enumerate(channel_names):
            for ch_b in channel_names[i+1:]:
                mean_a = channel_means[ch_a]
                mean_b = channel_means[ch_b]

                # Avoid division by zero
                safe_mean_b = max(mean_b, 1.0)
                safe_mean_a = max(mean_a, 1.0)

                # a/b ratio and difference
                features[f'{ch_a}_{ch_b}_ratio'] = mean_a / safe_mean_b
                features[f'{ch_a}_{ch_b}_diff'] = mean_a - mean_b

                # b/a ratio (reversed)
                features[f'{ch_b}_{ch_a}_ratio'] = mean_b / safe_mean_a

        # Channel specificity: primary vs max of other channels
        if primary_channel is not None and primary_channel in channel_means:
            primary_mean = channel_means[primary_channel]
            other_means = [v for k, v in channel_means.items() if k != primary_channel]

            if other_means:
                max_other = max(other_means)
                features['channel_specificity'] = primary_mean / max(max_other, 1.0)
                features['channel_specificity_diff'] = primary_mean - max_other

        return features

    def _safe_skewness(self, data: np.ndarray) -> float:
        """
        Compute skewness safely, returning 0 if not enough data.

        Args:
            data: 1D array of values

        Returns:
            Skewness value, or 0.0 if computation fails
        """
        if len(data) < 3:
            return 0.0
        if data.max() == data.min():  # constant array
            return 0.0
        try:
            result = float(skew(data))
            return result if np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """
        Compute kurtosis safely, returning 0 if not enough data.

        Args:
            data: 1D array of values

        Returns:
            Kurtosis value, or 0.0 if computation fails
        """
        if len(data) < 4:
            return 0.0
        if data.max() == data.min():  # constant array
            return 0.0
        try:
            result = float(kurtosis(data))
            return result if np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def extract_channel_intensity_simple(
        self,
        mask: np.ndarray,
        channel_data: np.ndarray,
        channel_name: str,
        _include_zeros: bool = False
    ) -> Dict[str, float]:
        """
        Extract basic intensity statistics for a single channel (lightweight version).

        This is a simplified version of extract_channel_stats() that only
        computes mean, std, max, min, and median. Use this when you don't
        need the full statistical profile (percentiles, skewness, etc.)
        for faster processing.

        Args:
            mask: Binary mask defining the region of interest (HxW boolean)
            channel_data: 2D array of intensity values for this channel (HxW)
            channel_name: Name prefix for the output features
            _include_zeros: If True, include zero-intensity pixels. Default False
                excludes zeros (CZI zero-padding at tile boundaries).

        Returns:
            Dictionary with basic intensity features (5 features):
                - {channel_name}_mean
                - {channel_name}_std
                - {channel_name}_max
                - {channel_name}_min
                - {channel_name}_median
        """
        if mask.sum() == 0:
            return {}

        if channel_data.shape != mask.shape:
            return {}

        masked_pixels = channel_data[mask].astype(np.float32)
        if not _include_zeros:
            # Exclude zero pixels (CZI zero-padding at tile boundaries)
            masked_pixels = masked_pixels[masked_pixels > 0]
        if len(masked_pixels) == 0:
            return {}

        return {
            f'{channel_name}_mean': float(np.mean(masked_pixels)),
            f'{channel_name}_std': float(np.std(masked_pixels)),
            f'{channel_name}_max': float(np.max(masked_pixels)),
            f'{channel_name}_min': float(np.min(masked_pixels)),
            f'{channel_name}_median': float(np.median(masked_pixels)),
        }
