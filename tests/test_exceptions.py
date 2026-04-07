"""Tests for the xldvp_seg exception hierarchy."""

import pytest

from xldvp_seg.exceptions import (
    ChannelResolutionError,
    ClassificationError,
    ConfigError,
    DataLoadError,
    DetectionError,
    ExportError,
    XldvpSegError,
)


class TestExceptionHierarchy:
    """All 7 exception classes are instantiable and follow dual-inheritance."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            XldvpSegError,
            ConfigError,
            ChannelResolutionError,
            DataLoadError,
            DetectionError,
            ClassificationError,
            ExportError,
        ],
    )
    def test_instantiable_with_message(self, exc_cls):
        e = exc_cls("test message")
        assert str(e) == "test message"

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigError,
            ChannelResolutionError,
            DataLoadError,
            DetectionError,
            ClassificationError,
            ExportError,
        ],
    )
    def test_all_inherit_from_base(self, exc_cls):
        assert isinstance(exc_cls("x"), XldvpSegError)

    def test_config_error_is_value_error(self):
        assert isinstance(ConfigError("x"), ValueError)

    def test_channel_resolution_error_is_value_error(self):
        assert isinstance(ChannelResolutionError("x"), ValueError)

    def test_data_load_error_is_io_error(self):
        assert isinstance(DataLoadError("x"), IOError)

    def test_detection_error_is_runtime_error(self):
        assert isinstance(DetectionError("x"), RuntimeError)

    def test_classification_error_is_runtime_error(self):
        assert isinstance(ClassificationError("x"), RuntimeError)

    def test_export_error_is_runtime_error(self):
        assert isinstance(ExportError("x"), RuntimeError)

    def test_except_value_error_catches_config_error(self):
        with pytest.raises(ValueError):
            raise ConfigError("bad config")

    def test_except_runtime_error_catches_detection_error(self):
        with pytest.raises(RuntimeError):
            raise DetectionError("detection failed")
