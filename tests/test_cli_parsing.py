"""Tests for xldvp_seg.pipeline.cli.build_parser argument parsing.

Covers:
- Basic arg parsing (--czi-path, --cell-type)
- --marker-snr-channels
- --no-contour-processing
- --sample-fraction default
- --html-sample-fraction
"""

import pytest

from xldvp_seg.pipeline.cli import build_parser


@pytest.fixture
def parser():
    return build_parser()


class TestCliParsing:
    def test_basic_args(self, parser):
        """Parse basic required-ish args."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.czi_path == "test.czi"
        assert args.cell_type == "cell"

    def test_cell_type_choices(self, parser):
        """All valid cell types should parse without error."""
        for ct in ["nmj", "mk", "cell", "vessel", "mesothelium", "islet", "tissue_pattern"]:
            args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", ct])
            assert args.cell_type == ct

    def test_invalid_cell_type(self, parser):
        """Invalid cell type should cause parse error."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--czi-path", "test.czi", "--cell-type", "invalid"])

    def test_marker_snr_channels(self, parser):
        """--marker-snr-channels should store the string value."""
        args = parser.parse_args(
            [
                "--czi-path",
                "test.czi",
                "--cell-type",
                "cell",
                "--marker-snr-channels",
                "SMA:1,CD31:3",
            ]
        )
        assert args.marker_snr_channels == "SMA:1,CD31:3"

    def test_marker_snr_channels_default(self, parser):
        """--marker-snr-channels default should be None."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.marker_snr_channels is None

    def test_no_contour_processing(self, parser):
        """--no-contour-processing should set contour_processing to False."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--no-contour-processing"]
        )
        assert args.contour_processing is False

    def test_contour_processing_default_true(self, parser):
        """contour_processing default should be True."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.contour_processing is True

    def test_sample_fraction_default(self, parser):
        """--sample-fraction default should be 1.0."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.sample_fraction == 1.0

    def test_html_sample_fraction(self, parser):
        """--html-sample-fraction should parse a float."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--html-sample-fraction", "0.10"]
        )
        assert args.html_sample_fraction == pytest.approx(0.10)

    def test_html_sample_fraction_default(self, parser):
        """--html-sample-fraction default should be 0 (no subsampling)."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.html_sample_fraction == 0

    def test_no_background_correction(self, parser):
        """--no-background-correction should set background_correction to False."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--no-background-correction"]
        )
        assert args.background_correction is False

    def test_background_correction_default_true(self, parser):
        """background_correction default should be True."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.background_correction is True

    def test_all_channels_flag(self, parser):
        """--all-channels should set all_channels to True."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--all-channels"]
        )
        assert args.all_channels is True

    def test_num_gpus(self, parser):
        """--num-gpus should parse an integer."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--num-gpus", "4"]
        )
        assert args.num_gpus == 4

    def test_verbose_flag(self, parser):
        """--verbose should set verbose to True."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell", "--verbose"])
        assert args.verbose is True

    def test_channel_spec(self, parser):
        """--channel-spec should store the string."""
        args = parser.parse_args(
            ["--czi-path", "test.czi", "--cell-type", "cell", "--channel-spec", "cyto=PM,nuc=488"]
        )
        assert args.channel_spec == "cyto=PM,nuc=488"

    def test_scene_default(self, parser):
        """--scene default should be 0."""
        args = parser.parse_args(["--czi-path", "test.czi", "--cell-type", "cell"])
        assert args.scene == 0
