"""Tests for CZI channel resolution logic."""

import pytest

from xldvp_seg.io.czi_loader import (
    ChannelResolutionError,
    parse_markers_from_filename,
    resolve_channel_indices,
)


def _mock_metadata(*channel_defs):
    """Build mock CZI metadata.

    Each channel_def is (index, name, excitation_nm) or (index, name, excitation_nm, emission_nm).
    """
    channels = []
    for cd in channel_defs:
        ch = {"index": cd[0], "name": cd[1], "excitation_nm": cd[2]}
        if len(cd) > 3:
            ch["emission_nm"] = cd[3]
        channels.append(ch)
    return {"channels": channels, "n_channels": len(channels)}


# Standard 4-channel metadata for most tests
FOUR_CH = _mock_metadata(
    (0, "AF488", 493, 517),
    (1, "AF647", 653, 668),
    (2, "AF750", 752, 779),
    (3, "AF555", 553, 568),
)


class TestResolveChannelIndices:
    def test_integer_index(self):
        result = resolve_channel_indices(FOUR_CH, ["2"])
        assert result == {"2": 2}

    def test_integer_index_zero(self):
        result = resolve_channel_indices(FOUR_CH, ["0"])
        assert result == {"0": 0}

    def test_integer_index_out_of_range(self):
        with pytest.raises(ChannelResolutionError, match="Cannot resolve"):
            resolve_channel_indices(FOUR_CH, ["10"])

    def test_wavelength_exact(self):
        result = resolve_channel_indices(FOUR_CH, ["653"])
        assert result["653"] == 1  # AF647 has excitation 653nm

    def test_wavelength_fuzzy_within_10nm(self):
        result = resolve_channel_indices(FOUR_CH, ["647"])
        assert result["647"] == 1  # 647 is within ±10nm of 653

    def test_wavelength_no_match(self):
        with pytest.raises(ChannelResolutionError, match="Cannot resolve"):
            resolve_channel_indices(FOUR_CH, ["800"])

    def test_wavelength_555(self):
        result = resolve_channel_indices(FOUR_CH, ["555"])
        assert result["555"] == 3  # AF555 has excitation 553nm

    def test_multiple_specs(self):
        result = resolve_channel_indices(FOUR_CH, ["647", "555"])
        assert result["647"] == 1
        assert result["555"] == 3

    def test_marker_name_from_filename(self):
        result = resolve_channel_indices(FOUR_CH, ["SMA"], filename="SMA647_CD31555.czi")
        assert result["SMA"] == 1  # SMA -> 647nm from filename -> ch1

    def test_marker_name_cd31_from_filename(self):
        result = resolve_channel_indices(FOUR_CH, ["CD31"], filename="SMA647_CD31555.czi")
        assert result["CD31"] == 3  # CD31 -> 555nm from filename -> ch3

    def test_metadata_name_substring_match(self):
        meta = _mock_metadata(
            (0, "Hoechst 33258", 405, 461),
            (1, "AF647", 653, 668),
        )
        result = resolve_channel_indices(meta, ["Hoechst"])
        assert result["Hoechst"] == 0

    def test_metadata_name_exact_match(self):
        meta = _mock_metadata(
            (0, "DAPI", 405, 461),
            (1, "AF647", 653, 668),
        )
        result = resolve_channel_indices(meta, ["DAPI"])
        assert result["DAPI"] == 0

    def test_unresolvable_name_raises(self):
        with pytest.raises(ChannelResolutionError, match="Cannot resolve"):
            resolve_channel_indices(FOUR_CH, ["UnknownMarker"])

    def test_mixed_specs(self):
        result = resolve_channel_indices(FOUR_CH, ["0", "647", "PM"], filename="PM750_NeuN647.czi")
        assert result["0"] == 0
        assert result["647"] == 1
        assert result["PM"] == 2  # PM -> 750nm from filename -> ch2


class TestParseMarkersFromFilename:
    def test_basic_pattern(self):
        markers = parse_markers_from_filename("SMA647_CD31555.czi")
        assert len(markers) == 2
        assert markers[0]["name"] == "SMA"
        assert markers[0]["wavelength"] == 647
        assert markers[1]["name"] == "CD31"
        assert markers[1]["wavelength"] == 555

    def test_nuc_pattern(self):
        markers = parse_markers_from_filename("nuc488_PM750.czi")
        names = [m["name"] for m in markers]
        assert "nuc" in names or "PM" in names

    def test_wavelength_before_name(self):
        markers = parse_markers_from_filename("488Slc17a7_647Gad1.czi")
        wavelengths = [m["wavelength"] for m in markers]
        assert 488 in wavelengths
        assert 647 in wavelengths

    def test_empty_filename(self):
        markers = parse_markers_from_filename("slide001.czi")
        assert markers == []
