"""Tests for Detection schema validation."""

from xldvp_seg.core.schema import Detection


def _sample_det(**overrides):
    d = {
        "uid": "slide_cell_100_200",
        "cell_type": "cell",
        "global_center": [100.5, 200.3],
        "global_center_um": [22.1, 44.1],
        "tile_origin": [0, 0],
        "mask_label": 3,
        "pixel_size_um": 0.22,
        "slide_name": "test_slide",
        "contour_px": [[90, 190], [110, 190], [110, 210], [90, 210]],
        "features": {"area": 500, "solidity": 0.92},
    }
    d.update(overrides)
    return d


class TestDetectionFromDict:
    def test_basic_roundtrip(self):
        d = _sample_det()
        det = Detection.from_dict(d)
        assert det.uid == "slide_cell_100_200"
        assert det.cell_type == "cell"
        assert det.pixel_size_um == 0.22
        result = det.to_dict()
        assert result["uid"] == d["uid"]
        assert result["features"]["area"] == 500

    def test_legacy_contour_field(self):
        d = _sample_det()
        d.pop("contour_px")
        d["contour_dilated_px"] = [[1, 2], [3, 4], [5, 6]]
        det = Detection.from_dict(d)
        assert det.contour_px == [[1, 2], [3, 4], [5, 6]]

    def test_legacy_id_field(self):
        d = _sample_det()
        d.pop("uid")
        d["id"] = "legacy_id_123"
        det = Detection.from_dict(d)
        assert det.uid == "legacy_id_123"

    def test_missing_optional_fields(self):
        d = _sample_det()
        det = Detection.from_dict(d)
        assert det.rf_prediction is None
        assert det.nuclei is None

    def test_to_dict_strips_none(self):
        d = _sample_det()
        det = Detection.from_dict(d)
        result = det.to_dict()
        assert "rf_prediction" not in result
        assert "nuclei" not in result

    def test_extra_keys_tolerated(self):
        d = _sample_det(extra_key="ignored", another=42)
        det = Detection.from_dict(d)
        assert det.uid == "slide_cell_100_200"


class TestValidateBatch:
    def test_valid_batch(self):
        dets = [_sample_det(), _sample_det(uid="det2")]
        errors = Detection.validate_batch(dets)
        assert errors == []

    def test_missing_uid(self):
        d = _sample_det()
        del d["uid"]
        errors = Detection.validate_batch([d])
        assert any("missing uid" in e for e in errors)

    def test_missing_global_center(self):
        d = _sample_det()
        del d["global_center"]
        errors = Detection.validate_batch([d])
        assert any("missing global_center" in e for e in errors)

    def test_missing_mask_and_contour(self):
        d = _sample_det()
        del d["mask_label"]
        del d["contour_px"]
        errors = Detection.validate_batch([d])
        assert any("missing both mask_label and contour_px" in e for e in errors)
