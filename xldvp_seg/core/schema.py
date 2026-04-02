"""Detection schema — canonical structure for pipeline detection dicts.

Provides validation at load/export boundaries without imposing overhead
in hot loops.  The pipeline still uses plain dicts internally for
performance; ``Detection`` is for validation and documentation.

Usage:
    from xldvp_seg.core.schema import Detection

    # Validate a detection dict (tolerates extra keys, handles legacy names)
    det = Detection.from_dict(raw_dict)

    # Serialize back to JSON-compatible dict
    d = det.to_dict()

    # Validate a batch
    validated = [Detection.from_dict(d) for d in detections]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Detection:
    """Canonical detection schema.

    Required fields are those produced by every detection strategy.
    Optional fields are added by post-dedup, classification, or
    marker analysis stages.
    """

    # --- Required (always present after detection) ---
    uid: str
    cell_type: str
    global_center: list[float]
    global_center_um: list[float]
    tile_origin: list[int]
    mask_label: int
    pixel_size_um: float

    # --- Optional (added by pipeline stages) ---
    slide_name: str = ""
    contour_px: list[list[float]] | None = None
    contour_um: list[list[float]] | None = None
    rf_prediction: float | None = None
    marker_profile: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    nuclei: list[dict] | None = None

    # --- Provenance ---
    pipeline_version: str = ""
    feature_extraction: str = "original_mask"

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        """Construct from a detection dict, tolerating extra keys and legacy names.

        Handles backwards compatibility:
        - ``contour_dilated_px`` → ``contour_px``
        - ``contour_dilated_um`` → ``contour_um``
        - ``id`` → ``uid`` (if uid missing)
        """
        # Resolve legacy field names
        uid = d.get("uid") or d.get("id", "")
        contour_px = d.get("contour_px")
        if contour_px is None:
            contour_px = d.get("contour_dilated_px")
        contour_um = d.get("contour_um")
        if contour_um is None:
            contour_um = d.get("contour_dilated_um")

        return cls(
            uid=uid,
            cell_type=d.get("cell_type", ""),
            global_center=d.get("global_center", [0, 0]),
            global_center_um=d.get("global_center_um", [0, 0]),
            tile_origin=d.get("tile_origin", [0, 0]),
            mask_label=d.get("mask_label", 0),
            pixel_size_um=d.get("pixel_size_um", 0.0),
            slide_name=d.get("slide_name") or d.get("slide", ""),
            contour_px=contour_px,
            contour_um=contour_um,
            rf_prediction=d.get("rf_prediction"),
            marker_profile=d.get("marker_profile")
            or (d.get("features") or {}).get("marker_profile"),
            features=d.get("features") or {},
            nuclei=d.get("nuclei"),
            pipeline_version=d.get("pipeline_version", ""),
            feature_extraction=d.get("feature_extraction", "original_mask"),
        )

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (same format as pipeline output)."""
        d = asdict(self)
        # Remove None optional fields to match pipeline output format
        for key in ("contour_px", "contour_um", "rf_prediction", "marker_profile", "nuclei"):
            if d.get(key) is None:
                del d[key]
        # Remove empty provenance
        if not d.get("pipeline_version"):
            del d["pipeline_version"]
        if d.get("feature_extraction") == "original_mask":
            del d["feature_extraction"]
        return d

    @staticmethod
    def validate_batch(detections: list[dict]) -> list[str]:
        """Validate a batch of detection dicts. Returns list of error messages (empty = valid)."""
        errors = []
        for i, d in enumerate(detections):
            if not d.get("uid") and not d.get("id"):
                errors.append(f"Detection {i}: missing uid/id")
            if not d.get("global_center"):
                errors.append(f"Detection {i}: missing global_center")
            if d.get("mask_label") is None and d.get("contour_px") is None:
                errors.append(f"Detection {i}: missing both mask_label and contour_px")
        return errors
