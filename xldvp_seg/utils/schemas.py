"""
JSON Schema validation for all data files in the segmentation pipeline.

Uses Pydantic for validation with clear error messages.

Usage:
    from xldvp_seg.utils.schemas import (
        validate_detection_file,
        validate_config_file,
        validate_annotations_file,
        Detection, Config, Annotations
    )

    # Validate a file
    detections = validate_detection_file("/path/to/detections.json")

    # Create validated objects
    det = Detection(uid="slide_mk_100_200", global_center=[100, 200], ...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from xldvp_seg.exceptions import DataLoadError

# =============================================================================
# Base Types
# =============================================================================


class Coordinate(BaseModel):
    """A 2D coordinate [x, y]."""

    x: float
    y: float

    @classmethod
    def from_list(cls, coords: list[float]) -> Coordinate:
        """Create from [x, y] list."""
        if len(coords) != 2:
            raise ValueError(f"Coordinate must have 2 elements, got {len(coords)}")
        return cls(x=coords[0], y=coords[1])


class BoundingBox(BaseModel):
    """A bounding box [x1, y1, x2, y2] or [minr, minc, maxr, maxc]."""

    x1: float
    y1: float
    x2: float
    y2: float


# =============================================================================
# Feature Schemas
# =============================================================================


class BaseFeatures(BaseModel):
    """Common features for all detection types."""

    area: float | None = None
    eccentricity: float | None = None
    solidity: float | None = None
    mean_intensity: float | None = None
    perimeter: float | None = None

    class Config:
        extra = "allow"  # Allow additional features


class NMJFeatures(BaseFeatures):
    """Features specific to NMJ detections."""

    skeleton_length: float | None = None
    elongation: float | None = None
    prob_nmj: float | None = None
    confidence: float | None = None


class VesselFeatures(BaseFeatures):
    """Features specific to vessel detections."""

    outer_diameter_um: float | None = None
    inner_diameter_um: float | None = None
    wall_thickness_mean_um: float | None = None
    wall_thickness_std_um: float | None = None
    lumen_area_um2: float | None = None
    wall_area_um2: float | None = None
    ring_completeness: float | None = None
    cd31_validated: bool | None = None


class MKFeatures(BaseFeatures):
    """Features specific to MK detections."""

    is_megakaryocyte: bool | None = None
    mk_confidence: float | None = None


# =============================================================================
# Detection Schemas
# =============================================================================


class Detection(BaseModel):
    """A single cell detection."""

    uid: str = Field(..., description="Unique identifier: slide_celltype_x_y")
    global_center: list[float] = Field(..., min_length=2, max_length=2)
    tile_origin: list[float] | None = Field(None, min_length=2, max_length=2)
    local_centroid: list[float] | None = Field(None, min_length=2, max_length=2)
    features: dict[str, Any] | None = None
    area_px: float | None = None
    area_um2: float | None = None
    image_b64: str | None = None

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, v: str) -> str:
        """Validate UID format."""
        parts = v.split("_")
        if len(parts) < 3:
            raise ValueError(f"UID must have at least 3 parts separated by '_', got: {v}")
        return v

    class Config:
        extra = "allow"


class DetectionFile(BaseModel):
    """Schema for *_detections.json files."""

    slide_name: str
    cell_type: Literal[
        "mk", "cell", "nmj", "vessel", "mesothelium", "hspc", "islet", "tissue_pattern"
    ]
    experiment_name: str | None = None
    pixel_size_um: float | None = None
    total_detections: int | None = None
    tiles_processed: int | None = None
    tiles_with_detections: int | None = None
    timestamp: str | None = None
    detections: list[Detection] = Field(default_factory=list)

    # Also allow 'nmjs' key for backwards compatibility
    nmjs: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def merge_nmjs_to_detections(self) -> DetectionFile:
        """Merge 'nmjs' into 'detections' for backwards compatibility."""
        if self.nmjs and not self.detections:
            self.detections = [Detection(**n) for n in self.nmjs]
        return self

    class Config:
        extra = "allow"


# =============================================================================
# Configuration Schema
# =============================================================================


class Config(BaseModel):
    """Schema for config.json files."""

    experiment_name: str | None = None
    cell_type: (
        Literal["mk", "cell", "nmj", "vessel", "mesothelium", "hspc", "islet", "tissue_pattern"]
        | None
    ) = None
    slide_name: str | None = None
    channel: int | None = None
    pixel_size_um: float | None = None
    normalization_percentiles: list[float] | None = None
    contour_color: list[int] | None = None
    contour_thickness: int | None = None
    samples_per_page: int | None = None
    tile_size: int | None = None

    class Config:
        extra = "allow"


# =============================================================================
# Annotation Schemas
# =============================================================================


class AnnotationsOldFormat(BaseModel):
    """Old annotation format: {positive: [...], negative: [...]}"""

    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class AnnotationsNewFormat(BaseModel):
    """New annotation format: {annotations: {uid: "yes"|"no"}}"""

    annotations: dict[str, Literal["yes", "no", "unsure"]]


class Annotations(BaseModel):
    """Unified annotation schema supporting both formats."""

    # Old format
    positive: list[str] | None = None
    negative: list[str] | None = None

    # New format
    annotations: dict[str, Literal["yes", "no", "unsure"]] | None = None

    def to_unified(self) -> dict[str, str]:
        """Convert to unified format {uid: "yes"|"no"}."""
        result = {}

        if self.annotations:
            result.update(self.annotations)

        if self.positive:
            for uid in self.positive:
                result[uid] = "yes"

        if self.negative:
            for uid in self.negative:
                result[uid] = "no"

        return result

    @property
    def positive_count(self) -> int:
        unified = self.to_unified()
        return sum(1 for v in unified.values() if v == "yes")

    @property
    def negative_count(self) -> int:
        unified = self.to_unified()
        return sum(1 for v in unified.values() if v == "no")

    class Config:
        extra = "allow"


# =============================================================================
# NMJ-Specific Schemas
# =============================================================================


class NMJFeatureFile(BaseModel):
    """Schema for nmj_features.json files in tile directories."""

    id: str
    centroid: list[float] = Field(..., min_length=2, max_length=2)
    area: int
    skeleton_length: int | None = None
    elongation: float | None = None
    eccentricity: float | None = None
    mean_intensity: float | None = None
    bbox: list[int] | None = None
    perimeter: float | None = None
    solidity: float | None = None

    class Config:
        extra = "allow"


# =============================================================================
# Validation Functions
# =============================================================================


def validate_json_file(
    file_path: str | Path, schema: type[BaseModel], raise_on_error: bool = True
) -> BaseModel | None:
    """
    Validate a JSON file against a schema.

    Args:
        file_path: Path to JSON file
        schema: Pydantic model class to validate against
        raise_on_error: If True, raise exception on validation error

    Returns:
        Validated model instance, or None if validation fails and raise_on_error=False
    """
    file_path = Path(file_path)

    if not file_path.exists():
        if raise_on_error:
            raise FileNotFoundError(f"File not found: {file_path}")
        return None

    try:
        with open(file_path) as f:
            data = json.load(f)

        return schema.model_validate(data)

    except json.JSONDecodeError as e:
        if raise_on_error:
            raise DataLoadError(f"Invalid JSON in {file_path}: {e}")
        return None

    except Exception as e:
        if raise_on_error:
            raise DataLoadError(f"Validation failed for {file_path}: {e}")
        return None


def validate_detection_file(
    file_path: str | Path, raise_on_error: bool = True
) -> DetectionFile | None:
    """Validate a *_detections.json file."""
    return validate_json_file(file_path, DetectionFile, raise_on_error)


def validate_config_file(file_path: str | Path, raise_on_error: bool = True) -> Config | None:
    """Validate a config.json file."""
    return validate_json_file(file_path, Config, raise_on_error)


def validate_annotations_file(
    file_path: str | Path, raise_on_error: bool = True
) -> Annotations | None:
    """Validate an annotations JSON file."""
    return validate_json_file(file_path, Annotations, raise_on_error)


def validate_nmj_features_file(
    file_path: str | Path, raise_on_error: bool = True
) -> list[NMJFeatureFile] | None:
    """Validate an nmj_features.json file (list of features)."""
    file_path = Path(file_path)

    if not file_path.exists():
        if raise_on_error:
            raise FileNotFoundError(f"File not found: {file_path}")
        return None

    try:
        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise DataLoadError("nmj_features.json must be a list")

        return [NMJFeatureFile.model_validate(item) for item in data]

    except Exception as e:
        if raise_on_error:
            raise DataLoadError(f"Validation failed for {file_path}: {e}")
        return None


def infer_and_validate(file_path: str | Path, raise_on_error: bool = True) -> BaseModel | None:
    """
    Infer schema type from filename and validate.

    Args:
        file_path: Path to JSON file
        raise_on_error: If True, raise exception on validation error

    Returns:
        Validated model instance
    """
    file_path = Path(file_path)
    name = file_path.name.lower()

    if "detection" in name:
        return validate_detection_file(file_path, raise_on_error)
    elif name == "config.json":
        return validate_config_file(file_path, raise_on_error)
    elif "annotation" in name:
        return validate_annotations_file(file_path, raise_on_error)
    elif "features" in name:
        return validate_nmj_features_file(file_path, raise_on_error)
    else:
        # Try detection file as default
        return validate_detection_file(file_path, raise_on_error)
