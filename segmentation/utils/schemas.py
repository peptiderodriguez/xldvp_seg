"""
JSON Schema validation for all data files in the segmentation pipeline.

Uses Pydantic for validation with clear error messages.

Usage:
    from shared.schemas import (
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Base Types
# =============================================================================

class Coordinate(BaseModel):
    """A 2D coordinate [x, y]."""
    x: float
    y: float

    @classmethod
    def from_list(cls, coords: List[float]) -> "Coordinate":
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
    area: Optional[float] = None
    eccentricity: Optional[float] = None
    solidity: Optional[float] = None
    mean_intensity: Optional[float] = None
    perimeter: Optional[float] = None

    class Config:
        extra = "allow"  # Allow additional features


class NMJFeatures(BaseFeatures):
    """Features specific to NMJ detections."""
    skeleton_length: Optional[float] = None
    elongation: Optional[float] = None
    prob_nmj: Optional[float] = None
    confidence: Optional[float] = None


class VesselFeatures(BaseFeatures):
    """Features specific to vessel detections."""
    outer_diameter_um: Optional[float] = None
    inner_diameter_um: Optional[float] = None
    wall_thickness_mean_um: Optional[float] = None
    wall_thickness_std_um: Optional[float] = None
    lumen_area_um2: Optional[float] = None
    wall_area_um2: Optional[float] = None
    ring_completeness: Optional[float] = None
    cd31_validated: Optional[bool] = None


class MKFeatures(BaseFeatures):
    """Features specific to MK detections."""
    is_megakaryocyte: Optional[bool] = None
    mk_confidence: Optional[float] = None


# =============================================================================
# Detection Schemas
# =============================================================================

class Detection(BaseModel):
    """A single cell detection."""
    uid: str = Field(..., description="Unique identifier: slide_celltype_x_y")
    global_center: List[float] = Field(..., min_length=2, max_length=2)
    tile_origin: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    local_centroid: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    features: Optional[Dict[str, Any]] = None
    area_px: Optional[float] = None
    area_um2: Optional[float] = None
    image_b64: Optional[str] = None

    @field_validator('uid')
    @classmethod
    def validate_uid(cls, v: str) -> str:
        """Validate UID format."""
        parts = v.split('_')
        if len(parts) < 3:
            raise ValueError(f"UID must have at least 3 parts separated by '_', got: {v}")
        return v

    class Config:
        extra = "allow"


class DetectionFile(BaseModel):
    """Schema for *_detections.json files."""
    slide_name: str
    cell_type: Literal["mk", "cell", "nmj", "vessel", "mesothelium", "hspc"]
    experiment_name: Optional[str] = None
    pixel_size_um: Optional[float] = None
    total_detections: Optional[int] = None
    tiles_processed: Optional[int] = None
    tiles_with_detections: Optional[int] = None
    timestamp: Optional[str] = None
    detections: List[Detection] = Field(default_factory=list)

    # Also allow 'nmjs' key for backwards compatibility
    nmjs: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode='after')
    def merge_nmjs_to_detections(self) -> "DetectionFile":
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
    experiment_name: Optional[str] = None
    cell_type: Optional[Literal["mk", "cell", "nmj", "vessel", "mesothelium", "hspc"]] = None
    slide_name: Optional[str] = None
    channel: Optional[int] = None
    pixel_size_um: Optional[float] = None
    normalization_percentiles: Optional[List[float]] = None
    contour_color: Optional[List[int]] = None
    contour_thickness: Optional[int] = None
    samples_per_page: Optional[int] = None
    tile_size: Optional[int] = None

    class Config:
        extra = "allow"


# =============================================================================
# Annotation Schemas
# =============================================================================

class AnnotationsOldFormat(BaseModel):
    """Old annotation format: {positive: [...], negative: [...]}"""
    positive: List[str] = Field(default_factory=list)
    negative: List[str] = Field(default_factory=list)


class AnnotationsNewFormat(BaseModel):
    """New annotation format: {annotations: {uid: "yes"|"no"}}"""
    annotations: Dict[str, Literal["yes", "no", "unsure"]]


class Annotations(BaseModel):
    """Unified annotation schema supporting both formats."""
    # Old format
    positive: Optional[List[str]] = None
    negative: Optional[List[str]] = None

    # New format
    annotations: Optional[Dict[str, Literal["yes", "no", "unsure"]]] = None

    def to_unified(self) -> Dict[str, str]:
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
    centroid: List[float] = Field(..., min_length=2, max_length=2)
    area: int
    skeleton_length: Optional[int] = None
    elongation: Optional[float] = None
    eccentricity: Optional[float] = None
    mean_intensity: Optional[float] = None
    bbox: Optional[List[int]] = None
    perimeter: Optional[float] = None
    solidity: Optional[float] = None

    class Config:
        extra = "allow"


# =============================================================================
# Validation Functions
# =============================================================================

def validate_json_file(
    file_path: Union[str, Path],
    schema: type[BaseModel],
    raise_on_error: bool = True
) -> Optional[BaseModel]:
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
        with open(file_path, 'r') as f:
            data = json.load(f)

        return schema.model_validate(data)

    except json.JSONDecodeError as e:
        if raise_on_error:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        return None

    except Exception as e:
        if raise_on_error:
            raise ValueError(f"Validation failed for {file_path}: {e}")
        return None


def validate_detection_file(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> Optional[DetectionFile]:
    """Validate a *_detections.json file."""
    return validate_json_file(file_path, DetectionFile, raise_on_error)


def validate_config_file(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> Optional[Config]:
    """Validate a config.json file."""
    return validate_json_file(file_path, Config, raise_on_error)


def validate_annotations_file(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> Optional[Annotations]:
    """Validate an annotations JSON file."""
    return validate_json_file(file_path, Annotations, raise_on_error)


def validate_nmj_features_file(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> Optional[List[NMJFeatureFile]]:
    """Validate an nmj_features.json file (list of features)."""
    file_path = Path(file_path)

    if not file_path.exists():
        if raise_on_error:
            raise FileNotFoundError(f"File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("nmj_features.json must be a list")

        return [NMJFeatureFile.model_validate(item) for item in data]

    except Exception as e:
        if raise_on_error:
            raise ValueError(f"Validation failed for {file_path}: {e}")
        return None


def infer_and_validate(
    file_path: Union[str, Path],
    raise_on_error: bool = True
) -> Optional[BaseModel]:
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
