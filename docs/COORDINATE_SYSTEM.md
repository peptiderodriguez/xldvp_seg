# Coordinate System Specification

This document defines the canonical coordinate system and UID format used throughout the vessel segmentation codebase.

## Coordinate System Overview

### Image Coordinate System

The codebase uses an **image coordinate system** with the following conventions:

```
Origin (0,0) -----> X (columns, width)
    |
    |
    v
    Y (rows, height)
```

- **Origin**: Top-left corner of the image (0, 0)
- **X-axis**: Horizontal, increases to the right (corresponds to columns)
- **Y-axis**: Vertical, increases downward (corresponds to rows)
- **Units**: Pixels (integer or float depending on context)

### Coordinate Formats

| Format | Description | Usage |
|--------|-------------|-------|
| `[x, y]` | List format | All exports, JSON, centroids in Detection objects |
| `(x, y)` | Tuple format | Function parameters, return values |
| `(row, col)` | Array index format | NumPy array indexing `array[row, col]` |

### Important: NumPy vs Image Coordinates

NumPy arrays use `[row, col]` indexing, which is the **opposite** of `[x, y]`:

```python
# NumPy indexing: array[row, col] = array[y, x]
pixel_value = image[y, x]  # NOT image[x, y]

# Correct conversion:
row, col = int(y), int(x)
x, y = col, row
```

### Scikit-image regionprops

Scikit-image's `regionprops` returns centroids as `(row, col)` = `(y, x)`:

```python
from skimage.measure import regionprops

props = regionprops(mask)
for prop in props:
    # prop.centroid is (row, col) = (y, x)
    y, x = prop.centroid
    centroid_xy = [x, y]  # Convert to [x, y] for export
```

Use the helper function for conversion:
```python
from segmentation.processing.coordinates import regionprop_centroid_to_xy

centroid_xy = regionprop_centroid_to_xy(prop)  # Returns [x, y]
```

## Global vs Local Coordinates

### Tile-Local Coordinates

Coordinates relative to a tile's top-left corner:
- Range: `[0, 0]` to `[tile_width, tile_height]`
- Used during tile processing

### Global (Mosaic) Coordinates

Coordinates in the full mosaic/slide space:
- Origin: Top-left of the full mosaic
- Used for exports, UIDs, and cross-tile operations

### Conversion Functions

```python
from segmentation.processing.coordinates import (
    tile_to_global_coords,
    global_to_tile_coords,
)

# Local to global
global_x, global_y = tile_to_global_coords(local_x, local_y, tile_origin_x, tile_origin_y)

# Global to local
local_x, local_y = global_to_tile_coords(global_x, global_y, tile_origin_x, tile_origin_y)
```

## Unique Identifier (UID) Format

### Canonical UID Format

All cell types use a **spatial UID format**:

```
{slide_name}_{cell_type}_{round(global_x)}_{round(global_y)}
```

**Examples:**
- `2025_11_18_FGC1_mk_12346_67890`
- `slide_01_hspc_5000_3000`
- `muscle_sample_nmj_1234_5678`
- `tissue_vessel_9876_5432`

### UID Components

| Component | Description | Example |
|-----------|-------------|---------|
| `slide_name` | Slide/sample identifier | `2025_11_18_FGC1` |
| `cell_type` | Detection type | `mk`, `hspc`, `nmj`, `vessel` |
| `round(global_x)` | Rounded X coordinate in pixels | `12346` |
| `round(global_y)` | Rounded Y coordinate in pixels | `67890` |

### UID Generation

```python
from segmentation.processing.coordinates import generate_uid

uid = generate_uid(
    slide_name="2025_11_18_FGC1",
    cell_type="mk",
    global_x=12345.6,
    global_y=67890.3
)
# Returns: "2025_11_18_FGC1_mk_12346_67890"
```

### Why Spatial UIDs?

1. **Globally unique**: Coordinates are unique per slide
2. **Self-documenting**: UID reveals detection location
3. **Consistent**: Same format across all cell types
4. **Reproducible**: Same detection always gets same UID

### Legacy Format (Deprecated)

The old numeric `global_id` format is deprecated:
```
{slide_name}_{cell_type}_{global_id}  # OLD - do not use
```

Use migration utilities if converting old data:
```python
from segmentation.processing.coordinates import migrate_uid_format

new_uid = migrate_uid_format(
    old_uid="slide_mk_123",
    global_x=12345,
    global_y=67890
)
```

## Export Coordinate Labeling

### JSON Exports

All coordinate fields must be explicitly labeled:

```json
{
    "uid": "slide_mk_12346_67890",
    "centroid_xy": [12345.6, 67890.3],
    "global_center_xy": [12345.6, 67890.3],
    "tile_origin_xy": [12000, 67000],
    "local_centroid_xy": [345.6, 890.3],
    "bbox_xyxy": [12300, 67800, 12400, 67950]
}
```

**Naming conventions:**
- `_xy` suffix: `[x, y]` order
- `_xyxy` suffix: `[x1, y1, x2, y2]` bounding box
- `_rowcol` suffix: `[row, col]` order (rare, avoid in exports)

### CSV Exports

Column names must indicate coordinate system:

```csv
uid,global_x_px,global_y_px,global_x_um,global_y_um
slide_mk_12346_67890,12345.6,67890.3,2715.8,14935.9
```

## Coordinate Validation

Use validation functions to catch errors early:

```python
from segmentation.processing.coordinates import (
    validate_xy_coordinates,
    validate_array_indices,
)

# Validate [x, y] coordinates
validate_xy_coordinates(x, y, image_width, image_height)

# Validate array indices
validate_array_indices(row, col, array_height, array_width)
```

## Summary of Key Rules

1. **Always store coordinates as `[x, y]`** in exports and Detection objects
2. **Use `(row, col)` only for NumPy indexing**: `array[row, col]`
3. **Convert regionprops centroids**: They return `(y, x)` not `(x, y)`
4. **Use spatial UIDs**: `{slide}_{type}_{x}_{y}` format
5. **Label all coordinate fields**: Include `_xy` suffix in field names
6. **Validate coordinates**: Use provided validation functions

## Utility Functions Reference

All coordinate utilities are in `segmentation/processing/coordinates.py`:

| Function | Purpose |
|----------|---------|
| `regionprop_centroid_to_xy(prop)` | Convert regionprops centroid to [x, y] |
| `xy_to_array_index(x, y)` | Convert [x, y] to (row, col) for array indexing |
| `array_index_to_xy(row, col)` | Convert (row, col) to (x, y) |
| `tile_to_global_coords(...)` | Convert local to global coordinates |
| `global_to_tile_coords(...)` | Convert global to local coordinates |
| `generate_uid(...)` | Generate canonical spatial UID |
| `parse_uid(uid)` | Extract components from UID |
| `migrate_uid_format(...)` | Convert legacy global_id UIDs |
| `validate_xy_coordinates(...)` | Validate x, y within bounds |
| `validate_array_indices(...)` | Validate row, col within bounds |
| `extract_crop_bounds(...)` | Calculate crop region centered on point |
| `extract_crop(...)` | Extract image crop with bounds checking |
