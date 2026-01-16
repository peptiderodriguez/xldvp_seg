# Comprehensive TODO List for AI Agent

This document contains all identified improvements, bug fixes, and optimizations for the xldvp_seg_repo unified cell segmentation pipeline.

## Priority Legend
- **P0**: Critical - Must fix before use
- **P1**: High - Important for production use
- **P2**: Medium - Nice to have improvements
- **P3**: Low - Future enhancements

---

## 1. Logging Updates (P2)

The following scripts still use `print()` statements and should be converted to use `shared.logging_config`:

### Scripts to Update
- [ ] `run_unified_FAST.py` - Many print statements throughout
- [ ] `run_nmj_segmentation.py` - Print statements in main processing loop
- [ ] `export_nmj_results_html.py` - Print statements in export functions
- [ ] `export_separate_mk_hspc.py` - Print statements throughout
- [ ] `regenerate_html.py` - Print statements for progress
- [ ] `train_nmj_classifier.py` - Training progress output
- [ ] `merge_nmj_annotations.py` - Status messages
- [ ] `serve_html.py` - Server start messages
- [ ] `run_lmd_export.py` - Export progress
- [ ] `shared/tissue_detection.py` - Calibration messages
- [ ] `shared/html_export.py` - Export progress

### Pattern to Apply
```python
from shared.logging_config import get_logger, setup_logging
logger = get_logger(__name__)

# Replace: print(f"message")
# With: logger.info(f"message")
# For errors: logger.error(f"message")
# For warnings: logger.warning(f"message")
# For debug: logger.debug(f"message")
```

---

## 2. CLI Completion (P1)

The `shared/cli.py` creates the unified CLI structure but returns errors asking to use specific scripts.

### Tasks
- [ ] Implement actual detection in `cmd_run()` by calling appropriate detector functions
- [ ] Add detector factory function to map cell_type to detector
- [ ] Wire up NMJ inference classifier loading
- [ ] Wire up SAM2/Cellpose/ResNet loading for MK/HSPC
- [ ] Add progress callbacks for batch processing
- [ ] Add `--verbose` / `-v` flag handling in all subcommands

### Code Location
`shared/cli.py:300-350` - `cmd_run()` function needs implementation

---

## 3. Performance Optimizations (P1)

### 3.1 Batch ResNet Feature Extraction
**File**: `run_unified_FAST.py`
**Issue**: ResNet features extracted one crop at a time
**Solution**: Batch multiple crops (8-16) for GPU efficiency

```python
# Current (slow):
for crop in crops:
    features = extract_resnet_features(crop)

# Better:
batch = torch.stack([transform(crop) for crop in crops[:16]])
features_batch = resnet(batch.to(device))
```

### 3.2 GPU-Accelerated Tissue Detection
**File**: `shared/tissue_detection.py`
**Issue**: Variance calculations use NumPy on CPU
**Solution**: Use CuPy for GPU acceleration when available

```python
try:
    import cupy as cp
    xp = cp  # Use GPU
except ImportError:
    xp = np  # Fallback to CPU

# Replace np operations with xp
```

### 3.3 Producer-Consumer Pipeline
**Issue**: CPU preprocessing and GPU inference are sequential
**Solution**: Overlap tile loading with GPU processing

```python
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Producer thread loads tiles while consumer does GPU inference
tile_queue = Queue(maxsize=4)
```

### 3.4 Memory-Mapped HDF5 Reading
**File**: `export_nmj_results_html.py`
**Issue**: Some HDF5 operations load full arrays
**Solution**: Use HDF5 slicing consistently

```python
# Current:
full_masks = f['masks'][:]
mask_region = full_masks[y1:y2, x1:x2]

# Better:
mask_region = f['masks'][y1:y2, x1:x2]
```

---

## 4. Code Unification (P2)

### 4.1 Migrate Scripts to Use Shared Modules

| Script | Should Use |
|--------|-----------|
| `run_unified_FAST.py` | `CZILoader`, `coordinates.py`, `DetectionPipeline` |
| `run_nmj_segmentation.py` | `CZILoader`, `coordinates.py` |
| `export_nmj_results_html.py` | `html_export.py` dark theme |
| `regenerate_html.py` | Already uses shared modules |

### 4.2 Consolidate Duplicate Code
- [ ] Remove duplicate `percentile_normalize()` from remaining scripts
- [ ] Remove duplicate `draw_mask_contour()` implementations
- [ ] Remove duplicate `image_to_base64()` implementations
- [ ] Consolidate CZI loading code to use `CZILoader` everywhere

### 4.3 Standardize Configuration
- [ ] Use `shared/config.py` for all default values
- [ ] Ensure all scripts respect environment variables
- [ ] Create config validation on startup

---

## 5. Bug Fixes (P1)

### 5.1 Coordinate System Consistency
**Status**: Mostly fixed, verify remaining files
**Files to Check**:
- [ ] `run_unified_FAST.py` - Verify [x, y] order
- [ ] `export_separate_mk_hspc.py` - Verify coordinate handling
- [ ] `train_nmj_classifier.py` - Verify crop extraction

### 5.2 localStorage Key Isolation
**Status**: Fixed in `export_nmj_results_html.py`
**Verify**: All HTML exports use per-experiment storage keys

### 5.3 UID Collision Prevention
**Status**: Fixed (using `round()` instead of `int()`)
**Verify**: `run_segmentation.py:2027` uses `round()`

---

## 6. Schema Validation Integration (P2)

### 6.1 Add Validation to All JSON Outputs
- [ ] `run_segmentation.py` - Validate detection JSON before saving
- [ ] `run_nmj_inference.py` - Validate output
- [ ] `run_unified_FAST.py` - Validate batch results

### 6.2 Add Validation to All JSON Inputs
- [ ] Validate config.json on load
- [ ] Validate annotation files on load
- [ ] Add helpful error messages for invalid files

### Code Pattern
```python
from shared.schemas import validate_detection_file, DetectionFile

# Before saving
DetectionFile.model_validate(detection_data)

# On loading
validated = validate_detection_file(path)
```

---

## 7. Testing (P2)

### 7.1 Create Unit Tests
- [ ] `test_coordinates.py` - Test coordinate conversions
- [ ] `test_schemas.py` - Test JSON validation
- [ ] `test_config.py` - Test configuration loading
- [ ] `test_tissue_detection.py` - Test with synthetic data

### 7.2 Create Integration Tests
- [ ] Test full NMJ pipeline on sample data
- [ ] Test batch processing with multiple slides
- [ ] Test HTML export and annotation workflow

---

## 8. Documentation (P3)

### 8.1 Update CLAUDE.md
- [ ] Add coordinate system specification section
- [ ] Add schema documentation
- [ ] Add CLI usage examples
- [ ] Add troubleshooting section for common issues

### 8.2 Add Docstrings
- [ ] `shared/coordinates.py` - Full docstrings
- [ ] `shared/czi_loader.py` - Full docstrings
- [ ] `shared/detection_pipeline.py` - Full docstrings
- [ ] `shared/batch.py` - Full docstrings

---

## 9. Feature Enhancements (P3)

### 9.1 Add Resume Capability
- [ ] Save processing state to checkpoint file
- [ ] Allow resuming from last processed tile
- [ ] Track which tiles are complete in batch

### 9.2 Add Progress Webhooks
- [ ] Optional webhook URL for progress updates
- [ ] JSON payload with completion percentage
- [ ] Useful for long-running batch jobs

### 9.3 Add Multi-GPU Support
- [ ] Distribute tiles across available GPUs
- [ ] Balance load for heterogeneous GPU setups

### 9.4 Add Export Formats
- [ ] Add GeoJSON export for spatial analysis
- [ ] Add QuPath-compatible format
- [ ] Add OMERO integration

---

## 10. Installation and Packaging (P2)

### 10.1 Complete pyproject.toml
- [ ] Add all optional dependencies
- [ ] Add dev dependencies
- [ ] Add test dependencies
- [ ] Configure pytest

### 10.2 Create Docker Image
- [ ] Dockerfile with CUDA support
- [ ] Include all model checkpoints
- [ ] Add docker-compose for easy deployment

### 10.3 Add Installation Scripts
- [ ] `install.sh` for Linux
- [ ] Conda environment.yml
- [ ] Verify with clean install

---

## Quick Start for AI Agent

1. **Read first**: `CLAUDE.md` for full project context
2. **Test changes**: Run `python -c "import shared"` to verify imports
3. **Key files**:
   - `shared/__init__.py` - All exports
   - `shared/logging_config.py` - Logging setup
   - `shared/schemas.py` - JSON validation
   - `shared/cli.py` - Unified CLI
   - `shared/batch.py` - Batch processing
4. **Environment**: `conda activate mkseg`
5. **Output locations**: See `shared/config.py` for DEFAULT_PATHS

---

## Files Modified in Recent Session

| File | Changes |
|------|---------|
| `pyproject.toml` | NEW - Package configuration |
| `shared/logging_config.py` | NEW - Logging setup |
| `shared/schemas.py` | NEW - Pydantic schemas |
| `shared/cli.py` | NEW - Unified CLI |
| `shared/batch.py` | NEW - Batch processing |
| `shared/coordinates.py` | NEW - Coordinate helpers |
| `shared/czi_loader.py` | NEW - CZI loading |
| `shared/detection_pipeline.py` | NEW - Detection framework |
| `shared/__init__.py` | UPDATED - New exports |
| `shared/config.py` | UPDATED - Environment paths |
| `shared/html_export.py` | UPDATED - Vectorized normalization |
| `run_segmentation.py` | UPDATED - Logging |
| `run_nmj_inference.py` | UPDATED - Logging, coordinate fix |
| `export_nmj_results_html.py` | UPDATED - localStorage key |
| `regenerate_html.py` | UPDATED - Vectorized mask loop |

---

*Last updated: January 2026*
