# xldvp_seg_repo - Comprehensive Improvement TODO List

Generated: 2026-01-15
Updated: 2026-01-15

## ✅ COMPLETED THIS SESSION

1. **Logging Infrastructure** - All scripts converted from print() to logger
2. **RAM-First Architecture** - `get_loader()` singleton with multi-channel cache
3. **Package Restructure** - `segmentation/` with `io/`, `detection/`, `processing/`, `utils/`
4. **CellDetector + Strategies** - Unified detector with pluggable strategies:
   - `CellDetector` base class with lazy-loaded SAM2, Cellpose, ResNet
   - `MKStrategy` - SAM2 auto-mask → size filter → classifier
   - `CellStrategy` - Cellpose → SAM2 refine (generic cell detection)
   - `NMJStrategy` - Intensity threshold + skeleton → classifier
   - `VesselStrategy` - Contour hierarchy → wall thickness
5. **Batch ResNet Feature Extraction** - GPU-efficient batched forward passes
6. **CLI Detector Wiring** - `segmentation run <cell_type>` works

---

## Priority Summary

### CRITICAL (Fix First)
1. Duplicate function definitions causing inconsistent behavior
2. GPU memory leak in batch processing (OOM crashes)
3. Missing bounds checking (index errors)
4. Bare except blocks (silent failures)
5. Tile size mismatch (silently wrong outputs)

### HIGH (Major Impact)
1. ~~Missing GPU batching opportunities~~ ✅ Batch ResNet added
2. ~~Network I/O inefficiency~~ ✅ RAM-first architecture
3. CPU core underutilization (48 cores available, using ~38)
4. ~~Code duplication in run_unified_FAST.py~~ ✅ CellDetector + strategies created

### MEDIUM (Maintenance/Correctness)
1. Deep nesting hard to debug
2. Main functions too large, not testable
3. Inconsistent coordinate handling (bug-prone)
4. Missing multiprocessing for I/O (2-3x speedup)

---

## 1. UNIFICATION/ABSTRACTION

### 1.1 Duplicate Function Definitions (HIGH)
**Files:** `run_unified_FAST.py:71-136`, `run_segmentation.py:79-84`
**Issue:** `create_hdf5_dataset()`, `draw_mask_contour()`, `percentile_normalize()`, `get_largest_connected_component()` defined in multiple files
**Fix:** Remove from main scripts, import from `shared/html_export.py`

### 1.2 Duplicate Tissue Detection Code (HIGH)
**Files:** `run_unified_FAST.py:910-1028`, `shared/tissue_detection.py`
**Issue:** Same functions in both files
**Fix:** Remove from `run_unified_FAST.py`, use shared module

### 1.3 Duplicate CZI Reading Patterns (MEDIUM) ✅ DONE
**Files:** `run_segmentation.py`, `run_nmj_inference.py`, `export_nmj_results_html.py`
**Issue:** Each has similar tile reading logic
**Fix:** All should use `shared/czi_loader.py:CZILoader` consistently
**Status:** All files now use `get_loader()` from `shared/czi_loader.py`

### 1.4 HTML Generation Patterns (MEDIUM)
**Files:** `run_nmj_segmentation.py:134-382`, `export_nmj_results_html.py`, `run_unified_FAST.py`
**Issue:** Duplicate JavaScript logic, card generation, localStorage handling
**Fix:** Create unified `HTMLPageGenerator` class in `shared/html_export.py`

### 1.5 Feature Dictionary Creation (MEDIUM)
**Files:** Multiple detector files
**Issue:** Each creates similar feature dicts with overlapping fields
**Fix:** Create `FeatureBuilder` base class with common fields

---

## 2. BUG/ERROR FIXES

### 2.1 Missing Bounding Box Bounds Checks (HIGH - CRASH RISK)
**File:** `run_segmentation.py:1921-1945`
**Issue:** Crop bounds extracted without verifying mask bounds match tile bounds
**Risk:** Index out of bounds errors
**Fix:** Add assertion `y1 < y2 and x1 < x2` before crop extraction

### 2.2 Unchecked Tile Size Mismatch (MEDIUM)
**Files:** `run_segmentation.py:1640-1643`, `run_nmj_segmentation.py:464-466`
**Issue:** Tile generation doesn't validate tiles fit within mosaic
**Risk:** Last tiles may be smaller, causing dimension mismatches
**Fix:** Adjust final tile size or pad with zeros

### 2.3 GPU Memory Leak in Batch Processing (HIGH - OOM)
**Files:** `run_nmj_inference.py:295`, `run_unified_FAST.py`
**Issue:** Classifier loops don't clear GPU cache between batches
**Fix:** Add `torch.cuda.empty_cache()` after each batch in long loops

### 2.4 Bare `except:` Blocks (MEDIUM)
**Files:** `tissue_detection.py:143,276`, `run_nmj_segmentation.py:498`, `export_nmj_results_html.py:88`
**Issue:** Real bugs get swallowed silently
**Fix:** Use `except Exception as e:` with logging at minimum

### 2.5 Missing Type Coercion on CZI Metadata (MEDIUM)
**File:** `run_nmj_inference.py:189-195`
**Issue:** XML parsing fallback doesn't log when using default pixel size
**Fix:** Log warning when fallback used, validate pixel_size is reasonable (0.1-1.0 µm/px)

### 2.6 Race Condition in Tile Processing (MEDIUM)
**File:** `run_unified_FAST.py:2408-2450`
**Issue:** ThreadPoolExecutor writing without coordination
**Fix:** Ensure HDF5 writes are atomic or use thread-safe wrapper

---

## 3. CODE FLOW IMPROVEMENTS

### 3.1 Deep Nesting in Tile Processing (MEDIUM)
**File:** `run_segmentation.py:2100-2250`
**Issue:** 6+ levels of nesting
**Fix:** Extract into named functions: `process_tile_detections()`, `extract_tile_features()`, `save_tile_outputs()`

### 3.2 Main Function Too Large (MEDIUM)
**File:** `run_segmentation.py:1980-2250`
**Issue:** 250+ line function handles everything
**Fix:** Break into discrete stages with SavePoint pattern

### 3.3 Inconsistent Coordinate Handling (MEDIUM - BUG SOURCE)
**Files:** Multiple files use [x, y] vs [row, col] inconsistently
**Fix:** Use type hints and doc comments "(x, y)" consistently, use `shared/coordinates.py` everywhere

### 3.4 Long Cell Type Dispatch Logic (MEDIUM)
**File:** `run_segmentation.py:2150-2200`
**Issue:** Large if/elif chains
**Fix:** Use Strategy pattern with dict of detector functions

---

## 4. SPEED/PERFORMANCE

### 4.1 GPU Batching (HIGH - 5-10x speedup)
**File:** `run_nmj_inference.py:104-146`
**Issue:** RTX 4090 can handle batch_size=64-128, currently smaller
**Fix:** Profile memory, increase batch size

### 4.2 RAM-First Architecture (HIGH - 10-100x speedup) ✅ DONE
**Issue:** Tile reads go over network repeatedly; multiple functions reload same data
**System:** 432GB RAM available - should load CZI channels ONCE, all code references RAM
**Status:** IMPLEMENTED - `get_loader()` singleton pattern with multi-channel support
- `shared/czi_loader.py` enhanced with global cache
- All pipelines use `get_loader()` → returns cached loader
- `loader.get_tile()` slices from RAM instantly
- Multi-channel support for CD31 validation etc.

### 4.3 CPU-GPU Pipeline Not Overlapped (MEDIUM - 20% speedup)
**File:** `run_unified_FAST.py:2371-2450`
**Issue:** Preprocessing and GPU sequential, not pipelined
**Fix:** Use queue for inter-process communication, preprocess batch N+1 while GPU processes N

### 4.4 NumPy Vectorization (MEDIUM)
**File:** `run_segmentation.py:91-145`
**Issue:** Loop-based HSV conversion
```python
# Current (slow)
hsv = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in masked_pixels[:100]])
# Better (vectorized)
hsv = matplotlib.colors.rgb_to_hsv(masked_pixels[:100] / 255)
```

### 4.5 HDF5 Storage Layout (MEDIUM - 20% I/O speedup)
**Issue:** Masks stored as uint32, could be uint8 or packed binary
**Fix:** Use chunks, compression:
```python
f.create_dataset('masks', data=data, chunks=(512, 512), compression='lz4')
```

---

## 5. RESOURCE UTILIZATION

### 5.1 Multiprocessing for Tile I/O (HIGH - 3-5x speedup)
**Files:** `run_nmj_segmentation.py`, `run_nmj_inference.py`
**Issue:** Single-threaded tile reading
**Fix:** Producer-consumer pattern - Thread 1 reads tiles, Thread 2 processes

### 5.2 No Parallel Preprocessing (MEDIUM - 2-3x speedup)
**Issue:** Only `run_unified_FAST.py` uses ThreadPoolExecutor for preprocessing
**Fix:** Apply same pattern to `run_segmentation.py`, `run_nmj_inference.py`

### 5.3 CPU Core Underutilization (MEDIUM)
**Current:** 48 cores available, only ~38 used (80%)
**Fix:** Use ProcessPoolExecutor for:
- Per-tile mask processing
- Feature calculation
- HTML image generation

---

## 6. FILE-SPECIFIC ISSUES

### 6.1 `run_unified_FAST.py` (114KB, 2700 lines) ✅ PARTIALLY DONE
- Lines 71-136: Duplicated functions from shared/ (still needs cleanup)
- Lines 910-1028: Duplicated tissue detection (still needs cleanup)
- ~~Contains TWO separate pipelines that should be one~~ ✅ CellDetector created
- **DONE:** Created unified `CellDetector` class with pluggable strategies:
  ```
  segmentation/detection/
  ├── cell_detector.py    # Base class with lazy-loaded SAM2, Cellpose, ResNet
  ├── strategies/
  │   ├── base.py         # DetectionStrategy ABC + Detection dataclass
  │   ├── mk.py           # SAM2 direct → size filter → classifier
  │   ├── cell.py         # Cellpose → SAM2 refine (generic)
  │   ├── nmj.py          # Intensity threshold + skeleton → classifier
  │   └── vessel.py       # Contour hierarchy → wall thickness
  ```
- **REMAINING:** Wire main scripts to use new CellDetector instead of inline logic

### 6.2 `run_segmentation.py` (94KB, 2400 lines)
- Line 579-581: gc.collect() after every few tiles (memory pressure sign)
- Line 1640-1643: Edge tile handling
- **Recommendation:** Split into pipeline orchestration + cell-type modules

### 6.3 `train_nmj_classifier.py`
- Line 91-92: Hardcoded slide prefix makes code not reusable
- **Fix:** Extract from metadata or pass as parameter

---

## 7. MISSING INFRASTRUCTURE

### 7.1 Testing (MAJOR GAP)
**Current:** Only 3 manual test files for NMJ
**Missing:**
- Unit tests for shared modules
- Integration tests for pipelines
- Regression tests for coordinate conversions
**Fix:** Create `tests/` directory with pytest fixtures

### 7.2 Configuration Management
**Current:** Hardcoded defaults scattered across files
**Fix:** Use `shared/config.py` consistently in all pipelines

---

## Implementation Phases

### Phase 1 (Critical Bugs) ✅ COMPLETE
- [x] 2.1 Bounds checking ✅ Added validation in 7 files
- [x] 2.3 GPU memory leak fix ✅ Added periodic cache clearing
- [x] 2.4 Replace bare except blocks ✅ Fixed in 10 files with logging
- [x] 1.1, 1.2 Remove duplicate functions ✅ Removed from run_unified_FAST.py

### Phase 2 (Performance)
- [x] 4.1 Increase GPU batch size ✅ Batch ResNet extraction
- [x] 4.2 RAM loading for network mounts ✅ get_loader() singleton
- [x] 5.1 Multiprocessing for tile I/O ✅ Created TilePipeline producer-consumer in segmentation/io/tile_pipeline.py
- [x] 6.1 Refactor run_unified_FAST.py ✅ CellDetector + strategies created

### Phase 3 (Code Quality)
- [x] 3.1 Reduce nesting ✅ Created tile_processing.py with helper functions
- [x] 3.2 Split large functions ✅ process_tile_complete, enrich_detection_features, etc.
- [x] 3.3 Consistent coordinates ✅ Fixed SAM2 embedding bug in run_unified_FAST.py, audited all files
- [x] 1.3 Unify CZI patterns ✅ All use CZILoader
- [x] 1.4 Unify HTML patterns ✅ Created HTMLPageGenerator class
- [x] Wire main scripts to use new CellDetector ✅ run_segmentation.py uses CellDetector + strategies

### Phase 4 (Polish)
- [x] 7.1 Add test infrastructure ✅ Created tests/ with pytest fixtures
- [x] 7.2 Centralize configuration ✅ Added DETECTION_DEFAULTS with cell-type-specific params
- [x] 4.4, 4.5 Vectorization and HDF5 optimization ✅ rgb_to_hsv_vectorized, LZ4 compression

### Phase 5 (Comprehensive Review - Jan 2026)
Pipeline review identified and fixed additional issues:
- [x] Issue #1: SAM2 division by zero ✅ Added zero-check in all strategy files (nmj.py, mk.py, vessel.py, cell.py, template.py)
- [x] Issue #2: Centroid bounds validation ✅ Added validation before crop extraction in export_nmj_results_html.py
- [x] Issue #3: Class weights bug ✅ Fixed empty class check in train_nmj_classifier.py
- [x] Issue #5, #15: HDF5 OOM bug ✅ Removed full-file fallback in export_nmj_results_html.py
- [x] Issue #7: Consolidated extract_morphological_features ✅ Moved to shared feature_extraction.py
- [x] Issue #8: Feature dimension constants ✅ Added SAM2_EMBEDDING_DIM=256, RESNET50_FEATURE_DIM=2048
- [x] Issue #12: CZI path validation ✅ Added FileNotFoundError check in czi_loader.py
- [x] Issue #13: Thread safety ✅ Already addressed with _image_cache_lock
- [x] Created strategy template ✅ segmentation/detection/strategies/template.py
