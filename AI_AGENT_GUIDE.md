# AI Agent Guide - xldvp_seg Pipeline

This document provides comprehensive technical details for AI agents working with this codebase. It explains the architecture, data flow, integration points, and common tasks.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Script Integration Matrix](#script-integration-matrix)
4. [Key Implementation Details](#key-implementation-details)
5. [Common Tasks & Solutions](#common-tasks--solutions)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Architecture

### Overview

The pipeline consists of three independent but integrated subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│ SUBSYSTEM 1: SEGMENTATION                                   │
│  - Input: CZI microscopy files                              │
│  - Processing: SAM2 (MK), Cellpose+SAM2 (HSPC)             │
│  - Output: HDF5 masks + JSON features                      │
│  - Script: run_unified_FAST.py                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SUBSYSTEM 2: ANNOTATION & TRAINING                          │
│  - Input: Segmentation output                               │
│  - Processing: HTML export → manual annotation → training   │
│  - Output: Random Forest classifiers (.pkl files)           │
│  - Scripts: export_separate_mk_hspc.py,                     │
│             convert_annotations_to_training.py,             │
│             train_separate_classifiers.py,                  │
│             validate_classifier.py                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SUBSYSTEM 3: CLASSIFIER APPLICATION                         │
│  - Input: Segmentation + trained classifiers               │
│  - Processing: Real-time filtering during segmentation      │
│  - Output: Filtered cell detections                        │
│  - Script: run_unified_FAST.py (with --mk-classifier)      │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Abstraction

The code supports two deployment environments:

**Lab Machine (NVIDIA):**
- GPU: 1× RTX 3090 (24GB VRAM, CUDA)
- CPU: 24 cores
- RAM: 512GB
- Config: `--num-workers 4-6`, `--tile-size 4096`
- Script: `run_all_slides_local.sh`

**Cluster (AMD):**
- GPU: 2× MI250X (40GB VRAM each, ROCm)
- CPU: 32 cores
- RAM: 128-192GB
- Config: `--num-workers 10-16`, `--tile-size 4096`
- Script: `run_unified_10pct.sh` (SLURM batch)

### Critical Parameters

All scripts must use **consistent parameters** for filter thresholds:

```python
MK_MIN_AREA_UM = 100   # µm² minimum MK area
MK_MAX_AREA_UM = 2100  # µm² maximum MK area
PIXEL_SIZE_UM = 0.1725 # µm per pixel
UM_TO_PX_FACTOR = PIXEL_SIZE_UM ** 2  # 0.02975625
```

**Conversion:**
- 100 µm² = 3,360 px²
- 2100 µm² = 70,573 px²

---

## Data Flow

### Phase 1: Initial Segmentation (10% Sampling)

```
run_unified_FAST.py
  ├─ Input: slide.czi
  ├─ Parameters:
  │   ├─ --sample-fraction 0.10  (10% of tiles)
  │   ├─ --mk-min-area-um 100
  │   └─ --mk-max-area-um 2100
  └─ Output:
      ├─ output_10pct/
      │   ├─ mk/tiles/0/
      │   │   ├─ features.json     [2326 features per cell]
      │   │   ├─ segmentation.h5   [HDF5 masks]
      │   │   ├─ classes.csv       [Cell IDs]
      │   │   └─ window.csv        [Tile coordinates]
      │   └─ hspc/tiles/...
```

**Expected cell counts (16 slides @ 10%):**
- MK: ~4,500 cells
- HSPC: ~110,000 cells

### Phase 2: HTML Export & Annotation

```
export_separate_mk_hspc.py
  ├─ Input: output_10pct/
  ├─ Parameters:
  │   ├─ --mk-min-area-um 100     [MUST MATCH segmentation!]
  │   └─ --mk-max-area-um 2100
  ├─ Processing:
  │   ├─ Recreates cell list in same order as segmentation
  │   ├─ Applies SAME µm-based filter
  │   ├─ Generates PNG thumbnails (300×300 px)
  │   └─ Creates paginated HTML (50 cells per page)
  └─ Output:
      ├─ html_output/
      │   ├─ mk_samples/
      │   │   ├─ page_0.html
      │   │   └─ ...
      │   └─ hspc_samples/
      └─ metadata.json
```

**User annotates via GitHub Pages:**
- Reviews each cell image
- Labels as "Good" (positive, class=1) or "Bad" (negative, class=0)
- Downloads `all_labels_combined.json`

### Phase 3: Training Data Conversion

```
convert_annotations_to_training.py
  ├─ Input:
  │   ├─ all_labels_combined.json  (annotations)
  │   └─ output_10pct/             (segmentation features)
  ├─ Parameters:
  │   ├─ --mk-min-area-um 100      [MUST MATCH previous steps!]
  │   └─ --mk-max-area-um 2100
  ├─ Processing:
  │   ├─ Maps annotation IDs to actual cells
  │   ├─ Applies SAME µm-based filter
  │   └─ Extracts features for labeled cells
  └─ Output:
      └─ training_data.json
          ├─ mk_samples: [{features, label}, ...]
          └─ hspc_samples: [{features, label}, ...]
```

**Critical invariant:** The cell at index `i` in HTML export must match cell at index `i` in segmentation output after filtering.

### Phase 4: Classifier Training

```
train_separate_classifiers.py
  ├─ Input: training_data.json
  ├─ Processing:
  │   ├─ Separate MK and HSPC data
  │   ├─ Calculate balanced class weights
  │   ├─ Train Random Forest (200 trees, max_depth=15)
  │   └─ 5-fold stratified cross-validation
  └─ Output:
      ├─ mk_classifier.pkl
      │   ├─ classifier (RandomForestClassifier)
      │   ├─ X_train, y_train
      │   ├─ feature_names
      │   └─ metrics
      └─ hspc_classifier.pkl
```

### Phase 5: Validation

```
validate_classifier.py
  ├─ Input:
  │   ├─ mk_classifier.pkl
  │   └─ hspc_classifier.pkl
  ├─ Thresholds:
  │   ├─ min_accuracy: 0.75
  │   ├─ min_recall: 0.70
  │   └─ min_precision: 0.70
  └─ Decision:
      ├─ ALL PASS → proceed to 100% processing
      └─ ANY FAIL → collect more annotations, retrain
```

### Phase 6: Full Processing (100%)

```
run_unified_FAST.py
  ├─ Input: slide.czi
  ├─ Parameters:
  │   ├─ --sample-fraction 1.0     (ALL tiles)
  │   ├─ --mk-classifier mk_classifier.pkl
  │   ├─ --hspc-classifier hspc_classifier.pkl
  │   ├─ --mk-min-area-um 100
  │   └─ --mk-max-area-um 2100
  ├─ Processing:
  │   ├─ Segment cell
  │   ├─ Extract features
  │   ├─ Apply classifier
  │   └─ If classifier rejects: REMOVE from mask
  └─ Output:
      └─ output_100pct/ (only positive cells)
```

**Expected cell counts (16 slides @ 100%):**
- MK: ~45,000 cells (filtered by classifier)
- HSPC: ~1,100,000 cells (filtered by classifier)

---

## Script Integration Matrix

| Script | Depends On | Produces | Used By |
|--------|-----------|----------|---------|
| `run_unified_FAST.py` | CZI files, optional classifiers | Segmentation output | All downstream scripts |
| `export_separate_mk_hspc.py` | Segmentation output | HTML for annotation | User (manual annotation) |
| `convert_annotations_to_training.py` | Annotations + segmentation | Training data JSON | `train_separate_classifiers.py` |
| `train_separate_classifiers.py` | Training data JSON | Classifier PKL files | `validate_classifier.py`, `run_unified_FAST.py` |
| `validate_classifier.py` | Classifier PKL files | Validation report | User decision (proceed or retrain) |
| `run_all_slides_local.sh` | All 16 CZI files | Batch segmentation | N/A (entry point) |
| `run_unified_10pct.sh` | CZI files (SLURM) | Batch segmentation (cluster) | N/A (entry point) |

### Parameter Consistency Check

**CRITICAL:** These parameters must be IDENTICAL across all scripts in a single workflow:

```bash
# Phase 1: Segmentation
python run_unified_FAST.py \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --sample-fraction 0.10

# Phase 2: Export
python export_separate_mk_hspc.py \
    --mk-min-area-um 100 \    # ← MUST MATCH
    --mk-max-area-um 2100      # ← MUST MATCH

# Phase 3: Convert
python convert_annotations_to_training.py \
    --mk-min-area-um 100 \    # ← MUST MATCH
    --mk-max-area-um 2100      # ← MUST MATCH

# Phase 6: Full processing
python run_unified_FAST.py \
    --mk-min-area-um 100 \    # ← MUST MATCH
    --mk-max-area-um 2100 \    # ← MUST MATCH
    --sample-fraction 1.0
```

**Why:** Mismatched filters cause annotation IDs to be offset from actual cells, resulting in classifier training on wrong cells.

---

## Key Implementation Details

### 1. Segmentation Pipeline (`run_unified_FAST.py`)

**Entry Point:**
```python
def run_unified_segmentation(
    czi_path,
    output_dir,
    tile_size=4096,
    num_workers=4,
    mk_classifier=None,
    hspc_classifier=None,
    mk_min_area_um=100,
    mk_max_area_um=2100,
    sample_fraction=0.1
):
```

**Multiprocessing Architecture:**
```
Main Process
  ├─ Memory-map CZI file (shared across workers)
  ├─ Tissue detection (calibration phase)
  ├─ Create worker pool
  └─ Spawn workers
      └─ Worker (1 per GPU core)
          ├─ Load models (SAM2, Cellpose, ResNet)
          ├─ Load classifiers (if provided)
          └─ Process tiles
              ├─ Segment MK (SAM2 automatic)
              ├─ Segment HSPC (Cellpose + SAM2 refinement)
              ├─ Extract features (morphology + embeddings)
              ├─ Apply classifier (if provided)
              └─ Save results
```

**Memory Management:**
- Each worker: ~4-6GB GPU, ~10-15GB RAM
- CZI file: memory-mapped (shared, not duplicated)
- Models: loaded once per worker
- Garbage collection: every 5 tiles

**Classifier Integration (lines 632, 738):**
```python
# Line 632 - MK filtering
is_positive, confidence = self.apply_classifier(morph, 'mk')
if not is_positive:
    # Remove from mask if classifier rejects
    mk_masks[mk_masks == mk_id] = 0
    continue

# Line 738 - HSPC filtering
is_positive, confidence = self.apply_classifier(morph, 'hspc')
if not is_positive:
    hspc_masks[hspc_masks == hspc_id] = 0
    continue
```

### 2. Export System (`export_separate_mk_hspc.py`)

**Critical Function:**
```python
def recreate_cell_list(base_dir, mk_min_area_um, mk_max_area_um):
    """
    MUST recreate cell list in EXACT same order as segmentation.
    Uses SAME µm-based filter to ensure ID consistency.
    """
    # Convert µm² to px²
    um_to_px_factor = 0.02975625
    mk_min_px = int(mk_min_area_um / um_to_px_factor)
    mk_max_px = int(mk_max_area_um / um_to_px_factor)

    # Filter cells (MUST match segmentation filter)
    filtered_cells = [
        c for c in all_cells
        if mk_min_px <= c['area_px'] <= mk_max_px
    ]

    return filtered_cells
```

**Output Format:**
- HTML pages: 50 cells per page for manageable annotation
- Cell thumbnails: 300×300 px PNG images
- LocalStorage: Saves annotations in browser

### 3. Training System (`train_separate_classifiers.py`)

**Random Forest Configuration:**
```python
clf = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=15,           # Limit tree depth (prevent overfitting)
    min_samples_split=5,    # Minimum samples to split node
    min_samples_leaf=2,     # Minimum samples per leaf
    class_weight='balanced', # Handle class imbalance
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores
)
```

**Class Weights:**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
# Example output: {0: 1.2, 1: 0.8} if 40% positive, 60% negative
```

**Cross-Validation:**
- 5-fold stratified CV
- Ensures each fold has similar class distribution
- Reports per-fold accuracy, precision, recall

### 4. Validation System (`validate_classifier.py`)

**Quality Thresholds:**
```python
THRESHOLDS = {
    'min_accuracy': 0.75,   # 75% overall correctness
    'min_recall': 0.70,     # 70% sensitivity (don't miss positives)
    'min_precision': 0.70   # 70% specificity (avoid false positives)
}
```

**Decision Logic:**
```python
if accuracy >= 0.75 and recall >= 0.70 and precision >= 0.70:
    return "PASS - Ready for deployment"
else:
    return "FAIL - Collect more annotations and retrain"
```

---

## Common Tasks & Solutions

### Task 1: Starting Fresh on Lab Machine

**Step 1: Clone and setup**
```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
conda create -n mkseg python=3.10
conda activate mkseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Step 2: Update paths in batch script**
```bash
nano run_all_slides_local.sh
# Edit: CZI_BASE="/your/actual/path/to/czi/files"
# Edit: OUTPUT_BASE="$HOME/xldvp_seg_output"
```

**Step 3: Run 10% segmentation**
```bash
# First, test single slide
python run_unified_FAST.py \
    --czi-path /path/to/test_slide.czi \
    --output-dir ./test_output \
    --tile-size 4096 \
    --num-workers 4 \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --sample-fraction 0.10

# If successful, run all slides
bash run_all_slides_local.sh
```

### Task 2: Debugging Annotation ID Mismatch

**Problem:** User reports "the cell I annotated as Good doesn't match what was trained"

**Diagnosis:**
```bash
# Check filter consistency
grep -r "mk-min-area-um" *.sh *.py
grep -r "mk_min_area" *.py

# All should show SAME values: 100-2100 µm²
```

**Solution:** Update all scripts to use identical parameters

### Task 3: Handling Low Classifier Accuracy

**Scenario:** Validation reports 65% accuracy (below 75% threshold)

**Analysis:**
```python
# Check training set size
with open('training_data.json') as f:
    data = json.load(f)
    mk_count = len(data['mk_samples'])
    hspc_count = len(data['hspc_samples'])

print(f"MK samples: {mk_count}")    # Target: >500
print(f"HSPC samples: {hspc_count}") # Target: >1000
```

**Solution:**
1. If <500 MK samples: Increase `--sample-fraction` from 0.10 to 0.15 or 0.20
2. Re-run segmentation, export, annotate, train
3. Validate again

### Task 4: Optimizing Worker Count for Lab Machine

**Measurement:**
```bash
# Start with 4 workers, monitor GPU
watch -n 1 nvidia-smi

# Look for:
#  - GPU utilization ~80-95% (good)
#  - GPU memory <22GB (safe margin)
```

**Tuning:**
```bash
# If GPU utilization <60%: increase workers
NUM_WORKERS=5  # or 6

# If GPU OOM errors: decrease workers
NUM_WORKERS=3  # or 2

# If system RAM OOM: decrease workers or tile size
TILE_SIZE=3000
```

---

## Troubleshooting Guide

### Error: "Annotation IDs don't match cells"

**Symptom:** Classifier trains on wrong cells

**Root Cause:** Filter parameters differ between scripts

**Fix:**
```bash
# Verify consistency
export_separate_mk_hspc.py: --mk-min-area-um 100 --mk-max-area-um 2100
convert_annotations_to_training.py: --mk-min-area-um 100 --mk-max-area-um 2100
run_unified_FAST.py: --mk-min-area-um 100 --mk-max-area-um 2100
```

### Error: "CUDA out of memory"

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions (in order):**
1. Reduce workers: `--num-workers 3` → `2` → `1`
2. Reduce tile size: `--tile-size 4096` → `3000`
3. Check for other GPU processes: `nvidia-smi` → kill others

### Error: "Killed (OOM)" (system RAM)

**Symptom:** Process killed, no Python traceback

**Root Cause:** Exceeded system RAM

**Solution:**
```bash
# Lab machine (512GB RAM):
# Each worker: ~10-15GB RAM
# Safe: 4-6 workers × 15GB = 60-90GB

# Reduce workers:
--num-workers 4  # or lower
```

### Error: "Classifier validation FAIL"

**Symptom:** Accuracy <75%, recall <70%, or precision <70%

**Solutions:**
1. **More data:** Increase sample fraction 0.10 → 0.15 → 0.20
2. **Check class balance:** Aim for 40-60% positive annotations
3. **Review annotations:** Look for systematic errors
4. **Try morph-only:** Use `--morph-only` flag (22 features instead of 2326)

### Error: "No module named 'segment_anything_2'"

**Symptom:** ImportError during startup

**Fix:**
```bash
# Check SAM2 installation
pip list | grep segment

# If missing:
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

---

## Important Implementation Notes for AI Agents

### When Modifying This Code

1. **NEVER change filter parameters** without updating ALL scripts
2. **ALWAYS verify µm² to px² conversion** uses `PIXEL_SIZE_UM = 0.1725`
3. **TEST on single slide** before batch processing all 16 slides
4. **PRESERVE multiprocessing** architecture (don't make synchronous)
5. **MAINTAIN classifier optional** (code must work with/without classifiers)

### When Debugging

1. **Check parameter consistency** across all scripts first
2. **Monitor resources** (nvidia-smi, htop) before changing code
3. **Validate on small dataset** (1-2 slides) before full run
4. **Compare cell counts** between segmentation and export
5. **Verify annotation mapping** with sample IDs

### When Adding Features

1. **Keep classifier interface** consistent (`apply_classifier()` method)
2. **Maintain backwards compatibility** (old outputs should still work)
3. **Update BOTH batch scripts** (local + cluster versions)
4. **Document parameter changes** in README and this guide
5. **Test on both** lab machine (CUDA) and cluster (ROCm) if possible

---

## Quick Reference

### File Extensions
- `.czi` - Zeiss microscopy image (input)
- `.h5` / `.hdf5` - Segmentation masks (output)
- `.json` - Features, metadata, training data
- `.pkl` - Pickled Random Forest classifiers
- `.csv` - Cell IDs, tile coordinates

### Key Directories
- `output_10pct/` - 10% segmentation results
- `output_100pct/` - Full segmentation results (post-classifier)
- `html_output/` - Annotation interface
- `checkpoints/` - Model weights (SAM2, etc.)

### Memory Estimates (Lab Machine)
- 1 worker: ~6GB GPU, ~15GB RAM
- 4 workers: ~24GB GPU (FULL), ~60GB RAM
- 6 workers: ~36GB GPU (OOM RISK), ~90GB RAM

### Typical Runtimes (Lab Machine, RTX 3090)
- Single slide @ 10%: ~15-20 minutes
- Single slide @ 100%: ~2-4 hours
- All 16 slides @ 10%: ~4-6 hours
- All 16 slides @ 100%: ~32-64 hours

---

**Last Updated:** 2025-12-16
**Maintainer:** @peptiderodriguez
**AI Agent Version:** v1.0
