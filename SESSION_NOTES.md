# MK+HSPC Segmentation & Annotation Pipeline - Session Notes
**Last Updated:** 2026-01-08

---

## SESSION: 2026-01-08 - Job Complete + Integrated HTML Export

### Results
**16-slide batch segmentation completed successfully!**

| Metric | Count |
|--------|-------|
| Total MKs | 39,362 |
| Total HSPCs | 23,296 |
| Slides Processed | 16/16 |
| Processing Time | ~20 hours |

### Performance After Optimizations
- **Calibration**: 1600 samples in 1:44 (parallelized with ThreadPoolExecutor)
- **Tissue checking**: 19,089 tiles at 10-70 tiles/sec (38 threads, ~20 min total)
- **GPU processing**: 1159 sampled tiles at ~60 sec/tile average

### Changes Made

#### 1. Parallel Tissue Checking (10-35x speedup)
Previous: Sequential, ~2 tiles/sec
Now: ThreadPoolExecutor with 38 threads (80% of CPU), 10-70 tiles/sec

```python
tissue_check_threads = max(1, int(os.cpu_count() * 0.8))
with ThreadPoolExecutor(max_workers=tissue_check_threads) as executor:
    futures = {executor.submit(check_tile_tissue, tile_args): tile_args for tile_args in all_tiles}
    for future in tqdm(as_completed(futures), total=len(all_tiles), desc="Checking tissue"):
        ...
```

#### 2. Parallel Calibration
Previous: Sequential, 30+ min stuck with no output
Now: ThreadPoolExecutor + tqdm progress bar, completes in ~2 min

#### 3. Integrated HTML Export (NEW)
**Problem:** Export script had to reload 320GB of CZI files after segmentation
**Solution:** Integrated export into main pipeline while slides are still in RAM

New functions in `run_unified_FAST.py`:
- `load_samples_from_ram()` - Extracts cell crops from in-memory slide arrays
- `export_html_from_ram()` - Generates HTML pages using RAM data
- `create_export_index()` - Creates index.html
- `generate_export_pages()` - Creates mk_page*.html and hspc_page*.html

New arguments:
- `--html-output-dir` - Directory for HTML export (default: output_dir/../docs)
- `--samples-per-page` - Cells per HTML page (default: 300)

Export now runs automatically before releasing slide data from RAM.

#### 4. HDF5 Plugin Fix
- Added `import hdf5plugin` to export_separate_mk_hspc.py
- Required for reading LZ4-compressed h5 files

### Files Modified
- `run_unified_FAST.py` - Added integrated HTML export, parallel tissue checking/calibration
- `export_separate_mk_hspc.py` - Added hdf5plugin import
- `run_local.sh` - Batch processing configuration

### HTML Export & Hosting
- **175 HTML pages generated** (132 MK + 42 HSPC + 1 index)
- **6.6 GB total** (~35MB per page with 300 samples each)
- **GitHub Pages failed** - 2GB pack size limit exceeded
- **Git LFS failed** - GitHub Pages doesn't serve LFS files
- **Solution:** Self-host locally with Python HTTP server

```bash
# Start annotation server
cd docs && python -m http.server 8080
# Open http://localhost:8080
```

### Annotation Workflow
1. Open http://localhost:8080 in browser
2. Click Yes/No/?  for each cell
3. Labels saved to browser localStorage
4. Click "Download Annotations JSON" when done
5. Use annotations for classifier training

---

## SESSION: 2026-01-07 - Local Machine Setup & Batch Processing

### Environment
- **Machine:** Local workstation (48 cores, 432GB RAM, RTX 4090 24GB VRAM)
- **CZI Source:** `/mnt/x/01_Users/EdRo_axioscan/bonemarrow/2025_11_18/`
- **Output:** `~/xldvp_seg_output/`

### Changes Made

#### 1. MK Filter Size Updated
- Changed from **100-2100 µm²** to **200-2000 µm²**
- Updated in:
  - `run_unified_FAST.py` (default arguments)
  - `run_local.sh` (configuration)
  - `export_separate_mk_hspc.py` (default arguments)

#### 2. Memory Leak Fixes
Previous runs crashed the machine. Found and fixed:
- Added `import gc` at module level
- Added `torch.cuda.empty_cache()` after MK processing
- Added explicit cleanup: `del new_mk, new_hspc` after HDF5 writes
- Changed GC from every 5 tiles to **every tile**
- Use generators for tile data to avoid memory spikes

#### 3. Batch Processing (Models Load ONCE)
- Added `--czi-paths` argument for multiple CZI files
- Rewrote `run_multi_slide_segmentation()` for unified sampling:
  - Loads ALL 16 slides into RAM (~350GB)
  - Identifies tissue tiles across ALL slides
  - Samples 10% from **combined pool** (truly representative)
  - Processes with models loaded ONCE

#### 4. Performance Tuning
- Changed `pred_iou_thresh` from 0.4 to **0.5** (faster processing)
- Set `NUM_WORKERS=1` (GPU memory is the bottleneck, not CPU/RAM)

#### 5. Pipelined Processing Architecture (NEW)
Created parallel CPU pre/post processing to maximize resource utilization:

```
Architecture:
  CPU ThreadPool (pre)  -->  Queue  -->  GPU Worker  -->  Queue  -->  CPU ThreadPool (save)
       23 threads                         1 process                       15 threads

Thread allocation (80% of CPU cores = 38 threads on 48-core machine):
  - Pre-process: 60% = 23 threads (extracting tiles from RAM, normalizing)
  - Post-process: 40% = 15 threads (saving HDF5, features JSON)
```

New functions in `run_unified_FAST.py`:
- `preprocess_tile_cpu()` - CPU-only tile extraction and normalization
- `save_tile_results()` - CPU-only HDF5 and features saving
- `process_tile_gpu_only()` - GPU-only SAM2/Cellpose/ResNet inference
- `run_pipelined_segmentation()` - Orchestrates the pipeline

### Current Job Status
**Batch processing 16 slides with 10% unified sampling**

```
Progress: 460/1159 tiles (40%)
Elapsed: ~10 hours
Remaining: ~14-15 hours
```

Resource usage during run:
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| CPU | ~3% | 48 cores | Low (expected - GPU bottleneck) |
| RAM | 355GB | 432GB | 82% (all slides loaded) |
| GPU VRAM | 4GB | 24GB | 17% |
| GPU Compute | 2% | 100% | Low (IO/preprocessing bound) |

### Files Modified
- `run_unified_FAST.py` - Major updates for batch processing, memory fixes, pipelined architecture
- `run_local.sh` - Local execution script with batch mode
- `export_separate_mk_hspc.py` - MK filter size update

---

## HISTORICAL: Cluster Work (2025-12-10)

### Directory Cleanup
- DELETED: `/viper/u2/edrod/MKsegmentation/` (253GB clutter)
- DELETED: `/viper/u2/edrod/MKsegmentation_clean/` (empty)
- PRESERVED: `/viper/ptmp2/edrod/MKsegmentation/` (official scripts)

### Job 4896272 FAILED - OUT OF MEMORY
- 16 workers × ~7.5GB each = ~120GB (exceeded 128GB limit)
- Processed 31% of first slide before OOM
- Solution: Reduce workers or increase memory

### Bug Fixes (Cluster)
1. **Cellpose Device Bug** - Passed string instead of `torch.device` object
2. **Export Script Filter Mismatch** - Hardcoded old filter values

---

## Workflow

### Current (Local Machine)
```bash
# 1. Run batch segmentation (MODE="batch" in run_local.sh)
./run_local.sh

# 2. Automatic HTML export after segmentation
# 3. Automatic git push to GitHub Pages
```

### Classifier Training (After Annotation)
```bash
# 1. Convert annotations
python convert_annotations_to_training.py \
    --annotations annotations/all_labels.json \
    --base-dir ~/xldvp_seg_output \
    --mk-min-area-um 200 --mk-max-area-um 2000 \
    --output annotations/training_data.json

# 2. Train classifiers
python train_separate_classifiers.py \
    --training-data annotations/training_data.json \
    --output-mk mk_classifier.pkl \
    --output-hspc hspc_classifier.pkl

# 3. Run 100% segmentation with classifiers
python run_unified_FAST.py \
    --czi-paths /path/to/*.czi \
    --output-dir ~/xldvp_seg_output \
    --mk-classifier mk_classifier.pkl \
    --hspc-classifier hspc_classifier.pkl \
    --sample-fraction 1.0
```

---

## 16 Slides
- FGC1-4 (Female GC)
- FHU1-4 (Female HU)
- MGC1-4 (Male GC)
- MHU1-4 (Male HU)

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--mk-min-area-um` | 200 | Minimum MK area in µm² |
| `--mk-max-area-um` | 2000 | Maximum MK area in µm² |
| `--sample-fraction` | 0.10 | 10% sampling for annotation |
| `--num-workers` | 1 | Limited by GPU memory |
| `--tile-size` | 3000 | 3000x3000 tiles |
| `pred_iou_thresh` | 0.5 | SAM2 confidence threshold |

---

## Technical Details

### µm to Pixel Conversion
- Pixel size: 0.1725 µm/px
- Conversion: 0.02975625 µm²/px²
- 200-2000 µm² = 6721-67212 px²

### Memory Layout
- Each slide: ~20-23 GB in RAM
- 16 slides total: ~350 GB
- Worker models: ~8-12 GB GPU VRAM per worker

### GitHub
- Repo: `peptiderodriguez/xldvp_seg`
- Live site: `https://peptiderodriguez.github.io/xldvp_seg/`

---

## Resource Optimization Summary

| Resource | Current Use | Max Available | Optimization |
|----------|-------------|---------------|--------------|
| RAM | 355GB (82%) | 432GB | All 16 slides loaded - MAXIMIZED |
| CPU | 3% | 48 cores | Pipelined mode uses 80% for pre/post processing |
| GPU VRAM | 4GB (17%) avg | 24GB | Single tile can spike to 100% - NO batching possible |
| GPU Compute | 2% | 100% | Waiting on I/O - pipelining helps |

### Implemented Optimizations
- Load all slides into RAM (avoids disk I/O during processing)
- Memory-mapped files in /dev/shm (fastest possible storage)
- Pipelined CPU pre/post processing (80% of cores)
- GC every tile + explicit cleanup (prevents memory leaks)
- Single GPU worker (avoids VRAM contention)

### Newly Implemented Optimizations
1. **LZ4 Compression** - HDF5 files now use LZ4 (3-5x faster than gzip)
   - Auto-fallback to gzip if `hdf5plugin` not installed
   - Install: `pip install hdf5plugin`
2. **Pinned Memory** - Pre-processing allocates pinned (page-locked) memory
   - Enables DMA transfer to GPU (doesn't block CPU)
   - Automatic when CUDA available

### NOT Viable
- **GPU Batch Inference** - Single tile can spike to 100% VRAM during SAM2 mask generation, no room for batching
- **True Double Buffering** - Can't preload to GPU while 100% VRAM in use

---

## Next Steps

1. **Current job completes** (~14-15 hours remaining)
2. **User annotates** via GitHub Pages
3. **Train classifiers** from annotations
4. **Validate classifiers** (min 75% accuracy, 70% recall/precision)
5. **Run 100% segmentation** with validated classifiers
6. **(Optional)** Enable pipelined mode for faster processing:
   - Add `--pipeline` flag to main script
   - Will use 80% CPU for pre/post processing
