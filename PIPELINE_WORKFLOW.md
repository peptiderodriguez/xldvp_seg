# Cell Segmentation Pipeline Workflow

## Overview
This is the standard workflow for running cell segmentation with normalization on the Viper cluster.

## Directory Structure
```
/viper/ptmp2/edrod/
├── 2025_11_18/                          # Raw CZI files (16 slides)
├── xldvp_seg_fresh/                     # Code and scripts
│   ├── mkseg_rocm_env/                  # Python environment
│   ├── slurm/                           # Batch scripts
│   ├── generate_html_from_features.py   # HTML generation (NO CZI reload needed!)
│   └── regenerate_html.py               # HTML generation (requires CZI reload)
└── unified_10pct_mi300a_normalized/    # Output directory
    ├── html_combined/                   # HTML viewers
    │   ├── index.html                   # MAIN LANDING PAGE
    │   ├── mk/index.html                # MK viewer
    │   └── hspc/index.html              # HSPC viewer
    └── 2025_11_18_*/                    # Per-slide results
        ├── mk/tiles/*/                  # MK detections
        │   ├── features.json            # Contains crop_b64 and mask_b64!
        │   └── window.csv
        └── hspc/tiles/*/                # HSPC detections
```

## Standard Workflow

### 1. Run Segmentation Batches
Batches process 2 slides each in parallel with normalization:
- 8 batches total for 16 slides
- Use AMD MI300A GPUs (2 per job)
- Allocate 150-200GB RAM per job
- Output: `features.json` files with embedded image crops

**Important:** The `features.json` files contain `crop_b64` and `mask_b64` - pre-rendered image crops with masks overlaid. No need to reload CZI files for HTML generation!

### 2. Generate HTML Viewers
**Always use `generate_html_from_features.py`** (not `regenerate_html.py`):
```bash
cd /viper/ptmp2/edrod/xldvp_seg_fresh
source mkseg_rocm_env/bin/activate

# Generate MK HTML
python generate_html_from_features.py \
    --output-dir /viper/ptmp2/edrod/unified_10pct_mi300a_normalized \
    --cell-type mk \
    --experiment-name normalized_feb2026_mk \
    --sort-by area \
    --sort-order desc

# Generate HSPC HTML
python generate_html_from_features.py \
    --output-dir /viper/ptmp2/edrod/unified_10pct_mi300a_normalized \
    --cell-type hspc \
    --experiment-name normalized_feb2026_hspc \
    --sort-by area \
    --sort-order desc
```

### 3. Create Landing Page
Create `/viper/ptmp2/edrod/unified_10pct_mi300a_normalized/html_combined/index.html` that links to both MK and HSPC viewers.

This should be automatic from the script, but if not, manually create it with links to:
- `mk/index.html`
- `hspc/index.html`

## Key Scripts

### `generate_html_from_features.py`
- **Use this one!** Reads pre-saved crops from `features.json`
- No CZI loading required
- Fast (processes 16 slides in ~1 minute)
- Creates combined HTML in `html_combined/`

### `regenerate_html.py`
- **Don't use this for batch output!** Requires reloading entire CZI files
- Only use for single-slide processing or when crops aren't pre-saved
- Slow (requires loading multi-GB CZI files)

## Common Issues

### Batch OOM Errors
- Increase RAM allocation to 200GB
- Check if normalization is enabled (uses more memory)

### Missing HTML Landing Page
- The scripts create individual MK and HSPC viewers
- Manually create `html_combined/index.html` if needed
- Standard pattern: always create a single landing page linking to both

### Masks Not Found
- For batch output: masks are NOT saved as .h5 files
- Image crops with masks are embedded in `features.json` as base64
- Use `generate_html_from_features.py`, not `regenerate_html.py`

## Cluster Details

### Partition: `apu`
- AMD MI300A GPUs
- Check availability: `sinfo -p apu`
- Node status: `sinfo -n vipa[node_numbers]`

### Typical Job Stats
- Runtime: 40-60 minutes per batch
- Memory: 105-154GB peak usage
- GPUs: 2 per job
- CPUs: 96 per job

### Monitoring Jobs
```bash
squeue -u edrod              # Check job status
squeue -j <job_id>           # Check specific job
sacct -j <job_id>            # Job history
tail -f logs/<logfile>       # Monitor logs
```

## Output Statistics (Latest Run - Feb 2026)

### Normalized Dataset
- **Total slides:** 16
- **MK cells:** 24,445 (82 HTML pages)
- **HSPC cells:** 6,973 (24 HTML pages)
- **Landing page:** `/viper/ptmp2/edrod/unified_10pct_mi300a_normalized/html_combined/index.html`

## Normalization Notes

### What We Tried (Feb 2026)
**Percentile normalization (p1-p99):**
- Sampled 1M pixels per image
- Rescaled all images to same p1-p99 range
- **Result:** Variance got WORSE, not better
- Not suitable for downstream analysis

### Reinhard Normalization Implementation (Feb 2, 2026)
**Status: ✅ READY TO TEST**

**Completed:**
- ✅ Added Reinhard functions to `stain_normalization.py`
  - `compute_reinhard_params_from_samples()` - Lab mean/std computation
  - `apply_reinhard_normalization()` - Apply Lab normalization with tile-based processing
- ✅ Created `compute_reinhard_params_8slides.py` with tissue-aware sampling
  - Samples 500k pixels per slide from tissue tiles only (not random)
  - Uses same tissue detection logic as main pipeline
- ✅ Created batch scripts for 8-slide test:
  - `slurm/step1_compute_reinhard_params_8slides.sbatch` (200GB RAM)
  - `slurm/step2_launch_parallel_reinhard_8slides.sh` (4 jobs × 2 slides, 200GB each)
  - `slurm/step2_launch_parallel_unnormalized_8slides.sh` (baseline comparison)
- ✅ Updated `run_unified_FAST.py` to support `--normalization-method reinhard`
  - Branching logic for percentile/reinhard/none methods
  - Loads Reinhard params from JSON
  - Applies tile-based normalization
- ✅ Created `validate_normalization.py` validation script
  - Compares unnormalized vs Reinhard vs percentile
  - Computes variance reduction metrics
  - Generates comparison HTML report

**Critical Fixes Applied (Feb 2, 2026):**
- 🔧 **OOM Fix:** Rewrote `apply_reinhard_normalization()` to use tile-based processing
  - Peak memory reduced from ~510GB to ~50GB per slide
  - Processes image in 10k×10k tiles to avoid memory explosion
- 🔧 **Crash Fix:** Added empty samples check in `compute_reinhard_params_8slides.py`
- 🔧 **Memory Leak Fix:** Added explicit cleanup after normalization in `run_unified_FAST.py`
- 🔧 **Pixel-Level Masking:** Added tissue pixel masking to avoid edge artifacts
  - Statistics computed from tissue pixels only (not background)
  - Normalization applied only to tissue pixels (background unchanged)
  - Prevents discoloration of background at tissue edges
  - Uses local variance (7×7 blocks) for pixel-level tissue detection

**Test Plan (8 slides):**
1. Compute Reinhard params (FGC1,3 + FHU2,4 + MGC1,3 + MHU2,4)
2. Run segmentation with Reinhard normalization (4 jobs)
3. Run segmentation without normalization (4 jobs, baseline)
4. Validate: >30% variance reduction = success

**Why Reinhard:**
- Memory-efficient (only computes mean/std in Lab color space)
- Won't cause OOM errors with 16 slides in batch processing
- Established method for color normalization
- More likely to help with variance than simple percentile rescaling

**Alternative approaches if Reinhard doesn't work:**
1. Macenko stain normalization (standard for H&E, but more memory-intensive)
2. CLAHE preprocessing
3. Vahadane stain normalization
4. Process without normalization (baseline comparison)

## Quick Reference

**View results:**
```
/viper/ptmp2/edrod/unified_10pct_mi300a_normalized/html_combined/index.html
```

**Activate environment:**
```bash
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
```

**HTML generation (standard):**
```bash
python generate_html_from_features.py --output-dir <output> --cell-type {mk|hspc} --experiment-name <name>
```

**Remember:**
- Use `generate_html_from_features.py`, NOT `regenerate_html.py`
- Always create a single landing page at `html_combined/index.html`
- Crops are already in `features.json` - no CZI reload needed!
