# Mask Cleanup Feature Extraction Fix

## Problem
Previously, when `--cleanup-masks` was used, the mask cleanup (largest component + hole filling) happened AFTER feature extraction. This caused a mismatch:

- ✗ Morphological features (circularity, solidity, elongation, etc.) computed from **ORIGINAL masks**
- ✗ SAM2 embeddings extracted from **ORIGINAL mask centroids**
- ✗ ResNet features extracted from **ORIGINAL mask crops**
- ✓ Only area and center were updated after cleanup
- ✓ Saved crops showed cleaned masks, but features didn't match

## Solution
Moved mask cleanup to happen **BEFORE** all feature extraction in `UnifiedSegmenter.process_tile()`:

### Order of Operations (FIXED):
```
1. SAM2/Cellpose detect masks
2. Apply cleanup (if --cleanup-masks enabled):
   - Keep largest connected component
   - Fill internal holes
   - Update mask in label array
   - Recompute centroid
3. Extract ALL features from cleaned masks:
   - Morphological features (22 features)
   - SAM2 embeddings (256 features) - extracted at cleaned centroid
   - ResNet features (2048 features) - extracted from cleaned crop
4. Generate visualization crops
5. Save to disk
```

### Files Modified:

**1. `segmentation/utils/mask_cleanup.py`:**
- Enhanced `apply_cleanup_to_detection()` to recompute ALL morphological features
- Added `image` parameter for full feature recomputation
- Now recomputes: area, perimeter, circularity, solidity, aspect_ratio, elongation, eccentricity, extent, RGB means/stds, HSV features, intensity variance, etc.

**2. `run_unified_FAST.py`:**
- Added cleanup parameters to `UnifiedSegmenter.process_tile()`: `cleanup_masks`, `fill_holes`, `pixel_size_um`
- Added cleanup logic in MK processing loop (lines ~862-870)
- Added cleanup logic in HSPC processing loop (lines ~1048-1056)
- Updated all 4 calls to `process_tile()` to pass cleanup config
- Cleanup now happens BEFORE `extract_morphological_features()`, SAM2 embedding extraction, and ResNet feature extraction

## Result
Now **ALL 2,326 features** correspond to the cleaned masks:
- ✓ 22 morphological features
- ✓ 256 SAM2 embeddings (from cleaned centroid)
- ✓ 2,048 ResNet features (from cleaned crop)

## To Re-run with Fixed Pipeline

The existing results need to be regenerated to have properly aligned features:

```bash
# Delete old results
rm -rf /viper/ptmp2/edrod/unified_10pct_mi300a

# Re-run with cleanup (features will now be correct)
sbatch slurm/run_mi300a_sequential_all16.sbatch
```

## Verification

To verify features match cleaned masks:
1. Check that circularity/solidity values change when cleanup is enabled vs disabled
2. Confirm that features.json area matches the cleaned mask area
3. Verify that crop_b64 images show masks that match the computed features

## Notes

- The old HTML viewers (in `html_combined/`) were generated from the OLD data with mismatched features
- After re-running, regenerate HTML: `python generate_html_from_features.py --cell-type mk ...`
- Mask cleanup parameters: `keep_largest=True`, `fill_internal_holes=True`, `max_hole_area_fraction=0.5`
