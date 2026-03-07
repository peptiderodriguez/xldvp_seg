# MK Analysis Status — March 2026

## Current State

### Dataset
- **16 slides**: FGC1-4, FHU1-4, MGC1-4, MHU1-4
- **Design**: 2×2×2 factorial — Sex (F/M) × Treatment (GC/HU) × Bone (femur/humerus)
- **HU = hindlimb unloading** (simulated microgravity). GC = ground control.
- **Bone meaning**: femur = unloaded hindlimb; humerus = overloaded forelimb (compensatory)
- **Combined JSON**: `all_mks_with_rejected3.json` (3808 cells), `all_mks_with_rejected3_full.json` (with features)

### Classifier Issue (BLOCKING)
The current RF classifier (`mk_classifier_2026-02-11.pkl`) was trained on 13 slides only.
Three "rescued" slides (FGC2, FGC4, MHU4) were originally excluded as color outliers in
Reinhard normalization, then hand-annotated separately and merged via `merge_rejected_slides.py`.

**Problem**: The classifier scores are not comparable across slide groups:
- Original 13 slides: 62–79% of cells score ≥0.80
- Rescued 3 slides: 3–22% of cells score ≥0.80
- Applying a uniform score threshold produces incomparable densities

**Solution in progress**:
1. Extract SAM2 embeddings for original 13 slides (currently all zeros — pipeline version didn't save them)
2. Merge rescued slide annotations (617 positives + 116 negatives) into training set
3. Retrain classifier on all 16 slides (1125 pos + 1333 neg = 2458 samples)
4. Re-score all cells uniformly
5. Rerun ANOVA dashboard with consistent thresholding

### SAM2 Extraction (NEXT STEP)
Original 13 slides' tile features at `unified_2026-02-11_100pct_2gpu/` have `sam2_0..sam2_255`
keys but ALL ZEROS. Rescued slides (in `mk_clf084_dataset/{slide}/mk/tiles/`) have real SAM2 values.

**To run on cluster**:
```bash
cd ~/xldvp_seg && git pull
# Verify paths in scripts/slurm_extract_sam2.sh, then:
sbatch scripts/slurm_extract_sam2.sh
```

After extraction:
```bash
python scripts/extract_sam2_embeddings.py merge \
    --target /path/to/all_mks_with_rejected3_full.json \
    --embeddings sam2_embeddings_original13.json
```

### After SAM2 Extraction: Retrain + Re-score
TODO: Write `scripts/retrain_mk_classifier.py` that:
1. Loads original training data (`mk_training_data_2026-02-11.json`, 1725 samples)
2. Loads rescued slide features (positives from full JSON, negatives from tile features)
3. Merges into unified training set (2458 samples, all with real SAM2 embeddings)
4. Retrains RF with same hyperparams (1000 trees, balanced, max_features=0.1)
5. Re-scores all 3808 cells in `all_mks_with_rejected3_full.json`
6. Outputs updated JSON with new `mk_score` values

Then rerun: `python scripts/mk_interaction_analysis.py --score-threshold 0.80`

## Key Data Paths
| What | Where |
|------|-------|
| Dataset dir | `/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset/` |
| Combined detections (light) | `all_mks_with_rejected3.json` |
| Combined detections (full) | `all_mks_with_rejected3_full.json` |
| Tissue areas | `tissue_areas_by_bone.json` |
| Original training data | `mk_clf_export_2026-02-11/mk_training_data_2026-02-11.json` |
| Original classifier | `mk_clf_export_2026-02-11/mk_classifier_2026-02-11.pkl` |
| Rescued annotations | `mk_annotations_2026-03-06_rejected3_unnorm_100pct.json` |
| 100% run tiles | `unified_2026-02-11_100pct_2gpu/` |
| CZI files | `/Volumes/pool-mann-axioscan/01_Users/EdRo_axioscan/bonemarrow/2025_11_18/` |
| Bone regions | `/Volumes/pool-mann-edwin/bm_lmd_feb2026/bone_regions.json` |

## Analysis Scripts
| Script | Purpose |
|--------|---------|
| `scripts/mk_interaction_analysis.py` | ART ANOVA + 3×3 dashboard (MAD filtering, density only) |
| `scripts/mk_mechanism_figure.py` | Data-driven mechanism figure (3× exaggerated morphology) |
| `scripts/select_mks_for_lmd.py` | LMD proteomics replicate selection |
| `scripts/merge_rejected_slides.py` | Merge FGC2/FGC4/MHU4 into combined dataset |
| `scripts/extract_sam2_embeddings.py` | Re-extract SAM2 embeddings from CZI tiles |

## Key Biological Findings (preliminary — pending classifier fix)

### Two-pathway model
1. **Local mechanical pathway**: Controls MK density. Bone-specific, dose-dependent.
   Femur (complete unload) > humerus (partial overload).
2. **Systemic/humoral pathway**: Controls MK morphology. Parallel in both bones.
   Sex hormone-dependent (not hardwired sexually dimorphic).

### Sex × Treatment interactions (from 12-slide analysis, needs re-validation)
- Males: density drops in unloaded femur (−25%), morphology becomes more compact
- Females: density increases in unloaded femur (+27%?), morphology becomes more elongated
- **CAUTION**: Female femur density finding may be artifactual — driven by rescued slides
  (FGC2, FGC4) having very few cells due to classifier incompatibility. Must re-validate
  after classifier retraining.

### Comparison with David et al. 2006
- Both find males more affected by unloading than females (aligns)
- David found humerus unaffected — we find MK changes in humerus (extends their findings)
- David framed females as "passively protected" — our data suggests active compensation
- Our MK data adds cellular dimension David couldn't access (bone architecture only)
