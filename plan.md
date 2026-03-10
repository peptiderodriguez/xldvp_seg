# Project Plan and Task Tracking
Last Updated: 2026-03-10

## Active To-Do

### MK Hindlimb Unloading
- [ ] Place reference crosses on remaining 15 slides (only FGC3 has crosses)
- [ ] Generate LMD XML export from replicate assignments (need crosses first)

### Pipeline (Low Priority)
- [ ] Per-tile resume for shard mode
- [ ] `compute_normalization_params.py` — remove hardcoded CZI directory path
- [ ] `serve_html.py` — use handler `directory` param instead of `os.chdir()`
- [ ] Cross-tile vessel merge — generate HTML crops for merged vessels

## Recently Completed (Mar 2026)
- [x] Multi-plate 384-well `well_plate.py` module (consolidated from 3 duplicates)
- [x] LMD replicate builder with area normalization, multi-plate, 10% QC empties
- [x] SAM2 extraction for all 16 MK slides (SLURM array, outputs in `sam2_embeddings/`)
- [x] MK classifier retrained (morph-only, morph+SAM2, morph-nocolor variants)
- [x] MK mask refinement dry-run (`mask_refinement_comparison.html`)
- [x] MK ANOVA significance analysis (ART, BH correction, dual-pathway model)
- [x] Publication dashboard with shape panel (circularity, maturation arrest)
- [x] n45 liver 5-channel Cellpose pipeline (detected, classified, exported)
- [x] Vessel community analysis script
- [x] Spatial viewer updates (KDE contours, graph-pattern regions)
- [x] All docs updated (CLAUDE.md, /analyze, /lmd-export, memory files)

## Previous Tasks
- Code review rounds 1-14 (completed)
- Pipeline modularization (completed 2026-03-01)
- SpatialData integration (completed 2026-03-02)
- Direct-to-SHM loading + perf optimizations (completed 2026-03-02)
