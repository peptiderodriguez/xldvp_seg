# Project Plan and Task Tracking
Last Updated: 2026-02-24 09:23 UTC

## Current Objectives
- Review scripts/generate_expr_spatial_viewer.py for correctness, security, edge cases

## Completed Tasks
- [x] Read full script (763 lines) - 2026-02-24 09:23
- [x] Read project documentation (CLAUDE.md, MEMORY.md) - 2026-02-24 09:23
- [x] Read upstream data pipeline (assign_tissue_zones.py) for schema understanding - 2026-02-24 09:23
- [x] Read related scripts (spatial_frequency_analysis, generate_tissue_overlay, html_export) - 2026-02-24 09:23

## Pending Tasks
- [ ] Line-by-line review of expression group classification logic - high
- [ ] Review convex hull computation (scipy) - high
- [ ] Review HTML/JS correctness (canvas, events, memory) - high
- [ ] Assess XSS risks in JSON/data serialization - high
- [ ] Check edge cases (empty scenes, missing data, large clusters) - high
- [ ] Write comprehensive review report - high

## Notes and Observations
- Script generates self-contained HTML with canvas-based 2x4 scene grid
- Brain FISH: 4 markers (Slc17a7, Htr2a, Ntrk2, Gad1), 4 CZIs x 2 scenes = 8 panels
- Expression groups: 2^4=16 combos, minus quad-neg = 15, but script maps to 12
  (Slc17a7+/Gad1+ is 1 group regardless of Htr2a/Ntrk2)

## Previous Session (Resume Pipeline Review)
- Reviewed run_segmentation.py resume implementation
- Open issues: M1 double normalization, L1 no-load-to-ram guard, N1/N2 minor
