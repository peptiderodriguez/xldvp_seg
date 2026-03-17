You are helping the user view pipeline results from the xldvp_seg image analysis system.

---

## What to do

**Step 1 — Find output directories with HTML.**
Search for `index.html` files in common output locations:
```bash
find "${REPO:-$(pwd)}/../" -maxdepth 5 -name "index.html" -path "*/html/*" 2>/dev/null | head -20
```

Also check if the user specified a path in arguments.

**Step 2 — Ask which project to view** if multiple found. Show the list with modification times.

**Step 3 — Detect environment** (silently):
```bash
$XLDVP_PYTHON $REPO/scripts/system_info.py --json
```

**Step 4 — Launch the viewer.**

**On SLURM cluster:** Submit a lightweight job to serve HTML:
```bash
sbatch --job-name=html_viewer \
    --partition=p.hpcl8 \
    --cpus-per-task=1 \
    --mem=4G \
    --time=4:00:00 \
    --wrap="PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/serve_html.py <html_dir>"
```
Then tail the output log to find the Cloudflare tunnel URL.

**On local:** Run directly:
```bash
$XLDVP_PYTHON $REPO/serve_html.py <html_dir>
```

**Step 5 — Display the tunnel URL.** Tell the user: *"Open this URL in your browser to view and annotate detections."*

**Step 5b — Check for SpatialData outputs.**
Look for `*_spatialdata.zarr` alongside the HTML:
```bash
ls -d <output_dir>/*_spatialdata.zarr 2>/dev/null
```
If found, tell the user: *"A SpatialData zarr store is also available for scverse analysis:"*
```python
import spatialdata as sd
sdata = sd.read_zarr("<path>")
adata = sdata["table"]
```
Also check for squidpy outputs (`*_squidpy/morans_i.csv`, `*_squidpy/*.png`) and show paths if they exist.

**Step 5c — One-command full visualization** (classify + spatial clustering + interactive viewer):
```bash
$XLDVP_PYTHON $REPO/scripts/view_slide.py \
    --detections <detections.json> \
    --czi-path <czi_path> \
    --output-dir <output>
```
This chains marker classification → spatial clustering → viewer generation → serves HTML with Cloudflare tunnel in one call. Useful after detection completes.

**Step 5d — Multi-slide spatial viewer** (KDE density contours, graph-pattern regions, DBSCAN clustering):
```bash
$XLDVP_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py \
    --input-dir <output> \
    --group-field <marker_class_field> \
    --output <output>/spatial_viewer.html
```
Client-side KDE bandwidth slider (50–1000 µm), multi-scale graph-pattern classification (linear/arc/ring/cluster), DBSCAN with convex hulls, ROI drawing.

For beginners, explain the annotation interface:
- Each card shows a detection crop with overlay
- Green checkmark = real detection (positive)
- Red X = false positive (negative)
- Navigate pages with arrow buttons
- "Export" button saves annotations as JSON for classifier training
- Annotations persist in browser localStorage (per-experiment, won't collide between runs)

---

## Adaptive Guidance

**After finding results:**
- If multiple HTML directories exist: help user pick the right one based on modification time and cell type. *"The most recent run is usually what you want. Here they are sorted by date."*
- If HTML is missing but detections exist: *"Detection JSON exists but HTML wasn't generated — this can happen if the run was interrupted. Regenerate with: `$XLDVP_PYTHON $REPO/scripts/regenerate_html.py --detections <json> --czi-path <czi>`"*

**When user starts annotating:**
- *"Aim for 200+ annotations, roughly balanced between positive and negative. The more you annotate, the better the classifier — but diminishing returns kick in after ~500."*
- *"Focus on borderline cases — the classifier already handles the obvious ones well. When you're unsure, that's exactly the kind of cell the classifier needs to learn about."*
- If score-filtered HTML (--score-threshold in the filename): *"This HTML is pre-filtered by classifier score. Good for reviewing the classifier's decisions, but for training a new classifier, annotate from unfiltered HTML."*

**After annotation export:**
- Check annotation count and balance: *"Exported N annotations (X positive, Y negative). That's a good training set."* or *"Only N annotations — consider annotating more for a robust classifier."*
- If highly imbalanced (>80% one class): *"Your annotations are skewed toward [positives/negatives]. The RF handles imbalance, but more [minority class] examples would help."*
- Suggest next step: *"Ready for classifier training? Use /classify to train an RF model on these annotations."*

**For spatial viewer:**
- If KDE contours show clear hotspots: *"The density map shows spatial clustering — these hotspots are regions of high detection density."*
- If graph patterns are enabled: *"Graph-pattern analysis classifies spatial arrangements into linear (along a boundary), arc/ring (around a structure), or cluster (focal group)."*

$ARGUMENTS
