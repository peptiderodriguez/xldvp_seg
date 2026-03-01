You are helping the user view pipeline results from the xldvp_seg image analysis system.

---

## What to do

**Step 1 — Find output directories with HTML.**
Search for `index.html` files in common output locations:
```bash
find /fs/pool/pool-mann-edwin/ -maxdepth 5 -name "index.html" -path "*/html/*" 2>/dev/null | head -20
```

Also check if the user specified a path in arguments.

**Step 2 — Ask which project to view** if multiple found. Show the list with modification times.

**Step 3 — Detect environment** (silently):
```bash
$MKSEG_PYTHON $REPO/scripts/system_info.py --json
```

**Step 4 — Launch the viewer.**

**On SLURM cluster:** Submit a lightweight job to serve HTML:
```bash
sbatch --job-name=html_viewer \
    --partition=p.hpcl8 \
    --cpus-per-task=1 \
    --mem=4G \
    --time=4:00:00 \
    --wrap="PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/serve_html.py <html_dir>"
```
Then tail the output log to find the Cloudflare tunnel URL.

**On local:** Run directly:
```bash
$MKSEG_PYTHON $REPO/serve_html.py <html_dir>
```

**Step 5 — Display the tunnel URL.** Tell the user: *"Open this URL in your browser to view and annotate detections."*

For beginners, explain the annotation interface:
- Each card shows a detection crop with overlay
- Green checkmark = real detection (positive)
- Red X = false positive (negative)
- Navigate pages with arrow buttons
- "Export" button saves annotations as JSON for classifier training
- Annotations persist in browser localStorage (per-experiment, won't collide between runs)

$ARGUMENTS
