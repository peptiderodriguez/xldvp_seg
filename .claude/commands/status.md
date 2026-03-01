You are monitoring pipeline jobs for the xldvp_seg image analysis system.

## What to do

**Step 1 — Check for running SLURM jobs:**
```bash
squeue -u $USER -o "%.10i %.30j %.10P %.8T %.12M %.6D %.20R"
```

If no SLURM is available (local workstation), check for running `run_segmentation.py` processes:
```bash
ps aux | grep run_segmentation | grep -v grep
```

**Step 2 — For each active job, offer to:**

1. **Tail the log** — Find the `.out` file:
   ```bash
   ls -t <output_dir>/slurm_*.out | head -1
   ```
   Then show the last 50 lines.

2. **Check GPU usage** — On SLURM:
   ```bash
   srun --jobid=<JOB_ID> --overlap nvidia-smi
   ```
   On local: `nvidia-smi`

3. **Show detection progress** — Count completed tiles and detections:
   ```bash
   # Count tile directories
   ls -d <output_dir>/*/tiles/tile_* 2>/dev/null | wc -l
   # Count detections in JSON (if exists)
   python -c "import json; d=json.load(open('<detections.json>')); print(f'{len(d)} detections')"
   ```

4. **Check HTML status** — Does `<output_dir>/html/index.html` exist?

5. **Check SpatialData status** — Does `<output_dir>/*_spatialdata.zarr` exist? If so, report the zarr store path and mention it's ready for scverse analysis.

**Step 3 — If a job failed:**

1. Find the `.err` file and show the last 100 lines
2. Check for common patterns:
   - `CUDA out of memory` → Suggest reducing `--num-gpus` or tile size
   - `FileNotFoundError` → Check CZI path
   - `Killed` / `slurmstepd: error` → OOM, request more memory
   - `RuntimeError: CUDA` → GPU issue, suggest reducing `--num-gpus` to 1
3. Show the fix and offer to resubmit with `--resume <output_dir>`

**Step 4 — Summary.**
Show a concise status table:
```
Job ID    Name              State     Time      Progress
12345     nmj_detection     RUNNING   1:23:45   142/350 tiles (41%)
12346     vessel_slide2     COMPLETED 0:45:12   Done — 1,247 detections
```

If multiple slides are running (array jobs), show per-slide status.

$ARGUMENTS
