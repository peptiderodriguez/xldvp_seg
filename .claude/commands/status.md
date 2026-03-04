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

1. **Find the log** — Use scontrol to get the exact log path, then tail it:
   ```bash
   scontrol show job <JOB_ID> | grep StdOut
   tail -50 <path_from_stdout>
   ```

2. **Check GPU usage** — On SLURM:
   ```bash
   srun --jobid=<JOB_ID> --overlap nvidia-smi
   ```
   On local: `nvidia-smi`

3. **Show detection progress** — Get the run dir from the log path, then count:
   ```bash
   # Run dir is the timestamped subdir containing tiles/ (get from scontrol StdOut path or ls)
   ls -d <run_dir>/tiles/tile_* 2>/dev/null | wc -l
   # Count detections in JSON (if dedup completed)
   $MKSEG_PYTHON -c "import json; d=json.load(open('<run_dir>/<celltype>_detections.json')); print(f'{len(d)} detections')"
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

---

## Adaptive Guidance

**During detection:**
- If tile progress is >50% and detection count is very low (<10 detections on a 1000+ tile slide): *"Very few detections so far. This could mean the detection channel is wrong, the threshold is too strict, or the tissue doesn't contain this cell type. Check the channel assignment with /czi-info."*
- If tile progress is slow (estimated >6 hours remaining): *"This is a large slide — that's normal. If you want faster turnaround on future runs, consider --tile-shard for multi-node parallelism."*
- If GPU utilization is low (<30%): *"GPUs are underutilized. This usually means I/O-bound or the tiles are mostly empty. Normal for sparse tissue."*

**During post-dedup:**
- Phase 1 (contour processing): *"Processing contours — this is fast, usually a few minutes."*
- Phase 2 (background estimation): *"Building KD-tree for local background — single-threaded, scales with detection count."*
- Phase 3 (feature re-extraction): *"Re-extracting intensity features on corrected data — parallelized, this is the longest post-dedup phase for large datasets."*

**On failure:**
- `CUDA out of memory`: *"GPU ran out of memory. Try --num-gpus 1 (less GPU memory fragmentation) or check if another job is using the same GPU. Resume with --resume <output_dir>."*
- `Killed` / `slurmstepd`: *"SLURM killed the job for exceeding memory. Request more RAM (--mem) or reduce --num-gpus. Detection checkpoints are saved — resume picks up where it stopped."*
- `FileNotFoundError` on CZI: *"CZI file not found — check if the network mount is accessible (ls the parent directory). Socket timeouts happen on this cluster."*
- `KeyError` on channel/feature: *"Missing feature or channel — likely a channel index mismatch. Run /czi-info to verify channel assignments."*
- Always offer the resume command: `--resume <output_dir>` picks up from the last checkpoint.

**On completion:**
- Report detection count and per-tile rate
- If HTML exists: *"Results are ready for annotation. Use /view-results to launch the viewer."*
- If SpatialData zarr exists: *"SpatialData was auto-exported — ready for scverse analysis."*

$ARGUMENTS
