#!/usr/bin/env python3
"""
Detect system environment, hardware resources, and recommend pipeline parameters.

Outputs JSON (--json) or human-readable text.  No heavy imports — runs in <1s.

Usage:
    python scripts/system_info.py
    python scripts/system_info.py --json
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MKSEG_PYTHON = Path(
    os.environ.get(
        "MKSEG_PYTHON",
        "/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python",
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, timeout=10):
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _parse_meminfo():
    """Return (total_gb, free_gb) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    info[parts[0].rstrip(":")] = int(parts[1]) / (1024 * 1024)
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", total)
        return round(total, 1), round(avail, 1)
    except OSError:
        return 0, 0


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_slurm():
    """Detect SLURM cluster info.  Returns dict or None."""
    if shutil.which("sinfo") is None:
        return None

    slurm = {"partitions": []}

    # Two sinfo queries: one for specs, one for busyness
    # Query 1: PARTITION NODELIST CPUS MEMORY GRES
    raw = _run(["sinfo", "-o", "%P %N %c %m %G", "--noheader"])
    if not raw:
        return slurm

    partitions = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        partition = parts[0].rstrip("*")
        nodelist = parts[1]
        cpus_str = parts[2]
        mem_str = parts[3]  # in MB
        gres = parts[4]  # e.g. "gpu:rtx5000:2" or "(null)"

        try:
            cpus = int(cpus_str)
            mem_mb = int(mem_str)
        except ValueError:
            continue

        if gres == "(null)":
            gres = ""

        # Strip trailing "(S:0)" from GRES like "gpu:l40s:4(S:0)"
        if "(" in gres:
            gres = gres[:gres.index("(")]

        # Count nodes from nodelist (e.g. "hpcl[8005-8059]")
        node_count = 1
        range_match = re.search(r'\[(\d+)-(\d+)\]', nodelist)
        if range_match:
            node_count = int(range_match.group(2)) - int(range_match.group(1)) + 1

        if partition not in partitions:
            partitions[partition] = {
                "name": partition,
                "node_count": node_count,
                "cpus_per_node": cpus,
                "mem_gb_per_node": round(mem_mb / 1024),
                "gres": gres,
                "gpus_per_node": 0,
                "nodes_allocated": 0,
                "nodes_idle": 0,
            }
        else:
            partitions[partition]["cpus_per_node"] = max(partitions[partition]["cpus_per_node"], cpus)
            partitions[partition]["mem_gb_per_node"] = max(partitions[partition]["mem_gb_per_node"], round(mem_mb / 1024))
            partitions[partition]["node_count"] = max(partitions[partition]["node_count"], node_count)
            if gres and not partitions[partition]["gres"]:
                partitions[partition]["gres"] = gres

    # Query 2: Node allocation counts (%A = allocated/idle)
    raw2 = _run(["sinfo", "-o", "%P %A", "--noheader"])
    if raw2:
        for line in raw2.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            pname = parts[0].rstrip("*")
            # %A format: "allocated/idle"
            alloc_idle = parts[1].split("/")
            if pname in partitions and len(alloc_idle) == 2:
                try:
                    partitions[pname]["nodes_allocated"] = int(alloc_idle[0])
                    partitions[pname]["nodes_idle"] = int(alloc_idle[1])
                except ValueError:
                    pass

    # Parse GPU counts from GRES
    for p in partitions.values():
        if p["gres"]:
            # e.g. "gpu:rtx5000:2" or "gpu:l40s:4"
            gparts = p["gres"].split(":")
            if len(gparts) >= 3:
                try:
                    p["gpus_per_node"] = int(gparts[-1])
                    p["gpu_type"] = gparts[1]
                except ValueError:
                    pass
            elif len(gparts) == 2:
                try:
                    p["gpus_per_node"] = int(gparts[-1])
                except ValueError:
                    pass

    slurm["partitions"] = sorted(partitions.values(), key=lambda x: -x["gpus_per_node"])

    # User's running jobs
    user = os.environ.get("USER", "")
    if user:
        raw = _run(["squeue", "-u", user, "--noheader", "-o", "%i %j %P %T %M %N"])
        if raw:
            slurm["running_jobs"] = []
            for line in raw.splitlines():
                parts = line.split(None, 5)
                if len(parts) >= 4:
                    slurm["running_jobs"].append({
                        "job_id": parts[0],
                        "name": parts[1],
                        "partition": parts[2],
                        "state": parts[3],
                        "time": parts[4] if len(parts) > 4 else "",
                        "node": parts[5] if len(parts) > 5 else "",
                    })

    return slurm


def detect_local_gpus():
    """Detect local NVIDIA GPUs via nvidia-smi."""
    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not raw:
        return []
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append({"name": parts[0], "memory_mb": int(float(parts[1]))})
            except ValueError:
                gpus.append({"name": parts[0], "memory_mb": 0})
    return gpus


def detect_disk_space(paths=None):
    """Check disk space on relevant mount points."""
    if paths is None:
        paths = ["/fs/pool/pool-mann-edwin", "/tmp"]
    result = {}
    for p in paths:
        try:
            st = os.statvfs(p)
            total_gb = round(st.f_blocks * st.f_frsize / (1024 ** 3), 1)
            free_gb = round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)
            result[p] = {"total_gb": total_gb, "free_gb": free_gb}
        except OSError:
            pass
    return result


def detect_conda_env():
    """Check if mkseg conda env exists."""
    return MKSEG_PYTHON.exists()


def detect_git_info():
    """Get repo git status."""
    info = {}
    head = _run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"])
    if head:
        parts = head.split(" ", 1)
        info["head_sha"] = parts[0]
        info["head_message"] = parts[1] if len(parts) > 1 else ""
    branch = _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        info["branch"] = branch
    return info


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend(slurm_info, local_gpus, total_ram_gb):
    """Build recommended resource allocation.

    SLURM cluster: 100% of node resources (max allocation).
    Prefer GPU partitions with idle nodes.  If the best GPU partition is
    fully allocated but a smaller GPU partition has idle nodes, recommend
    the one with availability (the job can actually start sooner).
    Local workstation: 75% of CPUs/RAM, all GPUs.
    """
    rec = {}

    if slurm_info and slurm_info.get("partitions"):
        gpu_parts = [p for p in slurm_info["partitions"] if p["gpus_per_node"] > 0]

        if gpu_parts:
            # Sort by: has idle nodes (desc), then GPU count (desc)
            gpu_parts_sorted = sorted(
                gpu_parts,
                key=lambda p: (p["nodes_idle"] > 0, p["gpus_per_node"]),
                reverse=True,
            )
            best = gpu_parts_sorted[0]
        else:
            best = slurm_info["partitions"][0]

        rec["environment"] = "slurm"
        rec["partition"] = best["name"]
        rec["cpus"] = best["cpus_per_node"]
        rec["mem_gb"] = best["mem_gb_per_node"]
        rec["gpus"] = best["gpus_per_node"]
        rec["gpu_type"] = best.get("gpu_type", "")
        rec["gres"] = f"{rec.get('gpu_type', 'gpu')}:{rec['gpus']}" if rec["gpus"] else ""
    else:
        rec["environment"] = "local"
        cpus = os.cpu_count() or 1
        rec["cpus"] = math.floor(cpus * 0.75)
        rec["mem_gb"] = math.floor(total_ram_gb * 0.75)
        rec["gpus"] = len(local_gpus)
        if local_gpus:
            rec["gpu_type"] = local_gpus[0]["name"]

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gather():
    """Collect all system info into a dict."""
    total_ram, avail_ram = _parse_meminfo()
    slurm = detect_slurm()
    local_gpus = detect_local_gpus()

    info = {
        "environment": "slurm" if slurm else "local",
        "repo_path": str(REPO),
        "mkseg_python": str(MKSEG_PYTHON),
        "mkseg_available": detect_conda_env(),
        "cpus": os.cpu_count(),
        "ram_total_gb": total_ram,
        "ram_available_gb": avail_ram,
        "local_gpus": local_gpus,
        "disk": detect_disk_space(),
        "git": detect_git_info(),
        "recommended": recommend(slurm, local_gpus, total_ram),
    }

    if slurm:
        info["slurm"] = slurm

    return info


def print_human(info):
    """Print human-readable summary."""
    env = info["environment"].upper()
    print(f"\n{'=' * 60}")
    print(f"  System Info  ({env})")
    print(f"{'=' * 60}")

    print(f"  Repo:         {info['repo_path']}")
    git = info.get("git", {})
    if git:
        print(f"  Git:          {git.get('branch', '?')} @ {git.get('head_sha', '?')}")
    print(f"  mkseg env:    {'OK' if info['mkseg_available'] else 'NOT FOUND'}")
    print(f"  CPUs:         {info['cpus']}")
    print(f"  RAM:          {info['ram_total_gb']:.0f} GB total, {info['ram_available_gb']:.0f} GB available")

    if info.get("local_gpus"):
        print(f"  Local GPUs:")
        for g in info["local_gpus"]:
            print(f"    - {g['name']} ({g['memory_mb']} MB)")

    if info.get("slurm"):
        slurm = info["slurm"]
        print(f"\n  SLURM Partitions:")
        for p in slurm["partitions"]:
            gpu_str = f", {p['gpus_per_node']}x {p.get('gpu_type', 'GPU')}" if p["gpus_per_node"] else ""
            idle = p.get("nodes_idle", 0)
            alloc = p.get("nodes_allocated", 0)
            total = p["node_count"]
            busy_pct = round(alloc / total * 100) if total else 0
            avail_str = f"  [{idle} idle / {total} total, {busy_pct}% busy]"
            print(f"    {p['name']:<12s}  {p['cpus_per_node']} CPUs, {p['mem_gb_per_node']}G RAM{gpu_str}{avail_str}")

        jobs = slurm.get("running_jobs", [])
        if jobs:
            print(f"\n  Running Jobs ({len(jobs)}):")
            for j in jobs:
                print(f"    [{j['job_id']}] {j['name']:<20s} {j['state']:<10s} {j['time']}")

    disk = info.get("disk", {})
    if disk:
        print(f"\n  Disk Space:")
        for path, d in disk.items():
            print(f"    {path}: {d['free_gb']:.0f} GB free / {d['total_gb']:.0f} GB total")

    rec = info["recommended"]
    print(f"\n  Recommended Resources:")
    if rec.get("partition"):
        print(f"    Partition:  {rec['partition']}")
        print(f"    --gres=gpu:{rec['gres']}")
    print(f"    CPUs:       {rec['cpus']}")
    print(f"    RAM:        {rec['mem_gb']} GB")
    print(f"    GPUs:       {rec['gpus']}")
    print(f"{'=' * 60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect system environment and recommend pipeline resources")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    info = gather()

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print_human(info)


if __name__ == "__main__":
    main()
