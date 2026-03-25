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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
XLDVP_PYTHON = Path(os.environ.get("XLDVP_PYTHON", os.environ.get("MKSEG_PYTHON", "python")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd, timeout=10):
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
            gres = gres[: gres.index("(")]

        # Count nodes from nodelist (e.g. "hpcl[8005-8059]")
        node_count = 1
        range_match = re.search(r"\[(\d+)-(\d+)\]", nodelist)
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
            partitions[partition]["cpus_per_node"] = max(
                partitions[partition]["cpus_per_node"], cpus
            )
            partitions[partition]["mem_gb_per_node"] = max(
                partitions[partition]["mem_gb_per_node"], round(mem_mb / 1024)
            )
            partitions[partition]["node_count"] = max(
                partitions[partition]["node_count"], node_count
            )
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
                    slurm["running_jobs"].append(
                        {
                            "job_id": parts[0],
                            "name": parts[1],
                            "partition": parts[2],
                            "state": parts[3],
                            "time": parts[4] if len(parts) > 4 else "",
                            "node": parts[5] if len(parts) > 5 else "",
                        }
                    )

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
        paths = [str(REPO), "/tmp"]
    result = {}
    for p in paths:
        try:
            st = os.statvfs(p)
            total_gb = round(st.f_blocks * st.f_frsize / (1024**3), 1)
            free_gb = round(st.f_bavail * st.f_frsize / (1024**3), 1)
            result[p] = {"total_gb": total_gb, "free_gb": free_gb}
        except OSError:
            pass
    return result


def detect_conda_env():
    """Check if xldvp_seg conda env exists."""
    return XLDVP_PYTHON.exists()


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
# Per-node availability query
# ---------------------------------------------------------------------------

MAX_RECOMMENDED_CPUS = 128  # Cap CPU recommendation at this value for schedulability


def query_per_node_availability(partition):
    """Query per-node idle CPUs, free memory (MB), GPU allocation, and state.

    Runs: sinfo -p <partition> -N -o "%n %C %e %G %T" --noheader

    %C format: "allocated/idle/other/total"
    %e: free memory in MB
    %G: GRES (e.g. "gpu:l40s:4(IDX:0,1,2,3)" or "(null)")
    %T: node state (idle, allocated, mixed, down, drain, etc.)

    Returns a list of dicts with keys: node, idle_cpus, free_mem_mb,
    gpus_total, gpus_allocated, state.  Returns [] on failure.
    """
    raw = _run(["sinfo", "-p", partition, "-N", "-o", "%n %C %e %G %T", "--noheader"])
    if not raw:
        return []

    nodes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        node = parts[0]
        cpu_str = parts[1]  # "allocated/idle/other/total"
        mem_str = parts[2]  # free MB (or "N/A")
        gres_str = parts[3]  # e.g. "gpu:l40s:4(IDX:...)" or "(null)"
        state = parts[4]

        # Parse CPU counts
        idle_cpus = 0
        cpu_parts = cpu_str.split("/")
        if len(cpu_parts) == 4:
            try:
                idle_cpus = int(cpu_parts[1])
            except ValueError:
                pass

        # Parse free memory
        free_mem_mb = 0
        try:
            free_mem_mb = int(mem_str)
        except ValueError:
            pass

        # Parse GPU total from GRES
        gpus_total = 0
        if gres_str not in ("(null)", "N/A", ""):
            # Strip "(IDX:...)" or "(S:...)" suffix
            if "(" in gres_str:
                gres_str = gres_str[: gres_str.index("(")]
            gparts = gres_str.split(":")
            if len(gparts) >= 3:
                try:
                    gpus_total = int(gparts[-1])
                except ValueError:
                    pass
            elif len(gparts) == 2:
                try:
                    gpus_total = int(gparts[-1])
                except ValueError:
                    pass

        nodes.append(
            {
                "node": node,
                "idle_cpus": idle_cpus,
                "free_mem_mb": free_mem_mb,
                "gpus_total": gpus_total,
                "state": state,
            }
        )

    return nodes


def _best_schedulable_resources(partition_info, per_node):
    """Given partition info and per-node data, return (cpus, mem_gb) that
    reflect what is actually schedulable right now on a single node.

    Strategy:
    - Find nodes in 'idle' or 'mixed' state
    - Among those, pick the node with the most idle CPUs
    - Cap at MAX_RECOMMENDED_CPUS and 75% of node spec
    - Fall back to 75% of partition spec (capped) if no per-node data
    """
    cpus_spec = partition_info["cpus_per_node"]
    mem_spec = partition_info["mem_gb_per_node"]

    # Filter to schedulable nodes (idle or mixed)
    schedulable = [n for n in per_node if n["state"] in ("idle", "mixed", "idle~", "mixed~")]

    if schedulable:
        # Sort by idle CPUs descending
        schedulable.sort(key=lambda n: n["idle_cpus"], reverse=True)
        best_node = schedulable[0]
        # Use min(idle_cpus, 75% of spec) to avoid requesting more than the node has
        raw_cpus = min(best_node["idle_cpus"], int(cpus_spec * 0.75))
        raw_mem = best_node["free_mem_mb"] // 1024  # MB → GB
        if raw_mem == 0:
            raw_mem = int(mem_spec * 0.75)
    else:
        # No per-node data or no schedulable nodes: fall back to 75% of spec
        raw_cpus = int(cpus_spec * 0.75)
        raw_mem = int(mem_spec * 0.75)

    # Always cap CPUs at MAX_RECOMMENDED_CPUS
    cpus = min(raw_cpus, MAX_RECOMMENDED_CPUS)
    # Ensure at least 1 CPU
    cpus = max(cpus, 1)
    # Ensure reasonable memory
    mem_gb = max(raw_mem, 1)

    return cpus, mem_gb


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def recommend(slurm_info, local_gpus, total_ram_gb):
    """Build recommended resource allocation.

    SLURM cluster: use per-node availability (sinfo -N) to recommend
    resources based on what is *currently schedulable*, capped at
    MAX_RECOMMENDED_CPUS (128) for reliability on mixed/busy nodes.
    Always request all GPUs on GPU partitions.  Prefer GPU partitions
    with idle nodes.
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

        # Query per-node availability for the selected partition
        per_node = query_per_node_availability(best["name"])
        cpus, mem_gb = _best_schedulable_resources(best, per_node)

        rec["environment"] = "slurm"
        rec["partition"] = best["name"]
        rec["cpus"] = cpus
        rec["mem_gb"] = mem_gb
        rec["gpus"] = best["gpus_per_node"]
        rec["gpu_type"] = best.get("gpu_type", "")
        # Always include a GPU gres on GPU partitions (even if gpus==0 due to parse failure)
        if best["gpus_per_node"] > 0:
            gpu_type = best.get("gpu_type", "")
            gres_type = f"gpu:{gpu_type}" if gpu_type else "gpu"
            rec["gres"] = f"{gres_type}:{best['gpus_per_node']}"
        else:
            rec["gres"] = ""
        rec["per_node_data"] = per_node
    else:
        rec["environment"] = "local"
        cpus = os.cpu_count() or 1
        rec["cpus"] = min(math.floor(cpus * 0.75), MAX_RECOMMENDED_CPUS)
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
        "xldvp_python": str(XLDVP_PYTHON),
        "xldvp_available": detect_conda_env(),
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
    print(f"  xldvp_seg env: {'OK' if info['xldvp_available'] else 'NOT FOUND'}")
    print(f"  CPUs:         {info['cpus']}")
    print(
        f"  RAM:          {info['ram_total_gb']:.0f} GB total, {info['ram_available_gb']:.0f} GB available"
    )

    if info.get("local_gpus"):
        print("  Local GPUs:")
        for g in info["local_gpus"]:
            print(f"    - {g['name']} ({g['memory_mb']} MB)")

    if info.get("slurm"):
        slurm = info["slurm"]
        print("\n  SLURM Partitions:")
        for p in slurm["partitions"]:
            gpu_str = (
                f", {p['gpus_per_node']}x {p.get('gpu_type', 'GPU')}" if p["gpus_per_node"] else ""
            )
            idle = p.get("nodes_idle", 0)
            alloc = p.get("nodes_allocated", 0)
            total = p["node_count"]
            busy_pct = round(alloc / total * 100) if total else 0
            avail_str = f"  [{idle} idle / {total} total, {busy_pct}% busy]"
            print(
                f"    {p['name']:<12s}  {p['cpus_per_node']} CPUs, {p['mem_gb_per_node']}G RAM{gpu_str}{avail_str}"
            )

        jobs = slurm.get("running_jobs", [])
        if jobs:
            print(f"\n  Running Jobs ({len(jobs)}):")
            for j in jobs:
                print(f"    [{j['job_id']}] {j['name']:<20s} {j['state']:<10s} {j['time']}")

    disk = info.get("disk", {})
    if disk:
        print("\n  Disk Space:")
        for path, d in disk.items():
            print(f"    {path}: {d['free_gb']:.0f} GB free / {d['total_gb']:.0f} GB total")

    rec = info["recommended"]
    print("\n  Recommended Resources:")
    if rec.get("partition"):
        print(f"    Partition:  {rec['partition']}")
        if rec.get("gres"):
            print(f"    --gres={rec['gres']}")
    print(f"    CPUs:       {rec['cpus']}  (capped at {MAX_RECOMMENDED_CPUS} for schedulability)")
    print(f"    RAM:        {rec['mem_gb']} GB")
    print(f"    GPUs:       {rec['gpus']}")
    # Show per-node availability summary for the recommended partition
    per_node = rec.get("per_node_data", [])
    if per_node:
        schedulable = [n for n in per_node if n["state"] in ("idle", "mixed", "idle~", "mixed~")]
        max_idle_cpus = max((n["idle_cpus"] for n in schedulable), default=0)
        print(
            f"    Node avail: {len(schedulable)}/{len(per_node)} nodes schedulable, "
            f"max {max_idle_cpus} idle CPUs on best node"
        )
    print(f"{'=' * 60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect system environment and recommend pipeline resources"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    info = gather()

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print_human(info)


if __name__ == "__main__":
    main()
