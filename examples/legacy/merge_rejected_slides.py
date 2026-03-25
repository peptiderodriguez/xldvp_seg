#!/usr/bin/env python3
"""Merge the 3 previously-excluded slides (FGC2, FGC4, MHU4) into the MK dataset.

Reads per-tile features, applies score threshold (0.55), removes annotated
negatives, deduplicates, assigns bone regions, and outputs a combined JSON
matching the existing format (uid, slide, bone, area_um2, mk_score, center_x, center_y).

Usage:
    python3 scripts/merge_rejected_slides.py
"""

import json
import glob
import sys
from pathlib import Path
from collections import Counter

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

BASE = Path("/path/to/data/bm_lmd_feb2026/mk_clf084_dataset")
ANNOT_FILE = BASE / "mk_annotations_2026-03-06_rejected3_unnorm_100pct.json"
REGIONS_FILE = Path("/path/to/data/bm_lmd_feb2026/bone_regions.json")
EXISTING_JSON = BASE / "all_mks_clf075_with_bone.json"
EXISTING_FULL_JSON = BASE / "all_mks_clf075_light.json"
OUTPUT_JSON = BASE / "all_mks_with_rejected3.json"
OUTPUT_FULL_JSON = BASE / "all_mks_with_rejected3_full.json"

NEW_SLIDES = ["2025_11_18_FGC2", "2025_11_18_FGC4", "2025_11_18_MHU4"]
SCORE_THRESHOLD = 0.55


def load_tile_detections(slide_dir):
    """Load and merge all per-tile feature JSONs for a slide."""
    tile_files = sorted(glob.glob(str(slide_dir / "mk" / "tiles" / "*" / "features.json")))
    print(f"  Reading {len(tile_files)} tile files...")
    cells = []
    for tf in tile_files:
        with open(tf) as f:
            cells.extend(json.load(f))
    return cells


def load_bone_polygons(regions_file):
    """Load bone region polygons as prepared Shapely geometries."""
    with open(regions_file) as f:
        data = json.load(f)
    slides = data["slides"]

    polys = {}
    for slide_name, slide_data in slides.items():
        polys[slide_name] = {}
        for bone in ("femur", "humerus"):
            if bone in slide_data and "vertices_px" in slide_data[bone]:
                verts = slide_data[bone]["vertices_px"]
                try:
                    poly = Polygon(verts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        polys[slide_name][bone] = prep(poly)
                except Exception as e:
                    print(f"  WARNING: Invalid {bone} polygon for {slide_name}: {e}")
    return polys


def assign_bone(x, y, slide_polys):
    """Assign a cell to femur/humerus based on centroid."""
    pt = Point(x, y)
    for bone, ppoly in slide_polys.items():
        if ppoly.contains(pt):
            return bone
    return None


def main():
    # Load annotations
    with open(ANNOT_FILE) as f:
        annot = json.load(f)
    neg_uids = set(annot["negative"])
    print(f"Loaded {len(neg_uids)} negative annotations")

    # Load bone regions
    bone_polys = load_bone_polygons(REGIONS_FILE)
    print(f"Loaded bone regions for {len(bone_polys)} slides")

    # Load existing datasets (light + full)
    with open(EXISTING_JSON) as f:
        existing = json.load(f)
    print(f"Loaded {len(existing)} existing detections from {EXISTING_JSON.name}")

    with open(EXISTING_FULL_JSON) as f:
        existing_full = json.load(f)
    print(f"Loaded {len(existing_full)} existing full detections from {EXISTING_FULL_JSON.name}")

    existing_uids = {c["uid"] for c in existing}

    # Process each new slide
    new_cells = []       # light format (uid, slide, bone, area, score, center)
    new_cells_full = []  # full format (with features dict for ANOVA)
    for slide_name in NEW_SLIDES:
        slide_dir = BASE / slide_name
        if not slide_dir.exists():
            print(f"WARNING: {slide_dir} not found, skipping")
            continue

        tag = slide_name.split("_")[-1]
        print(f"\n=== {tag} ===")
        all_cells = load_tile_detections(slide_dir)
        print(f"  Total detections: {len(all_cells)}")

        slide_polys = bone_polys.get(slide_name, {})
        if not slide_polys:
            print(f"  WARNING: No bone regions for {slide_name}")

        # Filter, deduplicate, assign bones
        seen_uids = set()
        kept = []
        bone_counts = Counter()
        n_below = n_neg = n_dup = 0

        for c in all_cells:
            if c["mk_score"] < SCORE_THRESHOLD:
                n_below += 1
                continue
            if c["uid"] in neg_uids:
                n_neg += 1
                continue
            if c["uid"] in seen_uids or c["uid"] in existing_uids:
                n_dup += 1
                continue
            seen_uids.add(c["uid"])

            # Get centroid — tile features use center: [x, y]
            center = c.get("center", [0, 0])
            cx, cy = float(center[0]), float(center[1])

            # Assign bone
            bone = assign_bone(cx, cy, slide_polys)
            bone_counts[bone] += 1

            area_um2 = c.get("area_um2", c.get("area", 0))

            # Light format (for LMD selection)
            kept.append({
                "uid": c["uid"],
                "slide": slide_name,
                "bone": bone,
                "area_um2": area_um2,
                "mk_score": c["mk_score"],
                "center_x": cx,
                "center_y": cy,
            })

            # Full format (for ANOVA — needs features dict)
            full_cell = {
                "uid": c["uid"],
                "slide": slide_name,
                "bone": bone,
                "area_um2": area_um2,
                "mk_score": c["mk_score"],
                "features": c.get("features", {}),
            }
            new_cells_full.append(full_cell)

        print(f"  Below {SCORE_THRESHOLD}: {n_below}")
        print(f"  Annotated neg:   {n_neg}")
        print(f"  Duplicates:      {n_dup}")
        print(f"  Kept:            {len(kept)}")
        print(f"  Bone assignment: {dict(bone_counts)}")

        if kept:
            scores = [c["mk_score"] for c in kept]
            areas = [c["area_um2"] for c in kept]
            print(f"  Score: {min(scores):.3f} - {max(scores):.3f}")
            print(f"  Area:  {min(areas):.0f} - {max(areas):.0f} um²")

        new_cells.extend(kept)

    # Combine
    combined = existing + new_cells
    print(f"\n--- Summary ---")
    print(f"Existing:  {len(existing)}")
    print(f"New:       {len(new_cells)}")
    print(f"Combined:  {len(combined)}")

    # Per-slide per-bone breakdown
    print(f"\n{'Slide':<8} {'femur':>6} {'humer':>6} {'unkn':>6} {'total':>6}")
    print("-" * 38)
    for slide in sorted(set(c["slide"] for c in combined)):
        tag = slide.split("_")[-1]
        sc = [c for c in combined if c["slide"] == slide]
        fem = sum(1 for c in sc if c.get("bone") == "femur")
        hum = sum(1 for c in sc if c.get("bone") == "humerus")
        unk = sum(1 for c in sc if c.get("bone") is None)
        print(f"{tag:<8} {fem:>6} {hum:>6} {unk:>6} {len(sc):>6}")

    # Write light output (for LMD selection)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(combined, f)
    print(f"\nWrote {len(combined)} detections to {OUTPUT_JSON.name}")
    print(f"File size: {OUTPUT_JSON.stat().st_size / 1024 / 1024:.1f} MB")

    # Write full output (for ANOVA — with features)
    combined_full = existing_full + new_cells_full
    with open(OUTPUT_FULL_JSON, "w") as f:
        json.dump(combined_full, f)
    print(f"Wrote {len(combined_full)} full detections to {OUTPUT_FULL_JSON.name}")
    print(f"File size: {OUTPUT_FULL_JSON.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
