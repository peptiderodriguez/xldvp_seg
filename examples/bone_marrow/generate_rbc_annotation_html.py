#!/usr/bin/env python3
"""Generate annotation HTML for RBC cluster candidates from unfiltered SAM detections.

Filters low-mk_score detections (SAM "rejects") from unfiltered detection JSON files,
randomly samples them with score-bin stratification, and produces a paginated HTML
annotation viewer matching the pipeline's standard dark-themed interface.

Annotation categories: "rbc" (RBC cluster), "mk" (megakaryocyte), "other".

The crop_b64 images are already present in the unfiltered JSON, so no CZI loading
is needed -- the script is fast and lightweight.

Usage:
    PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/generate_rbc_annotation_html.py \
        --input-dir /path/to/output/bm_lmd_feb2026/mk_clf084_dataset/per_slide_unfiltered/ \
        --output /path/to/output/bm_lmd_feb2026/mk_clf084_dataset/rbc_annotation/index.html \
        --max-samples 200 \
        --min-score 0.03 --max-score 0.30 \
        --min-area 100

    # Specific slides only:
    PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/generate_rbc_annotation_html.py \
        --input-dir /path/to/per_slide_unfiltered/ \
        --output /path/to/output/index.html \
        --slides FGC1 FGC2 MHU3
"""

import argparse
import html as html_mod
import json
import sys
from pathlib import Path

import base64
import io

import cv2
import numpy as np
from PIL import Image

from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _esc(value) -> str:
    """Escape a value for safe insertion into HTML/JS strings."""
    return html_mod.escape(str(value), quote=True)


def composite_crop_mask(crop_b64, mask_b64, contour_color=(0, 255, 0), alpha=0.3):
    """Overlay mask contour on crop image, return composited base64 JPEG.

    Args:
        crop_b64: Base64-encoded JPEG crop image.
        mask_b64: Base64-encoded PNG mask (250x250, binary).
        contour_color: BGR color for contour outline.
        alpha: Blend alpha for mask fill.

    Returns:
        Base64-encoded JPEG with mask overlay, or original crop_b64 on failure.
    """
    try:
        # Decode crop
        crop_bytes = base64.b64decode(crop_b64)
        crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_img = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
        if crop_img is None:
            return crop_b64

        # Decode mask
        mask_bytes = base64.b64decode(mask_b64)
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return crop_b64

        # Resize mask to match crop if needed
        ch, cw = crop_img.shape[:2]
        mh, mw = mask_img.shape[:2]
        if (mh, mw) != (ch, cw):
            mask_img = cv2.resize(mask_img, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # Threshold mask
        _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        # Draw contour outline only (no fill)
        result = crop_img.copy()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, contour_color, 2)

        # Re-encode
        _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('ascii')
    except Exception:
        return crop_b64


def load_and_sample_candidates(input_dir, min_score, max_score, min_area,
                                max_samples_per_slide, slides=None, seed=42):
    """Load, filter, and sample candidates one slide at a time to limit memory.

    Each slide's JSON is loaded, filtered, sampled, then discarded before
    the next slide is loaded. Only the sampled detections (with crop/mask)
    are kept in memory.

    Returns:
        (all_samples, slide_counts) where all_samples is a flat list of
        sampled detection dicts and slide_counts maps slide_name -> count.
    """
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.glob("*_full_unfiltered.json"))

    if not json_files:
        logger.error(f"No *_full_unfiltered.json files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} unfiltered detection files")

    rng = np.random.default_rng(seed)
    all_samples = []
    slide_counts = {}

    for jf in json_files:
        slide_name = jf.stem.replace("_full_unfiltered", "")

        # Filter by slide name if specified
        if slides:
            if not any(s in slide_name for s in slides):
                continue

        logger.info(f"Loading {jf.name}...")
        with open(jf) as f:
            detections = json.load(f)

        logger.info(f"  {len(detections):,} total detections")

        # Filter by score range and area — only keep keys we need
        filtered = []
        for det in detections:
            score = det.get("mk_score") or 0
            area = det.get("area_um2") or 0
            crop = det.get("crop_b64")

            if score < min_score or score >= max_score:
                continue
            if area < min_area:
                continue
            if not crop:
                continue

            filtered.append(det)

        # Free the full detection list immediately
        del detections

        logger.info(
            f"  {len(filtered):,} candidates in score range "
            f"[{min_score:.2f}, {max_score:.2f}) with area >= {min_area} um2"
        )

        if not filtered:
            continue

        # Sample from this slide
        sampled = stratified_sample(filtered, max_samples_per_slide, n_bins=5, rng=rng)
        del filtered  # free the rest

        # Tag with slide name
        for s in sampled:
            s["slide"] = slide_name

        slide_counts[slide_name] = len(sampled)
        all_samples.extend(sampled)
        logger.info(
            f"  {slide_name}: {len(sampled):,} sampled"
        )

    return all_samples, slide_counts


def stratified_sample(candidates, max_samples, n_bins=5, rng=None):
    """Sample detections stratified by score bins for diversity.

    Divides the score range into n_bins equal bins, samples equally from each,
    with overflow redistributed to bins that have more.

    Args:
        candidates: List of detection dicts with 'mk_score'.
        max_samples: Maximum number to sample.
        n_bins: Number of score bins for stratification.
        rng: numpy random Generator.

    Returns:
        Sampled list of detection dicts.
    """
    if len(candidates) <= max_samples:
        return list(candidates)

    if rng is None:
        rng = np.random.default_rng(42)

    # Get score range
    scores = np.array([d["mk_score"] for d in candidates])
    score_min = scores.min()
    score_max = scores.max()

    # Create bins
    bin_edges = np.linspace(score_min, score_max + 1e-9, n_bins + 1)
    bins = [[] for _ in range(n_bins)]
    for i, det in enumerate(candidates):
        s = det["mk_score"]
        bin_idx = int(np.searchsorted(bin_edges[1:], s, side="left"))
        bin_idx = min(bin_idx, n_bins - 1)
        bins[bin_idx].append(det)

    # Target per bin
    per_bin = max_samples // n_bins
    sampled = []
    overflow = 0

    for b in bins:
        target = per_bin + overflow
        if not b or target <= 0:
            overflow = max(0, target)
            continue
        if len(b) <= target:
            sampled.extend(b)
            overflow = target - len(b)
        else:
            indices = rng.choice(len(b), target, replace=False)
            sampled.extend([b[i] for i in indices])
            overflow = 0

    # If we still need more due to rounding, sample from remainder
    if len(sampled) < max_samples:
        sampled_ids = {id(d) for d in sampled}
        remaining = [d for d in candidates if id(d) not in sampled_ids]
        if remaining:
            extra = min(max_samples - len(sampled), len(remaining))
            indices = rng.choice(len(remaining), extra, replace=False)
            sampled.extend([remaining[i] for i in indices])

    # Shuffle final result
    shuffle_idx = rng.permutation(len(sampled))
    sampled = [sampled[i] for i in shuffle_idx]

    return sampled[:max_samples]


def get_rbc_css():
    """CSS for the RBC annotation viewer -- matches pipeline dark theme."""
    return '''
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #0a0a0a; color: #ddd; }

        .header {
            background: #111;
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header h1 {
            font-size: 1.2em;
            font-weight: normal;
        }

        .header-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-top: 2px;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            padding: 8px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }

        .nav-btn:hover {
            background: #222;
        }

        .page-info {
            padding: 8px 15px;
            color: #888;
        }

        .stats-row {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
        }

        .stats-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .stats-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat {
            padding: 4px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .stat.rbc-stat {
            border-left: 3px solid #e44;
        }

        .stat.mk-stat {
            border-left: 3px solid #4a4;
        }

        .stat.other-stat {
            border-left: 3px solid #aa4;
        }

        .content {
            padding: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }

        .card {
            background: #111;
            border: 2px solid #333;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .card.selected {
            box-shadow: 0 0 0 3px #fff;
        }

        .card.labeled-rbc {
            border-color: #e44 !important;
            background: #130f0f !important;
        }

        .card.labeled-mk {
            border-color: #4a4 !important;
            background: #0f130f !important;
        }

        .card.labeled-other {
            border-color: #aa4 !important;
            background: #13130f !important;
        }

        .card-img-container {
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .card-info {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #333;
            gap: 10px;
        }

        .card-meta {
            flex: 1;
            min-width: 0;
        }

        .card-id {
            font-size: 0.75em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-stats {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }

        .buttons {
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .btn:hover {
            background: #222;
        }

        .btn-rbc {
            border-color: #e44;
            color: #e44;
        }

        .btn-mk {
            border-color: #4a4;
            color: #4a4;
        }

        .btn-other {
            border-color: #aa4;
            color: #aa4;
        }

        .btn-export {
            border-color: #44a;
            color: #44a;
        }

        .btn-danger {
            border-color: #a44;
            color: #a44;
        }

        .btn-danger:hover {
            background: #311;
        }

        .keyboard-hint {
            text-align: center;
            padding: 15px;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #222;
        }

        .footer {
            background: #111;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
    '''


def get_rbc_js(total_pages, experiment_name, page_num):
    """JavaScript for RBC annotation with 3-class labeling (rbc/mk/other).

    Uses localStorage for persistence, matching the pipeline's pattern.
    Labels: 'rbc' = 1, 'mk' = 2, 'other' = 3.
    """
    experiment_name_safe = _esc(experiment_name)
    global_key = _esc(f"rbc_{experiment_name}_annotations")
    page_key = _esc(f"rbc_{experiment_name}_labels_page{page_num}")
    page_key_prefix = _esc(f"rbc_{experiment_name}_labels_page")

    return f'''
        const CELL_TYPE = 'rbc';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const TOTAL_PAGES = {total_pages};
        const GLOBAL_STORAGE_KEY = '{global_key}';
        const PAGE_STORAGE_KEY = '{page_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';

        // Label values: 1=rbc, 2=mk, 3=other
        const LABEL_RBC = 1;
        const LABEL_MK = 2;
        const LABEL_OTHER = 3;

        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        function saveLabels() {{
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            cards.forEach(card => {{ if (!(card.id in labels)) delete globalLabels[card.id]; }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        function loadAnnotations() {{
            try {{
                let saved = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!saved) {{
                    const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                    if (globalSaved) {{
                        const globalLabels = JSON.parse(globalSaved);
                        const pageUids = new Set(Array.from(cards).map(c => c.id));
                        for (const [uid, label] of Object.entries(globalLabels)) {{
                            if (pageUids.has(uid)) labels[uid] = label;
                        }}
                    }}
                }} else {{
                    labels = JSON.parse(saved);
                }}
            }} catch(e) {{ console.error(e); }}

            cards.forEach((card) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-rbc', 'labeled-mk', 'labeled-other');
            card.dataset.label = label;
            if (label === LABEL_RBC) card.classList.add('labeled-rbc');
            else if (label === LABEL_MK) card.classList.add('labeled-mk');
            else if (label === LABEL_OTHER) card.classList.add('labeled-other');
        }}

        function setLabel(uid, label, autoAdvance) {{
            autoAdvance = autoAdvance || false;
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-rbc', 'labeled-mk', 'labeled-other');
                    card.dataset.label = -1;
                }}
            }} else {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }}

            saveLabels();
            updateStats();

            if (autoAdvance && selectedIdx >= 0 && selectedIdx < cards.length - 1) {{
                selectCard(selectedIdx + 1);
            }}
        }}

        function updateStats() {{
            let pageRbc = 0, pageMk = 0, pageOther = 0;
            let globalRbc = 0, globalMk = 0, globalOther = 0;

            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] === LABEL_RBC) pageRbc++;
                else if (labels[uid] === LABEL_MK) pageMk++;
                else if (labels[uid] === LABEL_OTHER) pageOther++;
            }});

            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === LABEL_RBC) globalRbc++;
                        else if (v === LABEL_MK) globalMk++;
                        else if (v === LABEL_OTHER) globalOther++;
                    }}
                }}
            }} catch(e) {{ console.error(e); }}

            const el = (id) => document.getElementById(id);
            if (el('pageRbc')) el('pageRbc').textContent = pageRbc;
            if (el('pageMk')) el('pageMk').textContent = pageMk;
            if (el('pageOther')) el('pageOther').textContent = pageOther;
            if (el('globalRbc')) el('globalRbc').textContent = globalRbc;
            if (el('globalMk')) el('globalMk').textContent = globalMk;
            if (el('globalOther')) el('globalOther').textContent = globalOther;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function exportAnnotations() {{
            let allLabels = {{}};
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) allLabels = JSON.parse(globalSaved);
            }} catch(e) {{ console.error(e); }}

            const data = {{
                cell_type: 'rbc',
                experiment_name: EXPERIMENT_NAME,
                exported_at: new Date().toISOString(),
                rbc: [],
                mk: [],
                other: []
            }};

            for (const [uid, label] of Object.entries(allLabels)) {{
                if (label === LABEL_RBC) data.rbc.push(uid);
                else if (label === LABEL_MK) data.mk.push(uid);
                else if (label === LABEL_OTHER) data.other.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'rbc_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function importAnnotations() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {{
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (ev) => {{
                    try {{
                        const data = JSON.parse(ev.target.result);
                        let imported = {{}};
                        // Support rbc/mk/other format
                        (data.rbc || []).forEach(uid => imported[uid] = LABEL_RBC);
                        (data.mk || []).forEach(uid => imported[uid] = LABEL_MK);
                        (data.other || []).forEach(uid => imported[uid] = LABEL_OTHER);
                        // Also support annotations dict format
                        if (data.annotations) {{
                            for (const [uid, val] of Object.entries(data.annotations)) {{
                                if (val === 'rbc' || val === 1) imported[uid] = LABEL_RBC;
                                else if (val === 'mk' || val === 2) imported[uid] = LABEL_MK;
                                else if (val === 'other' || val === 3) imported[uid] = LABEL_OTHER;
                            }}
                        }}
                        let existing = {{}};
                        try {{
                            const gs = localStorage.getItem(GLOBAL_STORAGE_KEY);
                            if (gs) existing = JSON.parse(gs);
                        }} catch(ex) {{}}
                        const merged = {{...imported, ...existing}};
                        localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(merged));
                        for (const [uid, val] of Object.entries(imported)) {{
                            if (labels[uid] === undefined) labels[uid] = val;
                        }}
                        saveLabels();
                        cards.forEach(card => {{
                            const uid = card.id;
                            if (labels[uid] !== undefined) applyLabelToCard(card, labels[uid]);
                        }});
                        updateStats();
                        const n = Object.keys(imported).length;
                        alert('Imported ' + n + ' annotations (' + Object.keys(merged).length + ' total after merge)');
                    }} catch(err) {{
                        alert('Error importing: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        function clearPage() {{
            if (!confirm('Clear annotations on this page?')) return;
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    delete labels[uid];
                    card.classList.remove('labeled-rbc', 'labeled-mk', 'labeled-other');
                    card.dataset.label = -1;
                }}
            }});
            saveLabels();
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            labels = {{}};
            saveLabels();
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                localStorage.setItem(PAGE_KEY_PREFIX + i, JSON.stringify({{}}));
            }}
            cards.forEach(card => {{
                card.classList.remove('labeled-rbc', 'labeled-mk', 'labeled-other');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
        }}

        document.addEventListener('keydown', (e) => {{
            // Skip if typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Navigation
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                e.preventDefault();
                selectCard(Math.min(selectedIdx + 1, cards.length - 1));
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                e.preventDefault();
                selectCard(Math.max(selectedIdx - 1, 0));
            }}
            // Labeling
            else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'r') setLabel(uid, LABEL_RBC, true);
                else if (e.key.toLowerCase() === 'm') setLabel(uid, LABEL_MK, true);
                else if (e.key.toLowerCase() === 'o') setLabel(uid, LABEL_OTHER, true);
            }}
        }});

        // Initialize
        loadAnnotations();
    '''


def generate_annotation_page(samples, page_num, total_pages, experiment_name,
                              subtitle=None, page_prefix='rbc_page'):
    """Generate a single HTML annotation page for RBC candidates.

    Args:
        samples: List of sample dicts with uid, crop_b64, mk_score, area_um2, slide.
        page_num: Current page number (1-indexed).
        total_pages: Total number of pages.
        experiment_name: Experiment name for localStorage isolation.
        subtitle: Optional subtitle.
        page_prefix: Prefix for page filenames.

    Returns:
        HTML string.
    """
    title = "RBC Cluster Annotation"

    # Build navigation
    nav_html = '<div class="nav-buttons">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{page_prefix}_{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} / {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{page_prefix}_{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    subtitle_html = ''
    if subtitle:
        subtitle_html = f'<div class="header-subtitle">{_esc(subtitle)}</div>'

    # Build cards
    cards_html = ''
    for sample in samples:
        uid = _esc(sample['uid'])
        crop_b64 = sample['crop_b64']
        mask_b64 = sample.get('mask_b64')
        if mask_b64:
            crop_b64 = composite_crop_mask(crop_b64, mask_b64)
        mk_score = sample.get('mk_score', 0)
        area_um2 = sample.get('area_um2', 0)
        slide = _esc(sample.get('slide', ''))
        sam2_iou = sample.get('sam2_iou', None)
        sam2_stability = sample.get('sam2_stability', None)

        # Format stats line
        stats_parts = []
        stats_parts.append(f"{area_um2:.1f} &micro;m&sup2;")
        stats_parts.append(f"MK: {mk_score:.3f}")
        if sam2_iou is not None:
            stats_parts.append(f"IoU: {sam2_iou:.2f}")
        if sam2_stability is not None:
            stats_parts.append(f"stab: {sam2_stability:.2f}")
        stats_parts.append(f"{slide}")

        stats_str = ' | '.join(stats_parts)

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/jpeg;base64,{crop_b64}" alt="{uid}">
            </div>
            <div class="card-info">
                <div class="card-meta">
                    <div class="card-id">{uid}</div>
                    <div class="card-stats">{stats_str}</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-rbc" onclick="setLabel('{uid}', 1)">RBC</button>
                    <button class="btn btn-mk" onclick="setLabel('{uid}', 2)">MK</button>
                    <button class="btn btn-other" onclick="setLabel('{uid}', 3)">Other</button>
                </div>
            </div>
        </div>
'''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{_esc(title)} - Page {page_num}/{total_pages}</title>
    <style>{get_rbc_css()}</style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <div>
                <h1>{_esc(title)} - Page {page_num}/{total_pages}</h1>
                {subtitle_html}
            </div>
            {nav_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">Page:</span>
                <div class="stat rbc-stat">RBC: <span id="pageRbc">0</span></div>
                <div class="stat mk-stat">MK: <span id="pageMk">0</span></div>
                <div class="stat other-stat">Other: <span id="pageOther">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Total:</span>
                <div class="stat rbc-stat">RBC: <span id="globalRbc">0</span></div>
                <div class="stat mk-stat">MK: <span id="globalMk">0</span></div>
                <div class="stat other-stat">Other: <span id="globalOther">0</span></div>
            </div>
            <button class="btn btn-export" onclick="exportAnnotations()">Export</button>
            <button class="btn" onclick="importAnnotations()">Import</button>
            <button class="btn" onclick="clearPage()">Clear Page</button>
            <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
        </div>
    </div>

    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>

    <div class="keyboard-hint">
        Keyboard: R=RBC, M=MK, O=Other, Arrow keys=Navigate
    </div>

    <div class="footer">
        {nav_html}
    </div>

    <script>{get_rbc_js(total_pages, experiment_name, page_num)}</script>
</body>
</html>'''

    return html


def generate_index_page(total_samples, total_pages, experiment_name,
                         slide_counts, score_range, area_range,
                         page_prefix='rbc_page'):
    """Generate the index/landing page for RBC annotation.

    Args:
        total_samples: Total number of samples.
        total_pages: Total number of pages.
        experiment_name: Experiment name for localStorage.
        slide_counts: Dict of slide_name -> count of sampled detections.
        score_range: Tuple (min_score, max_score).
        area_range: Tuple (min_area, max_area) in um2.
        page_prefix: Prefix for page filenames.

    Returns:
        HTML string.
    """
    title = "RBC Cluster Annotation"
    experiment_name_safe = _esc(experiment_name)
    global_key = _esc(f"rbc_{experiment_name}_annotations")
    page_key_prefix = _esc(f"rbc_{experiment_name}_labels_page")

    # Build info lines
    info_lines = [
        "Task: Annotate low-MK-score SAM detections as RBC cluster, MK, or other",
        f"Score range: [{score_range[0]:.3f}, {score_range[1]:.3f})",
        f"Area range: [{area_range[0]:.0f}, {area_range[1]:.0f}] &micro;m&sup2;",
        f"Total candidates: {total_samples:,}",
        f"Pages: {total_pages}",
        f"Slides: {len(slide_counts)}",
    ]

    # Slide breakdown
    for slide, count in sorted(slide_counts.items()):
        info_lines.append(f"&nbsp;&nbsp;{_esc(slide)}: {count:,} samples")

    info_html = '<br>'.join(info_lines)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{_esc(title)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; text-align: center; }}

        .header {{
            background: #111;
            padding: 30px;
            border: 1px solid #333;
            margin-bottom: 20px;
        }}

        h1 {{
            font-size: 1.5em;
            font-weight: normal;
            margin-bottom: 10px;
        }}

        .info-block {{
            color: #aaa;
            line-height: 1.8;
            margin: 20px 0;
            font-size: 1.1em;
            text-align: left;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}

        .section {{
            margin: 30px 0;
            text-align: center;
        }}

        .btn {{
            padding: 15px 40px;
            background: #1a1a1a;
            border: 1px solid #4a4;
            color: #4a4;
            cursor: pointer;
            font-family: monospace;
            font-size: 1.1em;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}

        .btn:hover {{
            background: #0f130f;
        }}

        .btn-export {{
            border-color: #44a;
            color: #44a;
        }}

        .btn-export:hover {{
            background: #0f0f13;
        }}

        .btn-danger {{
            border-color: #a44;
            color: #a44;
        }}

        .btn-danger:hover {{
            background: #311;
        }}

        .instructions {{
            background: #111;
            border: 1px solid #333;
            padding: 20px 30px;
            margin: 20px auto;
            max-width: 600px;
            text-align: left;
            line-height: 1.8;
            color: #aaa;
        }}

        .instructions h2 {{
            font-size: 1.1em;
            color: #ddd;
            margin-bottom: 10px;
        }}

        .key {{
            display: inline-block;
            padding: 2px 8px;
            background: #222;
            border: 1px solid #555;
            border-radius: 3px;
            font-weight: bold;
            color: #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{_esc(title)}</h1>
        <div class="info-block">
            {info_html}
        </div>
    </div>

    <div class="instructions">
        <h2>Annotation Guide</h2>
        <p>These are <strong>low-MK-score SAM detections</strong> that the MK classifier rejected.
           Many are likely RBC clusters (red blood cell aggregates) which have distinctive
           morphology: dark/dense, multi-lobular, irregular shape.</p>
        <p><br><strong>Categories:</strong></p>
        <p>&bull; <span style="color:#e44;font-weight:bold">RBC</span> - RBC cluster or RBC aggregate</p>
        <p>&bull; <span style="color:#4a4;font-weight:bold">MK</span> - Megakaryocyte (misclassified by RF)</p>
        <p>&bull; <span style="color:#aa4;font-weight:bold">Other</span> - Neither RBC nor MK (debris, artifact, etc.)</p>
        <p><br><strong>Keyboard shortcuts:</strong></p>
        <p><span class="key">R</span> = RBC &nbsp; <span class="key">M</span> = MK &nbsp;
           <span class="key">O</span> = Other &nbsp;
           <span class="key">&larr;</span><span class="key">&rarr;</span> = Navigate</p>
    </div>

    <div class="section">
        <a href="{page_prefix}_1.html" class="btn">Start Annotation</a>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAnnotations()">Export Annotations</button>
        <button class="btn" onclick="importAnnotations()">Import Annotations</button>
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <script>
        const CELL_TYPE = 'rbc';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const STORAGE_KEY = '{global_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';

        function exportAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: 'rbc',
                experiment_name: EXPERIMENT_NAME,
                exported_at: new Date().toISOString(),
                rbc: [],
                mk: [],
                other: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.rbc.push(uid);
                else if (label === 2) data.mk.push(uid);
                else if (label === 3) data.other.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'rbc_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function importAnnotations() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {{
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (ev) => {{
                    try {{
                        const data = JSON.parse(ev.target.result);
                        let imported = {{}};
                        (data.rbc || []).forEach(uid => imported[uid] = 1);
                        (data.mk || []).forEach(uid => imported[uid] = 2);
                        (data.other || []).forEach(uid => imported[uid] = 3);
                        if (data.annotations) {{
                            for (const [uid, val] of Object.entries(data.annotations)) {{
                                if (val === 'rbc' || val === 1) imported[uid] = 1;
                                else if (val === 'mk' || val === 2) imported[uid] = 2;
                                else if (val === 'other' || val === 3) imported[uid] = 3;
                            }}
                        }}
                        let existing = {{}};
                        try {{
                            const gs = localStorage.getItem(STORAGE_KEY);
                            if (gs) existing = JSON.parse(gs);
                        }} catch(ex) {{}}
                        const merged = {{...imported, ...existing}};
                        localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
                        alert('Imported ' + Object.keys(imported).length + ' annotations');
                    }} catch(err) {{
                        alert('Error importing: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations? This cannot be undone.')) return;
            localStorage.removeItem(STORAGE_KEY);
            for (let i = 1; i <= {total_pages}; i++) {{
                localStorage.removeItem(PAGE_KEY_PREFIX + i);
            }}
            alert('All annotations cleared.');
        }}
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate RBC cluster annotation HTML from unfiltered SAM detections"
    )

    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing *_full_unfiltered.json files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output HTML path (index.html will be created in the parent directory)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Maximum samples per slide (default: 200)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.03,
        help="Minimum mk_score floor (default: 0.03)"
    )
    parser.add_argument(
        "--max-score", type=float, default=0.30,
        help="Maximum mk_score ceiling (default: 0.30)"
    )
    parser.add_argument(
        "--min-area", type=float, default=100,
        help="Minimum area in um2 (default: 100)"
    )
    parser.add_argument(
        "--slides", nargs="*", default=None,
        help="Optional list of slide name substrings to include"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--samples-per-page", type=int, default=200,
        help="Number of samples per HTML page (default: 200)"
    )
    parser.add_argument(
        "--sort-by", default="mk_score", choices=["mk_score", "area_um2", "slide"],
        help="Sort samples by this key (default: mk_score)"
    )

    args = parser.parse_args()
    setup_logging(level="INFO")

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load, filter, and sample — one slide at a time to limit memory
    all_samples, slide_counts = load_and_sample_candidates(
        args.input_dir, args.min_score, args.max_score, args.min_area,
        max_samples_per_slide=args.max_samples,
        slides=args.slides, seed=args.seed,
    )

    if not all_samples:
        logger.error("No candidates found matching criteria")
        sys.exit(1)

    logger.info(f"Total sampled: {len(all_samples):,}")

    # Sort
    if args.sort_by == "mk_score":
        all_samples.sort(key=lambda x: x.get("mk_score", 0))
    elif args.sort_by == "area_um2":
        all_samples.sort(key=lambda x: x.get("area_um2", 0))
    elif args.sort_by == "slide":
        all_samples.sort(key=lambda x: x.get("slide", ""))

    # Compute stats for index page
    areas = [s.get("area_um2", 0) for s in all_samples]
    scores = [s.get("mk_score", 0) for s in all_samples]
    area_range = (min(areas), max(areas)) if areas else (0, 0)
    score_range = (min(scores), max(scores)) if scores else (0, 0)

    experiment_name = "mk_rbc_annotation"
    page_prefix = "rbc_page"

    # Paginate
    pages = [
        all_samples[i:i + args.samples_per_page]
        for i in range(0, len(all_samples), args.samples_per_page)
    ]
    total_pages = len(pages)

    logger.info(f"Generating {total_pages} HTML pages ({args.samples_per_page} per page)...")

    # Generate annotation pages
    for page_num, page_samples in enumerate(pages, 1):
        html = generate_annotation_page(
            page_samples,
            page_num=page_num,
            total_pages=total_pages,
            experiment_name=experiment_name,
            subtitle=f"mk_score in [{args.min_score:.2f}, {args.max_score:.2f}), "
                     f"area >= {args.min_area:.0f} um2",
            page_prefix=page_prefix,
        )

        page_path = output_dir / f"{page_prefix}_{page_num}.html"
        with open(page_path, 'w') as f:
            f.write(html)

        file_size = page_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

    # Generate index page
    index_html = generate_index_page(
        total_samples=len(all_samples),
        total_pages=total_pages,
        experiment_name=experiment_name,
        slide_counts=slide_counts,
        score_range=score_range,
        area_range=area_range,
        page_prefix=page_prefix,
    )

    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(index_html)

    logger.info(f"HTML exported to {output_dir}")
    logger.info(f"  {len(all_samples):,} samples across {total_pages} pages")
    logger.info(f"  Index: {index_path}")
    logger.info(f"  Serve: python -m http.server 8080 --directory {output_dir}")


if __name__ == "__main__":
    main()
