"""JavaScript generators for HTML annotation pages.

Extracted from ``html_export.py``. Contains:

**get_js():** Unified JS for standard annotation pages — localStorage,
    keyboard navigation, contour/channel toggling, export/import.
**get_vessel_js():** Enhanced JS for vessel annotation — batch selection,
    feature filtering, RF training export (CSV + sklearn JSON).
**generate_preload_annotations_js():** Pre-load prior annotations into
    localStorage for round-2 annotation after classifier training.
"""

import json
from pathlib import Path

from xldvp_seg.io.html_utils import _esc, _js_esc


def generate_preload_annotations_js(
    annotations_path: str, cell_type: str, experiment_name: str = None
) -> str:
    """
    Generate JavaScript that pre-loads prior annotations into localStorage.

    This is used when regenerating HTML after classifier training, so that
    the user's round-1 annotations are visible alongside the classifier's
    new predictions.

    Args:
        annotations_path: Path to annotations JSON file (exported from HTML viewer)
        cell_type: Cell type identifier (e.g., 'nmj', 'mk')
        experiment_name: Optional experiment name for localStorage key isolation.
            Must match the experiment_name used in the page JS.

    Returns:
        JavaScript code string to write to preload_annotations.js
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        return None

    with open(annotations_path) as f:
        data = json.load(f)

    # Convert from export format {positive: [...], negative: [...]}
    # to localStorage format {uid: 1, uid: 0}
    ls_format = {}
    for uid in data.get("positive", []):
        ls_format[uid] = 1
    for uid in data.get("negative", []):
        ls_format[uid] = 0
    for uid in data.get("unsure", []):
        ls_format[uid] = 2

    # Also handle the alternative format {annotations: {uid: "yes", uid: "no"}}
    if "annotations" in data:
        for uid, label in data["annotations"].items():
            if label == "yes":
                ls_format[uid] = 1
            elif label == "no":
                ls_format[uid] = 0
            elif label == "unsure":
                ls_format[uid] = 2

    if not ls_format:
        return None

    if experiment_name:
        global_key = _js_esc(f"{cell_type}_{experiment_name}_annotations")
        page_key_prefix = _js_esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        global_key = _js_esc(f"{cell_type}_annotations")
        page_key_prefix = _js_esc(f"{cell_type}_labels_page")

    # Escape </ sequences to prevent </script> injection in inline JS
    safe_json = json.dumps(ls_format).replace("</", r"<\/").replace("<!--", r"<\!--")
    js_content = f"""// Pre-loaded annotations from {_esc(annotations_path.name)}
// Generated automatically during HTML export
// These are EXISTING annotations - new annotations take precedence

const PRELOADED_ANNOTATIONS = {safe_json};

// Merge: preloaded as base, existing localStorage on top (so new annotations aren't overwritten)
// Write to BOTH the global key and any existing page-specific keys
(function() {{
    try {{
        // Merge into global key
        let existingGlobal = {{}};
        const savedGlobal = localStorage.getItem('{global_key}');
        if (savedGlobal) existingGlobal = JSON.parse(savedGlobal);
        const mergedGlobal = {{...PRELOADED_ANNOTATIONS, ...existingGlobal}};
        localStorage.setItem('{global_key}', JSON.stringify(mergedGlobal));

        // Also merge into any existing page-specific keys for this cell type
        for (let i = 0; i < localStorage.length; i++) {{
            const key = localStorage.key(i);
            if (key && key.startsWith('{page_key_prefix}')) {{
                let existingPage = {{}};
                try {{ existingPage = JSON.parse(localStorage.getItem(key)); }} catch(e2) {{}}
                const mergedPage = {{...PRELOADED_ANNOTATIONS, ...existingPage}};
                localStorage.setItem(key, JSON.stringify(mergedPage));
            }}
        }}

        const preloadedCount = Object.keys(PRELOADED_ANNOTATIONS).length;
        const existingCount = Object.keys(existingGlobal).length;
        const newFromPreload = Object.keys(mergedGlobal).length - existingCount;

        if (newFromPreload > 0) {{
            console.log('Loaded ' + newFromPreload + ' annotations from preload file (' + preloadedCount + ' total in file)');
        }}
    }} catch(e) {{ console.error('Failed to load annotations:', e); }}
}})();
"""
    return js_content


def get_js(cell_type, total_pages, experiment_name=None, page_num=1):
    """
    Get the unified JavaScript for annotation handling.

    Uses BOTH a page-specific localStorage key and a global key so that
    annotations are visible from both single-page and cross-page views.

    Args:
        cell_type: Type identifier (e.g., 'nmj', 'mk', 'hspc')
        total_pages: Total number of pages
        experiment_name: Optional experiment name for localStorage key isolation
                        If provided, global key is '{cell_type}_{experiment_name}_annotations'
                        Otherwise, global key is '{cell_type}_annotations'
        page_num: Current page number (1-indexed) for page-specific key

    Returns:
        JavaScript code string
    """
    # Build storage keys: both global and page-specific
    if experiment_name:
        global_key = _js_esc(f"{cell_type}_{experiment_name}_annotations")
        page_key = _js_esc(f"{cell_type}_{experiment_name}_labels_page{page_num}")
    else:
        global_key = _js_esc(f"{cell_type}_annotations")
        page_key = _js_esc(f"{cell_type}_labels_page{page_num}")
    cell_type_safe = _js_esc(cell_type)
    experiment_name_safe = _js_esc(experiment_name or "")
    if experiment_name:
        page_key_prefix = _js_esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        page_key_prefix = _js_esc(f"{cell_type}_labels_page")

    return f"""
        const CELL_TYPE = '{cell_type_safe}';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const TOTAL_PAGES = {total_pages};
        const GLOBAL_STORAGE_KEY = '{global_key}';
        const PAGE_STORAGE_KEY = '{page_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';

        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        // Save to BOTH page-specific and global localStorage keys
        function saveLabels() {{
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            // Merge into global store instead of overwriting
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            // Remove toggled-off annotations: delete page UIDs not in labels
            cards.forEach(card => {{ if (!(card.id in labels)) delete globalLabels[card.id]; }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        // Load from localStorage: page-specific first, then global fallback
        function loadAnnotations() {{
            try {{
                let saved = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!saved) {{
                    // Fallback: load only UIDs present on this page from global
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

            // Apply to cards
            cards.forEach((card, i) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            card.dataset.label = label;
            if (label === 1) card.classList.add('labeled-yes');
            else if (label === 0) card.classList.add('labeled-no');
            else if (label === 2) card.classList.add('labeled-unsure');
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            // Toggle off if same label
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
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
            let localYes = 0, localNo = 0, localUnsure = 0;
            let globalYes = 0, globalNo = 0;

            // Count current page
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] === 1) localYes++;
                else if (labels[uid] === 0) localNo++;
                else if (labels[uid] === 2) localUnsure++;
            }});

            // Count global from the global key (contains all annotations)
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === 1) globalYes++;
                        else if (v === 0) globalNo++;
                    }}
                }}
            }} catch(e) {{ console.error(e); }}

            const localYesEl = document.getElementById('localYes');
            const localNoEl = document.getElementById('localNo');
            const globalYesEl = document.getElementById('globalYes');
            const globalNoEl = document.getElementById('globalNo');

            if (localYesEl) localYesEl.textContent = localYes;
            if (localNoEl) localNoEl.textContent = localNo;
            if (globalYesEl) globalYesEl.textContent = globalYes;
            if (globalNoEl) globalNoEl.textContent = globalNo;
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
            // Export from global key to get all annotations across pages
            let allLabels = {{}};
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) allLabels = JSON.parse(globalSaved);
            }} catch(e) {{ console.error(e); }}

            const data = {{
                cell_type: CELL_TYPE,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(allLabels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json' : CELL_TYPE + '_annotations.json';
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
                        if (data.annotations) {{
                            for (const [uid, val] of Object.entries(data.annotations)) {{
                                if (val === 'yes' || val === 1) imported[uid] = 1;
                                else if (val === 'no' || val === 0) imported[uid] = 0;
                                else if (val === 'unsure' || val === 2) imported[uid] = 2;
                            }}
                        }} else {{
                            (data.positive || []).forEach(uid => imported[uid] = 1);
                            (data.negative || []).forEach(uid => imported[uid] = 0);
                            (data.unsure || []).forEach(uid => imported[uid] = 2);
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
                            if (labels[uid] !== undefined) {{
                                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                                if (labels[uid] === 1) card.classList.add('labeled-yes');
                                else if (labels[uid] === 0) card.classList.add('labeled-no');
                                else if (labels[uid] === 2) card.classList.add('labeled-unsure');
                                card.dataset.label = labels[uid];
                            }}
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
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
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
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
        }}

        let contoursVisible = true;
        function toggleContours() {{
            contoursVisible = !contoursVisible;
            const btn = document.getElementById('toggleContourBtn');
            document.querySelectorAll('.img-contour').forEach(img => {{
                img.style.display = contoursVisible ? '' : 'none';
            }});
            if (btn) {{
                btn.style.background = contoursVisible ? '#2a5a2a' : '#555';
                btn.style.opacity = contoursVisible ? '1' : '0.5';
            }}
        }}

        const chState = {{ r: true, g: true, b: true }};
        const chColors = {{ r: '#8b2222', g: '#228b22', b: '#22228b' }};
        const chBtnIds = {{ r: 'toggleChRBtn', g: 'toggleChGBtn', b: 'toggleChBBtn' }};
        function toggleChannel(ch) {{
            chState[ch] = !chState[ch];
            const btn = document.getElementById(chBtnIds[ch]);
            if (btn) {{
                btn.style.background = chState[ch] ? chColors[ch] : '#555';
                btn.style.opacity = chState[ch] ? '1' : '0.5';
            }}
            // Build filter ID from off-channels
            let off = '';
            if (!chState.r) off += 'r';
            if (!chState.g) off += 'g';
            if (!chState.b) off += 'b';
            const filterVal = off ? 'url(#no-' + off + ')' : 'none';
            document.querySelectorAll('.img-base').forEach(img => {{
                img.style.filter = filterVal;
            }});
        }}

        document.addEventListener('keydown', (e) => {{
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
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
            }}
        }});

        // Initialize
        loadAnnotations();
    """


def get_vessel_js(cell_type, total_pages, experiment_name=None, all_features_json="{}", page_num=1):
    """
    Get enhanced JavaScript for vessel annotation with RF training support.

    Args:
        cell_type: Type identifier (e.g., 'vessel')
        total_pages: Total number of pages
        experiment_name: Optional experiment name for localStorage key isolation
        all_features_json: JSON string of all vessel features for filtering/export

    Returns:
        JavaScript code string
    """
    # Build storage keys: both global and page-specific
    if experiment_name:
        global_key = _js_esc(f"{cell_type}_{experiment_name}_annotations")
        page_key = _js_esc(f"{cell_type}_{experiment_name}_labels_page{page_num}")
    else:
        global_key = _js_esc(f"{cell_type}_annotations")
        page_key = _js_esc(f"{cell_type}_labels_page{page_num}")
    cell_type_safe = _js_esc(cell_type)
    experiment_name_safe = _js_esc(experiment_name or "")
    if experiment_name:
        page_key_prefix = _js_esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        page_key_prefix = _js_esc(f"{cell_type}_labels_page")

    return f"""
        const CELL_TYPE = '{cell_type_safe}';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const TOTAL_PAGES = {total_pages};
        const GLOBAL_STORAGE_KEY = '{global_key}';
        const PAGE_STORAGE_KEY = '{page_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';
        const ALL_FEATURES = {all_features_json};

        let labels = {{}};
        let selectedIdx = -1;
        let batchSelected = new Set();
        const cards = document.querySelectorAll('.card');

        // Save to BOTH page-specific and global localStorage keys
        function saveLabels() {{
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            // Merge into global store instead of overwriting
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            // Remove toggled-off annotations: delete page UIDs not in labels
            cards.forEach(card => {{ if (!(card.id in labels)) delete globalLabels[card.id]; }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        // Load from localStorage: page-specific first, then global fallback
        function loadAnnotations() {{
            try {{
                let saved = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!saved) {{
                    // Fallback: load only UIDs present on this page from global
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

            // Apply to cards
            cards.forEach((card, i) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            card.dataset.label = label;
            if (label === 1) card.classList.add('labeled-yes');
            else if (label === 0) card.classList.add('labeled-no');
            else if (label === 2) card.classList.add('labeled-unsure');
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            // Toggle off if same label
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }} else {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }}

            saveLabels();
            updateStats();

            if (autoAdvance && selectedIdx >= 0) {{
                // Find next visible card
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx >= 0 && currentVisibleIdx < visibleCards.length - 1) {{
                    const nextCard = visibleCards[currentVisibleIdx + 1];
                    const nextIdx = Array.from(cards).indexOf(nextCard);
                    selectCard(nextIdx);
                }}
            }}
        }}

        function updateStats() {{
            let localYes = 0, localNo = 0, localUnsure = 0, localTotal = 0;
            let globalYes = 0, globalNo = 0, globalTotal = 0;

            // Count current page (visible only)
            cards.forEach(card => {{
                if (!card.classList.contains('hidden')) {{
                    localTotal++;
                    const uid = card.id;
                    if (labels[uid] === 1) localYes++;
                    else if (labels[uid] === 0) localNo++;
                    else if (labels[uid] === 2) localUnsure++;
                }}
            }});

            // Count global from the global key
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === 1) globalYes++;
                        else if (v === 0) globalNo++;
                        globalTotal++;
                    }}
                }}
            }} catch(e) {{ console.error(e); }}

            const localRemaining = localTotal - localYes - localNo - localUnsure;

            document.getElementById('localYes').textContent = localYes;
            document.getElementById('localNo').textContent = localNo;
            document.getElementById('localRemaining').textContent = localRemaining;
            document.getElementById('localTotal').textContent = localTotal;
            document.getElementById('globalYes').textContent = globalYes;
            document.getElementById('globalNo').textContent = globalNo;
            document.getElementById('globalTotal').textContent = globalTotal;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        // Batch selection functions
        function toggleBatchSelect(uid) {{
            const card = document.getElementById(uid);
            if (batchSelected.has(uid)) {{
                batchSelected.delete(uid);
                card.classList.remove('batch-selected');
            }} else {{
                batchSelected.add(uid);
                card.classList.add('batch-selected');
            }}
            updateBatchToolbar();
        }}

        function updateBatchToolbar() {{
            const toolbar = document.getElementById('batchToolbar');
            const countEl = document.getElementById('batchCount');
            if (batchSelected.size > 0) {{
                toolbar.classList.remove('hidden');
                countEl.textContent = batchSelected.size + ' selected';
            }} else {{
                toolbar.classList.add('hidden');
            }}
        }}

        function batchLabel(label) {{
            batchSelected.forEach(uid => {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }});
            saveLabels();
            clearBatchSelection();
            updateStats();
        }}

        function clearBatchSelection() {{
            batchSelected.forEach(uid => {{
                const card = document.getElementById(uid);
                if (card) card.classList.remove('batch-selected');
            }});
            batchSelected.clear();
            updateBatchToolbar();
        }}

        function selectAllVisible() {{
            cards.forEach(card => {{
                if (!card.classList.contains('hidden')) {{
                    batchSelected.add(card.id);
                    card.classList.add('batch-selected');
                }}
            }});
            updateBatchToolbar();
        }}

        function selectUnannotated() {{
            cards.forEach(card => {{
                if (!card.classList.contains('hidden') && labels[card.id] === undefined) {{
                    batchSelected.add(card.id);
                    card.classList.add('batch-selected');
                }}
            }});
            updateBatchToolbar();
        }}

        // Filtering functions
        function applyFilters() {{
            const minDiam = parseFloat(document.getElementById('filterMinDiam').value) || 0;
            const maxDiam = parseFloat(document.getElementById('filterMaxDiam').value) || 9999;
            const confidence = document.getElementById('filterConfidence').value;
            const showAnnotated = document.getElementById('filterShowAnnotated').checked;
            const showUnannotated = document.getElementById('filterShowUnannotated').checked;

            let visibleCount = 0;
            cards.forEach(card => {{
                const uid = card.id;
                const feat = ALL_FEATURES[uid] || {{}};
                const diam = feat.outer_diameter_um || 0;
                const conf = feat.confidence || 'unknown';
                const isAnnotated = labels[uid] !== undefined;

                let visible = true;

                // Diameter filter
                if (diam < minDiam || diam > maxDiam) visible = false;

                // Confidence filter
                if (confidence !== 'all' && conf !== confidence) visible = false;

                // Annotation status filter
                if (isAnnotated && !showAnnotated) visible = false;
                if (!isAnnotated && !showUnannotated) visible = false;

                if (visible) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});

            updateStats();
            clearBatchSelection();
        }}

        function resetFilters() {{
            document.getElementById('filterMinDiam').value = '';
            document.getElementById('filterMaxDiam').value = '';
            document.getElementById('filterConfidence').value = 'all';
            document.getElementById('filterShowAnnotated').checked = true;
            document.getElementById('filterShowUnannotated').checked = true;

            cards.forEach(card => card.classList.remove('hidden'));
            updateStats();
        }}

        // Export functions
        function exportAnnotationsJSON() {{
            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                total_annotations: Object.keys(labels).length,
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json' : CELL_TYPE + '_annotations.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportForRF() {{
            // Export annotations with features for Random Forest training
            const rows = [];
            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            // Header
            const header = ['uid', 'annotation'].concat(featureKeys);
            rows.push(header.join(','));

            // Data rows
            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1 || label === 0) {{  // Only yes/no, skip unsure
                    const feat = ALL_FEATURES[uid] || {{}};
                    const row = [uid, label === 1 ? 'yes' : 'no'];
                    featureKeys.forEach(k => {{
                        const val = feat[k];
                        row.push(typeof val === 'number' ? val : '');
                    }});
                    rows.push(row.join(','));
                }}
            }}

            const csv = rows.join('\\n');
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_rf_training.csv' : CELL_TYPE + '_rf_training.csv';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportRFJSON() {{
            // Export in scikit-learn compatible format
            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            const X = [];
            const y = [];
            const uids = [];

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1 || label === 0) {{
                    const feat = ALL_FEATURES[uid] || {{}};
                    const row = featureKeys.map(k => feat[k] || 0);
                    X.push(row);
                    y.push(label);
                    uids.push(uid);
                }}
            }}

            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                feature_names: featureKeys,
                X: X,
                y: y,
                uids: uids,
                n_positive: y.filter(v => v === 1).length,
                n_negative: y.filter(v => v === 0).length
            }};

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_rf_sklearn.json' : CELL_TYPE + '_rf_sklearn.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function previewExport() {{
            const modal = document.getElementById('exportModal');
            const preview = document.getElementById('exportPreview');

            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            let posCount = 0, negCount = 0;
            for (const label of Object.values(labels)) {{
                if (label === 1) posCount++;
                else if (label === 0) negCount++;
            }}

            const summary = `Export Summary
==============
Cell Type: ${{CELL_TYPE}}
Experiment: ${{EXPERIMENT_NAME || 'N/A'}}

Annotations:
  - Positive (yes): ${{posCount}}
  - Negative (no): ${{negCount}}
  - Total for training: ${{posCount + negCount}}

Features: ${{featureKeys.length}} numeric features
${{featureKeys.slice(0, 20).join('\\n')}}
${{featureKeys.length > 20 ? '... and ' + (featureKeys.length - 20) + ' more' : ''}}

Export Formats:
  1. JSON (simple) - positive/negative/unsure lists
  2. CSV (RF) - uid, annotation, all features
  3. JSON (sklearn) - X matrix, y vector, feature names`;

            preview.textContent = summary;
            modal.classList.add('show');
        }}

        function closeModal() {{
            document.getElementById('exportModal').classList.remove('show');
        }}

        function clearPage() {{
            if (!confirm('Clear annotations on this page?')) return;
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    delete labels[uid];
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
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
            // Also clear all other page-specific keys for this experiment
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                localStorage.setItem(PAGE_KEY_PREFIX + i, JSON.stringify({{}}));
            }}
            cards.forEach(card => {{
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // Navigation
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                e.preventDefault();
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx < visibleCards.length - 1) {{
                    const nextCard = visibleCards[currentVisibleIdx + 1];
                    selectCard(Array.from(cards).indexOf(nextCard));
                }}
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                e.preventDefault();
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx > 0) {{
                    const prevCard = visibleCards[currentVisibleIdx - 1];
                    selectCard(Array.from(cards).indexOf(prevCard));
                }}
            }}
            // Labeling
            else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
                else if (e.key === ' ') {{
                    e.preventDefault();
                    toggleBatchSelect(uid);
                }}
            }}
            // Escape to close modal or clear selection
            if (e.key === 'Escape') {{
                closeModal();
                clearBatchSelection();
            }}
        }});

        // Initialize
        loadAnnotations();
    """
