// Component: init — extracted from generate_multi_slide_spatial_viewer.py
// Data decoding, state declarations, IntersectionObserver, panel initialization,
// focus-view key/button bindings, and boot sequence.
// Requires globals (data constants): SLIDE_META, SLIDE_POS_B64, SLIDE_GRP_B64,
//   GROUP_LABELS, GROUP_COLORS, N_GROUPS, IS_SINGLE, GENERATED, GROUP_FIELD,
//   TITLE, AUTO_EPS, MIN_HULL, REGION_DATA, HAS_REGIONS, HAS_MULTISCALE,
//   SCALE_KEYS, FLUOR_META, FLUOR_CH_B64, CONTOUR_DATA, HAS_FLUOR,
//   HAS_CONTOURS, CH_NAMES
// Requires functions: b64toF32, b64toU8, scheduleRender, scheduleRenderAll,
//   resizePanel, resizePanels, fitPanel, renderPanel, renderDrawOverlay,
//   handleWheel, handlePanStart, handleGlobalMouseMove, handleGlobalMouseUp,
//   screenToData, addROI, enterFocusView, exitFocusView,
//   initLegend, initControls, reclusterAll

// ===================================================================
// Decode binary data into per-slide arrays
// ===================================================================
const SLIDES = SLIDE_META.map((meta, i) => {
  const pos = b64toF32(SLIDE_POS_B64[i]);
  const grp = b64toU8(SLIDE_GRP_B64[i]);
  const rd = REGION_DATA[i] || {};
  return {
    name: meta.name,
    n: meta.n,
    xr: meta.xr,
    yr: meta.yr,
    pos: pos,  // interleaved [x0,y0,x1,y1,...] Float32Array
    grp: grp,  // group index per cell Uint8Array
    regions: rd.regions || [],
    regionScales: rd.regionScales || null,
  };
});

// Free the base64 strings to reduce memory
SLIDE_POS_B64.length = 0;
SLIDE_GRP_B64.length = 0;

// Build fluorescence image objects (decoded lazily on first render)
const fluorImages = SLIDES.map((_, si) => {
  const meta = FLUOR_META[si];
  if (!meta) return null;
  const imgs = [null, null, null];
  let loadedCount = 0;
  const result = { meta, imgs, ready: false, _canvas: null, _dirty: true };
  for (let ci = 0; ci < 3; ci++) {
    const b64 = FLUOR_CH_B64[si * 3 + ci];
    if (!b64) { loadedCount++; if (loadedCount === 3) result.ready = true; continue; }
    const img = new Image();
    img.onload = () => {
      imgs[ci] = img;
      result._dirty = true;
      loadedCount++;
      if (loadedCount === 3) {
        result.ready = true;
        // Re-render all panels once images are ready
        scheduleRenderAll();
      }
    };
    img.src = 'data:image/png;base64,' + b64;
  }
  return result;
});

// Free large base64 channel strings
FLUOR_CH_B64.length = 0;

// ===================================================================
// State
// ===================================================================
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.7;
let showHulls = false, showLabels = false;
let drawMode = 'pan';  // pan | circle | rect | polygon

// KDE state
const KDE_RADII = [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000];
let showKDE = false, kdeBWIdx = 3, kdeLevels = 3, kdeAlpha = 0.5, kdeFill = false, kdeLines = false;
const kdeCache = new Map();  // slideIdx -> {bwIdx, levels, hiddenKey, data}
let kdeDebounceTimer = null;

// Region state
let showRegions = HAS_REGIONS, showRegionLabels = HAS_REGIONS, showRegionBnd = HAS_REGIONS;
let regionAlpha = 0.25;

// Fluorescence + contour state
let showFluor = HAS_FLUOR, fluorAlpha = 0.8;
let chEnabled = [true, true, true];
let showContours = HAS_CONTOURS;
let showDots = true;

// Channel tint colors: R, G, B for additive compositing
const CH_TINTS = [[255,0,0], [0,255,0], [0,100,255]];

// Clustering state
const clusterData = new Array(SLIDES.length).fill(null);  // per-slide cluster results

// ROI storage
const rois = [];
let roiCounter = 0;
let roiFilterActive = false;

// Polygon in-progress
let polySlideIdx = -1;
let polyVerts = [];

// Drag/draw in-progress
let drawStart = null;
let drawCurrent = null;
let corridorWidth = 100;

// Panel state
const panels = [];
let activePanel = null;
let focusedIdx = -1;  // -1 = grid view, >= 0 = focused panel index

// RAF batching
let rafId = 0;
const rafDirty = new Set();

// ===================================================================
// IntersectionObserver for lazy rendering
// ===================================================================
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const idx = parseInt(entry.target.dataset.idx);
    const p = panels[idx];
    if (entry.isIntersecting) {
      p.visible = true;
      scheduleRender(p);
    } else {
      p.visible = false;
    }
  }
}, { root: document.getElementById('grid'), threshold: 0.01 });

// ===================================================================
// Panel initialization
// ===================================================================
function initPanels() {
  const grid = document.getElementById('grid');
  const select = document.getElementById('slide-select');

  SLIDES.forEach((slide, idx) => {
    const div = document.createElement('div');
    div.className = 'panel';
    div.dataset.idx = idx;

    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = slide.name;

    const countEl = document.createElement('div');
    countEl.className = 'panel-count';

    const measureEl = document.createElement('div');
    measureEl.className = 'panel-measure';

    const canvas = document.createElement('canvas');
    const drawCanvas = document.createElement('canvas');
    drawCanvas.className = 'draw-overlay';

    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(measureEl);
    div.appendChild(canvas);
    div.appendChild(drawCanvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');
    const dctx = drawCanvas.getContext('2d');

    const state = {
      div, canvas, ctx, drawCanvas, dctx, countEl, measureEl, slide, idx,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
      visible: false, cw: 0, ch: 0,
    };
    panels.push(state);
    observer.observe(div);

    // Double-click to focus (grid -> focus view)
    if (!IS_SINGLE) {
      div.addEventListener('dblclick', e => {
        if (drawMode !== 'pan') return;
        enterFocusView(idx);
        e.preventDefault();
      });
    }

    // Pan on data canvas
    canvas.addEventListener('mousedown', e => {
      handlePanStart(state, div, e);
    });

    // Drawing events on overlay canvas
    drawCanvas.addEventListener('mousedown', e => {
      if (drawMode === 'pan') return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);

      if (drawMode === 'polygon' || drawMode === 'path') {
        if (polySlideIdx !== idx) {
          polySlideIdx = idx;
          polyVerts = [];
        }
        polyVerts.push([dx, dy]);
        renderDrawOverlay(state);
      } else {
        drawStart = { x: dx, y: dy, panel: state };
        drawCurrent = { x: dx, y: dy };
      }
      e.preventDefault();
    });

    // mousemove/mouseup for draw are handled at window level (below)
    // to avoid losing events when mouse leaves the canvas during drag

    drawCanvas.addEventListener('dblclick', e => {
      if (drawMode === 'polygon' && polySlideIdx === idx && polyVerts.length >= 3) {
        addROI(idx, 'polygon', { verts: polyVerts.slice() });
      } else if (drawMode === 'path' && polySlideIdx === idx && polyVerts.length >= 2) {
        addROI(idx, 'path', { waypoints: polyVerts.slice(), corridorWidth: corridorWidth });
      } else {
        return;
      }
      polyVerts = [];
      polySlideIdx = -1;
      renderDrawOverlay(state);
      e.preventDefault();
      e.stopPropagation();
    });

    // Wheel zoom on both canvases
    canvas.addEventListener('wheel', e => handleWheel(state, div, e), { passive: false });
    drawCanvas.addEventListener('wheel', e => handleWheel(state, div, e), { passive: false });

    // Slide dropdown
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = slide.name + ' (' + slide.n.toLocaleString() + ')';
    select.appendChild(opt);
  });

  // Global mouse handlers for pan drag + draw drag
  window.addEventListener('mousemove', handleGlobalMouseMove);
  window.addEventListener('mouseup', handleGlobalMouseUp);
}

// Escape key exits focus view
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && focusedIdx >= 0) {
    exitFocusView();
  }
});

document.getElementById('focus-back').addEventListener('click', exitFocusView);

// ===================================================================
// Boot
// ===================================================================
initPanels();
initLegend();
initControls();

function fullInit() {
  resizePanels();
  panels.forEach(fitPanel);
  reclusterAll();
  scheduleRenderAll();
}

// Single-slide: go straight to focus-like rendering (full panel)
if (IS_SINGLE && panels.length === 1) {
  panels[0].visible = true;
}

setTimeout(fullInit, 80);

let _resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    if (focusedIdx >= 0) {
      resizePanel(panels[focusedIdx]);
    } else {
      resizePanels();
    }
    scheduleRenderAll();
    panels.forEach(p => renderDrawOverlay(p));
  }, 100);
});
