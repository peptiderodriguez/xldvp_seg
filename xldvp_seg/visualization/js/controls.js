// Component: controls — extracted from generate_multi_slide_spatial_viewer.py
// Legend initialization, recluster, and all sidebar control bindings
// Requires globals: SLIDES, GROUP_LABELS, GROUP_COLORS, N_GROUPS, AUTO_EPS,
//   MIN_HULL, KDE_RADII, SCALE_KEYS, hidden, panels, clusterData,
//   dotSize, dotAlpha, showHulls, showLabels, drawMode, drawStart, drawCurrent,
//   polyVerts, polySlideIdx, roiFilterActive, corridorWidth, rois, focusedIdx,
//   showKDE, kdeBWIdx, kdeLevels, kdeAlpha, kdeFill, kdeLines, kdeCache,
//   showRegions, showRegionLabels, showRegionBnd, regionAlpha,
//   showFluor, fluorAlpha, chEnabled, fluorImages, showContours, showDots
// Requires functions: scheduleRenderAll, resizePanels, fitPanel, renderDrawOverlay,
//   dbscan, convexHull, getGroupPositions, downloadROIs, enterFocusView,
//   exitFocusView, kdeDebounceTimer (state)

function reclusterAll() {
  const mult = parseFloat(document.getElementById('eps-slider').value);
  const minCells = parseInt(document.getElementById('min-cells').value);
  let totalClusters = 0, totalHulls = 0;
  let epsMin = Infinity, epsMax = 0;
  const t0 = performance.now();

  for (let si = 0; si < SLIDES.length; si++) {
    const gpos = getGroupPositions(si);
    const slideClusters = [];

    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) { slideClusters.push([]); continue; }
      const gp = gpos[gi];
      if (gp.n === 0) { slideClusters.push([]); continue; }

      const eps = AUTO_EPS[si][gi] * mult;
      if (eps < epsMin) epsMin = eps;
      if (eps > epsMax) epsMax = eps;
      const labels = dbscan(gp.x, gp.y, gp.n, eps, minCells);

      const clusterMap = new Map();
      for (let i = 0; i < gp.n; i++) {
        const cl = labels[i];
        if (cl === -1) continue;
        let arr = clusterMap.get(cl);
        if (!arr) { arr = []; clusterMap.set(cl, arr); }
        arr.push(i);
      }

      const groupClusters = [];
      let num = 0;
      for (const [clId, indices] of clusterMap) {
        num++;
        totalClusters++;
        const pts = [];
        let sx = 0, sy = 0;
        for (const idx of indices) {
          const px = gp.x[idx], py = gp.y[idx];
          pts.push([px, py]);
          sx += px; sy += py;
        }
        const cx = sx / indices.length;
        const cy = sy / indices.length;

        let hull = [];
        if (indices.length >= MIN_HULL) {
          hull = convexHull(pts);
          if (hull.length >= 3) totalHulls++;
          else hull = [];
        }

        groupClusters.push({
          label: GROUP_LABELS[gi] + ' #' + num,
          n: indices.length,
          hull: hull,
          cx: cx,
          cy: cy,
        });
      }
      slideClusters.push(groupClusters);
    }
    clusterData[si] = slideClusters;
  }

  const dt = (performance.now() - t0).toFixed(0);
  const epsRange = epsMin === Infinity ? '' :
    ' | eps ' + Math.round(epsMin) + '-' + Math.round(epsMax) + ' um';
  document.getElementById('cluster-status').textContent =
    totalClusters + ' clusters (' + totalHulls + ' hulls) ' + dt + 'ms' + epsRange;
}

function initLegend() {
  const legDiv = document.getElementById('leg-items');
  // Compute total counts per group
  const totals = new Array(N_GROUPS).fill(0);
  for (const slide of SLIDES) {
    for (let i = 0; i < slide.n; i++) {
      totals[slide.grp[i]]++;
    }
  }

  for (let gi = 0; gi < N_GROUPS; gi++) {
    const item = document.createElement('div');
    item.className = 'leg-item';
    item.dataset.gi = gi;

    const dot = document.createElement('span');
    dot.className = 'leg-dot';
    dot.style.background = GROUP_COLORS[gi];

    const label = document.createElement('span');
    label.className = 'leg-label';
    label.title = GROUP_LABELS[gi];
    label.textContent = GROUP_LABELS[gi];

    const count = document.createElement('span');
    count.className = 'leg-count';
    count.textContent = totals[gi].toLocaleString();

    item.appendChild(dot);
    item.appendChild(label);
    item.appendChild(count);

    item.onclick = () => {
      const lbl = GROUP_LABELS[gi];
      if (hidden.has(lbl)) {
        hidden.delete(lbl);
        item.classList.remove('hidden');
      } else {
        hidden.add(lbl);
        item.classList.add('hidden');
      }
      kdeCache.clear();
      reclusterAll();
      scheduleRenderAll();
    };
    legDiv.appendChild(item);
  }
}

function initControls() {
  // Dot size
  document.getElementById('dot-size').oninput = e => {
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  };

  // Opacity
  document.getElementById('opacity').oninput = e => {
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  };

  // Show all / hide all
  document.getElementById('btn-show-all').onclick = () => {
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    kdeCache.clear();
    reclusterAll();
    scheduleRenderAll();
  };
  document.getElementById('btn-hide-all').onclick = () => {
    GROUP_LABELS.forEach(l => hidden.add(l));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    kdeCache.clear();
    reclusterAll();
    scheduleRenderAll();
  };

  // Reset zoom (sidebar button + floating button)
  const resetZoomFn = () => {
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
    panels.forEach(p => renderDrawOverlay(p));
  };
  document.getElementById('btn-reset-zoom').onclick = resetZoomFn;
  document.getElementById('floating-reset-zoom').onclick = resetZoomFn;

  // Slide jump
  document.getElementById('slide-select').onchange = e => {
    const idx = parseInt(e.target.value);
    if (isNaN(idx) || !panels[idx]) return;
    if (focusedIdx >= 0) {
      // In focus view: switch to this slide
      exitFocusView();
      setTimeout(() => enterFocusView(idx), 100);
    } else {
      panels[idx].div.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Draw mode buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.onclick = () => {
      drawMode = btn.dataset.mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      panels.forEach(p => {
        if (drawMode === 'pan') {
          p.div.classList.remove('draw-mode');
        } else {
          p.div.classList.add('draw-mode');
        }
      });
      // Clear in-progress drawing
      drawStart = null;
      drawCurrent = null;
      if (drawMode !== 'polygon' && drawMode !== 'path') {
        polyVerts = [];
        polySlideIdx = -1;
      }
      panels.forEach(p => {
        p.measureEl.style.display = 'none';
        renderDrawOverlay(p);
      });
    };
  });

  // ROI controls
  document.getElementById('btn-download-roi').onclick = downloadROIs;
  document.getElementById('roi-filter').onchange = e => {
    roiFilterActive = e.target.checked;
    scheduleRenderAll();
  };
  document.getElementById('corridor-slider').oninput = e => {
    corridorWidth = parseFloat(e.target.value);
    document.getElementById('corridor-val').textContent = corridorWidth;
    rois.forEach(r => { if (r.type === 'path') r.data.corridorWidth = corridorWidth; });
    panels.forEach(p => renderDrawOverlay(p));
    updateROIStats();
    if (roiFilterActive) scheduleRenderAll();
  };

  // Clustering controls
  const epsSlider = document.getElementById('eps-slider');
  const minCellsSlider = document.getElementById('min-cells');
  epsSlider.oninput = e => {
    document.getElementById('eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  };
  epsSlider.onchange = () => { reclusterAll(); scheduleRenderAll(); };
  minCellsSlider.oninput = e => {
    document.getElementById('min-cells-val').textContent = e.target.value;
  };
  minCellsSlider.onchange = () => { reclusterAll(); scheduleRenderAll(); };
  document.getElementById('show-hulls').onchange = e => {
    showHulls = e.target.checked;
    scheduleRenderAll();
  };
  document.getElementById('show-labels').onchange = e => {
    showLabels = e.target.checked;
    scheduleRenderAll();
  };

  // KDE controls
  const showKDECb = document.getElementById('show-kde');
  if (showKDECb) {
    showKDECb.onchange = e => { showKDE = e.target.checked; scheduleRenderAll(); };

    function kdeDebounced() {
      if (kdeDebounceTimer) clearTimeout(kdeDebounceTimer);
      kdeDebounceTimer = setTimeout(() => { kdeCache.clear(); scheduleRenderAll(); }, 200);
    }

    document.getElementById('kde-bw').oninput = e => {
      kdeBWIdx = parseInt(e.target.value);
      document.getElementById('kde-bw-val').textContent = KDE_RADII[kdeBWIdx] + ' \u00b5m';
      kdeDebounced();
    };
    document.getElementById('kde-levels').oninput = e => {
      kdeLevels = parseInt(e.target.value);
      document.getElementById('kde-levels-val').textContent = kdeLevels;
      kdeDebounced();
    };
    document.getElementById('kde-opacity').oninput = e => {
      kdeAlpha = parseFloat(e.target.value);
      document.getElementById('kde-op-val').textContent = kdeAlpha.toFixed(2);
      scheduleRenderAll();
    };
    document.getElementById('kde-fill').onchange = e => { kdeFill = e.target.checked; scheduleRenderAll(); };
    document.getElementById('kde-lines').onchange = e => { kdeLines = e.target.checked; scheduleRenderAll(); };
  }

  // Region controls
  const showRegCb = document.getElementById('show-regions');
  if (showRegCb) {
    showRegCb.onchange = e => { showRegions = e.target.checked; scheduleRenderAll(); };
    document.getElementById('show-region-labels').onchange = e => { showRegionLabels = e.target.checked; scheduleRenderAll(); };
    document.getElementById('show-region-bnd').onchange = e => { showRegionBnd = e.target.checked; scheduleRenderAll(); };
    document.getElementById('region-opacity').oninput = e => {
      regionAlpha = parseFloat(e.target.value);
      document.getElementById('region-op-val').textContent = regionAlpha.toFixed(2);
      scheduleRenderAll();
    };

    // Scale slider (multi-scale regions)
    const scaleSlider = document.getElementById('region-scale');
    if (scaleSlider) {
      scaleSlider.oninput = e => {
        const idx = parseInt(e.target.value);
        const key = String(SCALE_KEYS[idx]);
        document.getElementById('region-scale-val').textContent = key + ' \u00b5m';
        for (const slide of SLIDES) {
          if (slide.regionScales && slide.regionScales[key]) {
            slide.regions = slide.regionScales[key];
          }
        }
        scheduleRenderAll();
      };
    }
  }

  // Fluorescence controls
  const showFluorCb = document.getElementById('show-fluor');
  if (showFluorCb) {
    showFluorCb.onchange = e => { showFluor = e.target.checked; scheduleRenderAll(); };
  }
  const fluorOpSlider = document.getElementById('fluor-opacity');
  if (fluorOpSlider) {
    fluorOpSlider.oninput = e => {
      fluorAlpha = parseFloat(e.target.value);
      document.getElementById('fluor-op-val').textContent = fluorAlpha.toFixed(2);
      scheduleRenderAll();
    };
  }
  for (let ci = 0; ci < 3; ci++) {
    const btn = document.getElementById('btn-ch' + ci);
    if (btn) {
      btn.onclick = () => {
        chEnabled[ci] = !chEnabled[ci];
        btn.classList.toggle('active', chEnabled[ci]);
        // Invalidate composited canvas for all slides
        fluorImages.forEach(fd => { if (fd) fd._dirty = true; });
        scheduleRenderAll();
      };
    }
  }
  const showContoursCb = document.getElementById('show-contours');
  if (showContoursCb) {
    showContoursCb.onchange = e => { showContours = e.target.checked; scheduleRenderAll(); };
  }
  const showDotsCb = document.getElementById('show-dots');
  if (showDotsCb) {
    showDotsCb.onchange = e => { showDots = e.target.checked; scheduleRenderAll(); };
  }
}
