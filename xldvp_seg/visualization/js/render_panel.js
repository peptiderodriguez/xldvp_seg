// Component: render_panel — extracted from generate_multi_slide_spatial_viewer.py
// Main panel rendering: fluorescence, regions, KDE, contours, cell dots, cluster hulls
// Requires globals: SLIDES, GROUP_LABELS, GROUP_COLORS, N_GROUPS, hidden,
//   showFluor, HAS_FLUOR, showRegions, showKDE, showContours, HAS_CONTOURS,
//   showDots, dotSize, dotAlpha, showHulls, showLabels, roiFilterActive, rois,
//   clusterData, focusedIdx, regionAlpha, showRegionLabels, showRegionBnd,
//   kdeAlpha, kdeFill, kdeLines, CONTOUR_DATA
// Requires functions: drawFluorescence, drawRegions, getKDE, drawKDEContours,
//   drawContours, cellPassesROIFilter

function renderPanel(p) {
  if (!p.visible && focusedIdx !== p.idx) return;
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const ctx = p.ctx;
  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);
  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  // Layer 0: Fluorescence background
  if (showFluor && HAS_FLUOR) {
    drawFluorescence(ctx, p.idx, p.zoom);
  }

  // Layer 1: Regions (lowest)
  if (showRegions && p.slide.regions && p.slide.regions.length > 0) {
    drawRegions(ctx, p.slide.regions, p.zoom, regionAlpha, showRegionLabels, showRegionBnd);
  }

  // Layer 2: KDE contours
  if (showKDE) {
    const kdeData = getKDE(p.idx);
    drawKDEContours(ctx, kdeData, p.zoom, kdeAlpha, kdeFill, kdeLines);
  }

  // Layer 2.5: Detection contours
  if (showContours && HAS_CONTOURS) {
    drawContours(ctx, p, p.zoom);
  }

  // Layer 3: Cell dots
  const r = dotSize / p.zoom;
  const halfR = r / 2;
  const slide = p.slide;
  const pos = slide.pos;
  const grp = slide.grp;
  const n = slide.n;
  let total = 0;

  const useROIFilter = roiFilterActive && rois.length > 0;

  if (showDots) {
    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) continue;
      ctx.globalAlpha = dotAlpha;
      ctx.fillStyle = GROUP_COLORS[gi];

      for (let i = 0; i < n; i++) {
        if (grp[i] !== gi) continue;
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        if (useROIFilter && !cellPassesROIFilter(x, y, p.idx)) continue;
        ctx.fillRect(x - halfR, y - halfR, r, r);
        total++;
      }
    }
  } else {
    // Count visible cells even when dots are hidden
    const useFilter = roiFilterActive && rois.length > 0;
    for (let i = 0; i < n; i++) {
      const gi = grp[i];
      if (hidden.has(GROUP_LABELS[gi])) continue;
      if (useFilter && !cellPassesROIFilter(pos[i*2], pos[i*2+1], p.idx)) continue;
      total++;
    }
  }

  // Layer 4: Cluster hulls (top)
  const sc = clusterData[p.idx];
  if (sc) {
    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) continue;
      const groupClusters = sc[gi];
      if (!groupClusters) continue;

      for (const cl of groupClusters) {
        if (showHulls && cl.hull && cl.hull.length >= 3) {
          ctx.globalAlpha = 1;
          const path = new Path2D();
          path.moveTo(cl.hull[0][0], cl.hull[0][1]);
          for (let i = 1; i < cl.hull.length; i++) {
            path.lineTo(cl.hull[i][0], cl.hull[i][1]);
          }
          path.closePath();

          ctx.setLineDash([6/p.zoom, 4/p.zoom]);
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 2.5 / p.zoom;
          ctx.stroke(path);
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1.2 / p.zoom;
          ctx.stroke(path);
          ctx.setLineDash([]);
        }

        if (showLabels && cl.hull && cl.hull.length >= 3) {
          ctx.globalAlpha = 1;
          const fontSize = 11 / p.zoom;
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const line1 = cl.n + ' cells';
          ctx.fillStyle = '#000';
          ctx.fillText(line1, cl.cx + 0.5/p.zoom, cl.cy + 0.5/p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(line1, cl.cx, cl.cy);
        }
      }
    }
  }

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}
