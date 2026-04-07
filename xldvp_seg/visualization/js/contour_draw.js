// Component: contour_draw — extracted from generate_multi_slide_spatial_viewer.py
// Detection contour rendering with Path2D and viewport culling
// Requires globals: CONTOUR_DATA (array of per-slide contour arrays), showContours

function drawContours(ctx, p, panZoom) {
  const contours = CONTOUR_DATA[p.idx];
  if (!contours || contours.length === 0) return;

  // Compute visible data bounds (in um) for viewport culling
  // Panel transform: screen = data * zoom + pan => data = (screen - pan) / zoom
  const vx1 = (0 - p.panX) / p.zoom;
  const vy1 = (0 - p.panY) / p.zoom;
  const vx2 = (p.cw - p.panX) / p.zoom;
  const vy2 = (p.ch - p.panY) / p.zoom;

  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 1.5 / panZoom;
  ctx.setLineDash([6 / panZoom, 4 / panZoom]);
  ctx.globalAlpha = 0.85;

  for (let ci = 0; ci < contours.length; ci++) {
    const c = contours[ci];
    // Viewport culling via pre-computed bounding box
    if (c.bx2 < vx1 || c.bx1 > vx2 || c.by2 < vy1 || c.by1 > vy2) continue;

    const pts = c.pts;
    if (!pts || pts.length < 6) continue;  // at least 3 points (6 floats)

    const path = new Path2D();
    path.moveTo(pts[0], pts[1]);
    for (let j = 2; j < pts.length; j += 2) {
      path.lineTo(pts[j], pts[j + 1]);
    }
    path.closePath();
    ctx.stroke(path);
  }

  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}
