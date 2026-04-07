// Component: regions — extracted from generate_multi_slide_spatial_viewer.py
// Draw precomputed spatial regions (graph patterns) with labels and borders
// Requires globals: hidden

function drawRegions(ctx, regions, panZoom, opacity, showLbl, showBnd) {
  if (!regions || regions.length === 0) return;

  for (const reg of regions) {
    if (reg.bnd.length < 3) continue;
    if (reg.type && hidden.has(reg.type)) continue;

    const path = new Path2D();
    path.moveTo(reg.bnd[0][0], reg.bnd[0][1]);
    for (let i = 1; i < reg.bnd.length; i++) {
      path.lineTo(reg.bnd[i][0], reg.bnd[i][1]);
    }
    path.closePath();

    ctx.globalAlpha = opacity;
    ctx.fillStyle = reg.color;
    ctx.fill(path);

    if (showBnd) {
      ctx.globalAlpha = Math.min(opacity * 3, 0.9);
      ctx.strokeStyle = reg.color;
      ctx.lineWidth = 2.5 / panZoom;
      ctx.stroke(path);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.0 / panZoom;
      ctx.stroke(path);
    }

    if (showLbl) {
      let cx = 0, cy = 0;
      for (const pt of reg.bnd) { cx += pt[0]; cy += pt[1]; }
      cx /= reg.bnd.length;
      cy /= reg.bnd.length;

      const fontSize = 11 / panZoom;
      ctx.font = 'bold ' + fontSize + 'px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.globalAlpha = 0.9;

      const line1 = reg.label;
      const line2 = reg.n + ' cells (' + (reg.dfrac * 100).toFixed(0) + '%)';
      const lh = fontSize * 1.3;

      ctx.fillStyle = '#000';
      ctx.fillText(line1, cx + 0.8/panZoom, cy - lh/2 + 0.8/panZoom);
      ctx.fillText(line2, cx + 0.8/panZoom, cy + lh/2 + 0.8/panZoom);
      ctx.fillStyle = '#fff';
      ctx.fillText(line1, cx, cy - lh/2);
      ctx.fillText(line2, cx, cy + lh/2);
    }
  }
}
