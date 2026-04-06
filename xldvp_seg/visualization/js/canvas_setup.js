// Component: canvas_setup — extracted from generate_multi_slide_spatial_viewer.py
// Panel resize, fit-to-view, and requestAnimationFrame render batching

function resizePanel(p) {
  const dpr = window.devicePixelRatio || 1;
  const rect = p.div.getBoundingClientRect();
  const w = Math.floor(rect.width);
  const h = Math.floor(rect.height);
  if (w <= 0 || h <= 0) return;
  p.cw = w;
  p.ch = h;
  p.canvas.width = w * dpr;
  p.canvas.height = h * dpr;
  p.canvas.style.width = w + 'px';
  p.canvas.style.height = h + 'px';
  p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  p.drawCanvas.width = w * dpr;
  p.drawCanvas.height = h * dpr;
  p.drawCanvas.style.width = w + 'px';
  p.drawCanvas.style.height = h + 'px';
  p.dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resizePanels() {
  panels.forEach(resizePanel);
}

function fitPanel(p) {
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const s = p.slide;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {
    p.zoom = 1;
    p.panX = cw / 2;
    p.panY = ch / 2;
    return;
  }
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2 * pad)), ch / (dataH * (1 + 2 * pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}

function scheduleRender(p) {
  rafDirty.add(p);
  if (!rafId) {
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    });
  }
}

function scheduleRenderAll() {
  panels.forEach(p => rafDirty.add(p));
  if (!rafId) {
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    });
  }
}
