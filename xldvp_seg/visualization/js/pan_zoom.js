// Component: pan_zoom — extracted from generate_multi_slide_spatial_viewer.py
// Mouse wheel zoom handler, mouse drag for panning, and draw drag handling.
// These are set up per-panel inside initPanels() and as global window handlers.

// Per-panel wheel zoom (attached to both data canvas and draw overlay canvas):
function handleWheel(state, div, e) {
  e.preventDefault();
  const rect = div.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  state.panX = mx - factor * (mx - state.panX);
  state.panY = my - factor * (my - state.panY);
  state.zoom *= factor;
  state.zoom = Math.max(0.001, Math.min(500, state.zoom));
  scheduleRender(state);
  renderDrawOverlay(state);
}

// Per-panel mousedown on data canvas (pan start):
function handlePanStart(state, div, e) {
  if (drawMode !== 'pan') return;
  activePanel = state;
  div.classList.add('dragging');
  state.dragStartX = e.clientX;
  state.dragStartY = e.clientY;
  state.panStartX = state.panX;
  state.panStartY = state.panY;
  e.preventDefault();
}

// Global mousemove handler (pan drag + draw drag for circle/rect):
function handleGlobalMouseMove(e) {
  // Pan drag
  if (activePanel) {
    activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
    activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
    scheduleRender(activePanel);
    return;
  }
  // Draw drag (circle/rect)
  if (drawStart && drawMode !== 'pan') {
    const p = drawStart.panel;
    const rect = p.div.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [dx, dy] = screenToData(p, sx, sy);
    drawCurrent = { x: dx, y: dy };
    if (drawMode === 'circle') {
      const ddx = dx - drawStart.x, ddy = dy - drawStart.y;
      const r = Math.sqrt(ddx * ddx + ddy * ddy);
      p.measureEl.style.display = 'block';
      p.measureEl.textContent = 'r = ' + r.toFixed(0) + ' \u00b5m';
    } else if (drawMode === 'rect') {
      const w = Math.abs(dx - drawStart.x);
      const h = Math.abs(dy - drawStart.y);
      p.measureEl.style.display = 'block';
      p.measureEl.textContent = w.toFixed(0) + ' \u00d7 ' + h.toFixed(0) + ' \u00b5m';
    }
    renderDrawOverlay(p);
  }
}

// Global mouseup handler (pan end + draw end for circle/rect):
function handleGlobalMouseUp(e) {
  // Pan drag end
  if (activePanel) {
    activePanel.div.classList.remove('dragging');
    activePanel = null;
    return;
  }
  // Draw drag end (circle/rect)
  if (drawStart && drawMode !== 'pan' && drawMode !== 'polygon' && drawMode !== 'path') {
    const p = drawStart.panel;
    const rect = p.div.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [dx, dy] = screenToData(p, sx, sy);
    if (drawMode === 'circle') {
      const cdx = dx - drawStart.x, cdy = dy - drawStart.y;
      const r = Math.sqrt(cdx * cdx + cdy * cdy);
      if (r > 1) addROI(p.idx, 'circle', { cx: drawStart.x, cy: drawStart.y, r });
    } else if (drawMode === 'rect') {
      const w = Math.abs(dx - drawStart.x), h = Math.abs(dy - drawStart.y);
      if (w > 1 && h > 1) addROI(p.idx, 'rect', { x1: drawStart.x, y1: drawStart.y, x2: dx, y2: dy });
    }
    drawStart = null;
    drawCurrent = null;
    p.measureEl.style.display = 'none';
    renderDrawOverlay(p);
  }
}
