// Component: focus_view — extracted from generate_multi_slide_spatial_viewer.py
// Enter/exit focus view for single-panel zoom (double-click a panel in grid)

function enterFocusView(idx) {
  if (IS_SINGLE) return;
  focusedIdx = idx;
  const focusView = document.getElementById('focus-view');
  const focusLabel = document.getElementById('focus-label');
  const grid = document.getElementById('grid');
  const p = panels[idx];

  // Move panel div into focus view
  focusView.appendChild(p.div);
  p.div.style.position = 'absolute';
  p.div.style.top = '0';
  p.div.style.left = '0';
  p.div.style.width = '100%';
  p.div.style.height = '100%';
  p.div.style.borderRadius = '0';

  focusLabel.textContent = p.slide.name + ' \u2014 ' + p.slide.n.toLocaleString() + ' cells';
  focusView.classList.add('active');
  grid.style.display = 'none';

  // Resize and re-render
  setTimeout(() => {
    resizePanel(p);
    fitPanel(p);
    p.visible = true;
    scheduleRender(p);
    renderDrawOverlay(p);
  }, 50);
}

function exitFocusView() {
  if (focusedIdx < 0) return;
  const p = panels[focusedIdx];
  const focusView = document.getElementById('focus-view');
  const grid = document.getElementById('grid');

  // Move panel back to grid
  p.div.style.position = '';
  p.div.style.top = '';
  p.div.style.left = '';
  p.div.style.width = '';
  p.div.style.height = '';
  p.div.style.borderRadius = '';

  // Re-insert at correct position in grid
  const nextIdx = focusedIdx + 1;
  if (nextIdx < panels.length) {
    grid.insertBefore(p.div, panels[nextIdx].div);
  } else {
    grid.appendChild(p.div);
  }

  focusView.classList.remove('active');
  grid.style.display = '';
  focusedIdx = -1;

  // Resize all and re-render
  setTimeout(() => {
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
    panels.forEach(pp => renderDrawOverlay(pp));
  }, 50);
}
