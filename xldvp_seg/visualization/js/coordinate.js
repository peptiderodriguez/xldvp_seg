// Component: coordinate — extracted from generate_multi_slide_spatial_viewer.py
// Coordinate transforms between screen space and data space

function screenToData(p, sx, sy) {
  return [(sx - p.panX) / p.zoom, (sy - p.panY) / p.zoom];
}

function dataToScreen(p, dx, dy) {
  return [dx * p.zoom + p.panX, dy * p.zoom + p.panY];
}
