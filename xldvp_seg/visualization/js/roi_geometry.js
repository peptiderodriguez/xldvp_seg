// Component: roi_geometry — extracted from generate_multi_slide_spatial_viewer.py
// ROI geometry tests: point-in-shape functions for circle, rect, polygon, path

function pointInCircle(px, py, cx, cy, r) {
  const dx = px - cx, dy = py - cy;
  return dx * dx + dy * dy <= r * r;
}

function pointInRect(px, py, x1, y1, x2, y2) {
  const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
  return px >= minX && px <= maxX && py >= minY && py <= maxY;
}

function pointInPolygon(px, py, verts) {
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function pointNearPath(px, py, waypoints, halfWidth) {
  const hw2 = halfWidth * halfWidth;
  for (let i = 0; i < waypoints.length - 1; i++) {
    const [ax, ay] = waypoints[i];
    const [bx, by] = waypoints[i + 1];
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) continue;
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const projX = ax + t * dx, projY = ay + t * dy;
    const distSq = (px - projX) * (px - projX) + (py - projY) * (py - projY);
    if (distSq <= hw2) return true;
  }
  return false;
}

function pointInROI(px, py, roi) {
  if (roi.type === 'circle') {
    return pointInCircle(px, py, roi.data.cx, roi.data.cy, roi.data.r);
  } else if (roi.type === 'rect') {
    return pointInRect(px, py, roi.data.x1, roi.data.y1, roi.data.x2, roi.data.y2);
  } else if (roi.type === 'polygon') {
    return pointInPolygon(px, py, roi.data.verts);
  } else if (roi.type === 'path') {
    return pointNearPath(px, py, roi.data.waypoints, (roi.data.corridorWidth || corridorWidth) / 2);
  }
  return false;
}

function cellPassesROIFilter(px, py, slideIdx) {
  if (!roiFilterActive || rois.length === 0) return true;
  for (const roi of rois) {
    if (roi.slideIdx === slideIdx && pointInROI(px, py, roi)) return true;
  }
  return false;
}
