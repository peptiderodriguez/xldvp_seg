// Component: convex_hull — extracted from generate_multi_slide_spatial_viewer.py
// Convex hull computation using Andrew's monotone chain algorithm

function convexHull(points) {
  const n = points.length;
  if (n < 3) return points.slice();
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const pts = [sorted[0]];
  for (let i = 1; i < n; i++) {
    if (sorted[i][0] !== sorted[i-1][0] || sorted[i][1] !== sorted[i-1][1])
      pts.push(sorted[i]);
  }
  if (pts.length < 3) return pts;

  function cross(O, A, B) {
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}
