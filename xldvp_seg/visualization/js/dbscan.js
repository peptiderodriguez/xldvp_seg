// Component: dbscan — extracted from generate_multi_slide_spatial_viewer.py
// DBSCAN clustering with grid-based spatial index for neighbor lookups

function dbscan(x, y, n, eps, minPts) {
  const labels = new Int32Array(n).fill(-1);
  if (n === 0 || eps <= 0) return labels;

  const grid = new Map();
  for (let i = 0; i < n; i++) {
    const gx = Math.floor(x[i] / eps);
    const gy = Math.floor(y[i] / eps);
    const key = gx + ',' + gy;
    let cell = grid.get(key);
    if (!cell) { cell = []; grid.set(key, cell); }
    cell.push(i);
  }

  const eps2 = eps * eps;
  function getNeighbors(idx) {
    const px = x[idx], py = y[idx];
    const gx = Math.floor(px / eps);
    const gy = Math.floor(py / eps);
    const result = [];
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = grid.get((gx + dx) + ',' + (gy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k++) {
          const j = cell[k];
          const ddx = x[j] - px, ddy = y[j] - py;
          if (ddx * ddx + ddy * ddy <= eps2) result.push(j);
        }
      }
    }
    return result;
  }

  let clusterId = 0;
  const visited = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = 1;
    const nbrs = getNeighbors(i);
    if (nbrs.length < minPts) continue;

    labels[i] = clusterId;
    const queue = [];
    for (let k = 0; k < nbrs.length; k++) {
      if (nbrs[k] !== i) queue.push(nbrs[k]);
    }
    let qi = 0;
    while (qi < queue.length) {
      const j = queue[qi++];
      if (!visited[j]) {
        visited[j] = 1;
        const jnbrs = getNeighbors(j);
        if (jnbrs.length >= minPts) {
          for (let k = 0; k < jnbrs.length; k++) {
            if (!visited[jnbrs[k]]) queue.push(jnbrs[k]);
          }
        }
      }
      if (labels[j] === -1) labels[j] = clusterId;
    }
    clusterId++;
  }
  return labels;
}
