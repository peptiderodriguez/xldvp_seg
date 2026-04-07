// Component: kde — extracted from generate_multi_slide_spatial_viewer.py
// Full KDE pipeline: 2D histogram, Gaussian blur, marching squares isoline
// extraction, per-group KDE computation with caching, and contour rendering.
//
// Requires globals: SLIDES, GROUP_LABELS, GROUP_COLORS, N_GROUPS, hidden,
//   KDE_RADII, kdeCache, kdeBWIdx, kdeLevels
// Requires functions: getGroupPositions (viewer-specific), hexToRgb (below)

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.substring(0,2),16), parseInt(h.substring(2,4),16), parseInt(h.substring(4,6),16)];
}

function getGroupPositions(slideIdx) {
  const slide = SLIDES[slideIdx];
  if (slide._groupPos) return slide._groupPos;
  const gp = new Array(N_GROUPS).fill(null).map(() => ({xi:[], yi:[]}));
  for (let i = 0; i < slide.n; i++) {
    const gi = slide.grp[i];
    gp[gi].xi.push(slide.pos[i*2]);
    gp[gi].yi.push(slide.pos[i*2+1]);
  }
  slide._groupPos = gp.map(g => ({
    x: new Float32Array(g.xi),
    y: new Float32Array(g.yi),
    n: g.xi.length,
  }));
  return slide._groupPos;
}

function computeHistogram2D(x, y, w, nx, ny, xr, yr) {
  const grid = new Float32Array(ny * nx);
  const sx = nx / (xr[1] - xr[0]);
  const sy = ny / (yr[1] - yr[0]);
  const n = x.length;
  for (let i = 0; i < n; i++) {
    const gx = Math.min(Math.floor((x[i] - xr[0]) * sx), nx - 1);
    const gy = Math.min(Math.floor((y[i] - yr[0]) * sy), ny - 1);
    if (gx >= 0 && gy >= 0) {
      grid[gy * nx + gx] += (w ? w[i] : 1);
    }
  }
  return grid;
}

function gaussianBlur1D(src, nx, ny, sigma, horizontal) {
  const radius = Math.ceil(sigma * 3);
  const kernel = new Float32Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    kernel[i + radius] = Math.exp(-0.5 * (i / sigma) * (i / sigma));
    ksum += kernel[i + radius];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const dst = new Float32Array(ny * nx);

  if (horizontal) {
    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const c = Math.min(Math.max(col + k, 0), nx - 1);
          sum += src[row * nx + c] * kernel[k + radius];
        }
        dst[row * nx + col] = sum;
      }
    }
  } else {
    for (let col = 0; col < nx; col++) {
      for (let row = 0; row < ny; row++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const r = Math.min(Math.max(row + k, 0), ny - 1);
          sum += src[r * nx + col] * kernel[k + radius];
        }
        dst[row * nx + col] = sum;
      }
    }
  }
  return dst;
}

function gaussianBlur(grid, nx, ny, sigma) {
  if (sigma < 0.5) return grid;
  const tmp = gaussianBlur1D(grid, nx, ny, sigma, true);
  return gaussianBlur1D(tmp, nx, ny, sigma, false);
}

function marchingSquares(grid, nx, ny, threshold, xr, yr) {
  const stepX = (xr[1] - xr[0]) / nx;
  const stepY = (yr[1] - yr[0]) / ny;

  function lerp(v1, v2) {
    const d = v2 - v1;
    return Math.abs(d) < 1e-10 ? 0.5 : (threshold - v1) / d;
  }

  const segments = [];
  for (let row = 0; row < ny - 1; row++) {
    for (let col = 0; col < nx - 1; col++) {
      const tl = grid[row * nx + col] >= threshold ? 1 : 0;
      const tr = grid[row * nx + col + 1] >= threshold ? 1 : 0;
      const br = grid[(row + 1) * nx + col + 1] >= threshold ? 1 : 0;
      const bl = grid[(row + 1) * nx + col] >= threshold ? 1 : 0;
      let code = (tl << 3) | (tr << 2) | (br << 1) | bl;

      if (code === 0 || code === 15) continue;

      const x0 = xr[0] + col * stepX;
      const y0 = yr[0] + row * stepY;
      const vTL = grid[row * nx + col];
      const vTR = grid[row * nx + col + 1];
      const vBR = grid[(row + 1) * nx + col + 1];
      const vBL = grid[(row + 1) * nx + col];

      const top = [x0 + lerp(vTL, vTR) * stepX, y0];
      const right = [x0 + stepX, y0 + lerp(vTR, vBR) * stepY];
      const bottom = [x0 + lerp(vBL, vBR) * stepX, y0 + stepY];
      const left = [x0, y0 + lerp(vTL, vBL) * stepY];

      if (code === 5 || code === 10) {
        const center = (vTL + vTR + vBR + vBL) / 4;
        if (center >= threshold) {
          if (code === 5) code = 17;
          else code = 18;
        }
      }

      let segs;
      switch (code) {
        case 1:  segs = [[left, bottom]]; break;
        case 2:  segs = [[bottom, right]]; break;
        case 3:  segs = [[left, right]]; break;
        case 4:  segs = [[right, top]]; break;
        case 5:  segs = [[left, top], [bottom, right]]; break;
        case 17: segs = [[left, bottom], [top, right]]; break;
        case 6:  segs = [[bottom, top]]; break;
        case 7:  segs = [[left, top]]; break;
        case 8:  segs = [[top, left]]; break;
        case 9:  segs = [[top, bottom]]; break;
        case 10: segs = [[top, right], [left, bottom]]; break;
        case 18: segs = [[top, left], [bottom, right]]; break;
        case 11: segs = [[top, right]]; break;
        case 12: segs = [[right, left]]; break;
        case 13: segs = [[right, bottom]]; break;
        case 14: segs = [[bottom, left]]; break;
        default: segs = null;
      }

      if (segs) {
        for (const seg of segs) segments.push(seg);
      }
    }
  }

  if (segments.length === 0) return [];

  const eps = stepX * 0.01;
  const eps2 = eps * eps;

  function ptKey(p) {
    return Math.round(p[0] / eps) + ',' + Math.round(p[1] / eps);
  }

  const endHash = new Map();
  for (let i = 0; i < segments.length; i++) {
    for (let e = 0; e < 2; e++) {
      const k = ptKey(segments[i][e]);
      if (!endHash.has(k)) endHash.set(k, []);
      endHash.get(k).push({ si: i, ei: e });
    }
  }

  function dist2(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  const polys = [];
  const used = new Uint8Array(segments.length);

  for (let start = 0; start < segments.length; start++) {
    if (used[start]) continue;
    used[start] = 1;
    const poly = [segments[start][0], segments[start][1]];

    let found = true;
    while (found) {
      found = false;
      const tail = poly[poly.length - 1];
      const rx = Math.round(tail[0] / eps);
      const ry = Math.round(tail[1] / eps);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const bucket = endHash.get((rx + dx) + ',' + (ry + dy));
          if (!bucket) continue;
          for (const entry of bucket) {
            if (used[entry.si]) continue;
            const seg = segments[entry.si];
            if (dist2(tail, seg[entry.ei]) < eps2) {
              poly.push(seg[1 - entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }
            if (dist2(tail, seg[1 - entry.ei]) < eps2) {
              poly.push(seg[entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }
          }
          if (found) break;
        }
        if (found) break;
      }
    }
    if (poly.length >= 3) polys.push(poly);
  }

  return polys;
}

function computeKDE(slideIdx, bandwidthUm, nLevels) {
  const slide = SLIDES[slideIdx];
  const xr = slide.xr, yr = slide.yr;
  const dataW = xr[1] - xr[0], dataH = yr[1] - yr[0];
  if (dataW <= 0 || dataH <= 0) return null;

  const nxGrid = 200, nyGrid = 200;
  const pixelSize = Math.max(dataW, dataH) / Math.max(nxGrid, nyGrid);
  const sigma = bandwidthUm / pixelSize;

  let nx, ny;
  if (dataW > dataH) {
    nx = nxGrid;
    ny = Math.max(1, Math.round(nxGrid * dataH / dataW));
  } else {
    ny = nyGrid;
    nx = Math.max(1, Math.round(nyGrid * dataW / dataH));
  }

  const gpos = getGroupPositions(slideIdx);
  const result = [];

  for (let gi = 0; gi < N_GROUPS; gi++) {
    if (hidden.has(GROUP_LABELS[gi])) continue;
    const gp = gpos[gi];
    if (gp.n < 5) continue;

    const hist = computeHistogram2D(gp.x, gp.y, null, nx, ny, xr, yr);
    const blurred = gaussianBlur(hist, nx, ny, sigma);

    let maxD = 0;
    for (let i = 0; i < blurred.length; i++) {
      if (blurred[i] > maxD) maxD = blurred[i];
    }
    if (maxD <= 0) continue;

    const contours = [];
    for (let li = 1; li <= nLevels; li++) {
      const frac = li / (nLevels + 1);
      const threshold = maxD * frac;
      const polys = marchingSquares(blurred, nx, ny, threshold, xr, yr);
      contours.push({ level: frac, polys });
    }

    result.push({ gi, color: GROUP_COLORS[gi], contours });
  }

  return result;
}

function getKDE(slideIdx) {
  const hiddenKey = Array.from(hidden).sort().join('|');
  const cached = kdeCache.get(slideIdx);
  if (cached && cached.bwIdx === kdeBWIdx && cached.levels === kdeLevels && cached.hiddenKey === hiddenKey) {
    return cached.data;
  }
  const bw = KDE_RADII[kdeBWIdx];
  const data = computeKDE(slideIdx, bw, kdeLevels);
  kdeCache.set(slideIdx, { bwIdx: kdeBWIdx, levels: kdeLevels, hiddenKey, data });
  return data;
}

function drawKDEContours(ctx, kdeData, panZoom, opacity, fill, lines) {
  if (!kdeData) return;

  for (const entry of kdeData) {
    const color = entry.color;
    const [r, g, b] = hexToRgb(color);

    for (let li = entry.contours.length - 1; li >= 0; li--) {
      const { level, polys } = entry.contours[li];

      for (const poly of polys) {
        if (poly.length < 3) continue;

        const path = new Path2D();
        path.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
          path.lineTo(poly[i][0], poly[i][1]);
        }
        path.closePath();

        if (fill) {
          ctx.globalAlpha = opacity * (1 - level) * 0.6;
          ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',1)';
          ctx.fill(path);
        }

        if (lines) {
          ctx.globalAlpha = opacity;
          ctx.strokeStyle = color;
          ctx.lineWidth = (1.5 - level * 0.5) / panZoom;
          ctx.stroke(path);
        }
      }
    }
  }
}
