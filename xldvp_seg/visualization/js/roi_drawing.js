// Component: roi_drawing — extracted from generate_multi_slide_spatial_viewer.py
// Render draw overlay: existing ROIs, in-progress shapes (circle/rect/polygon/path)

function renderDrawOverlay(p) {
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const dctx = p.dctx;
  dctx.clearRect(0, 0, cw, ch);
  dctx.save();
  dctx.translate(p.panX, p.panY);
  dctx.scale(p.zoom, p.zoom);

  const lw = 1.5 / p.zoom;

  // Draw existing ROIs for this slide
  for (const roi of rois) {
    if (roi.slideIdx !== p.idx) continue;
    dctx.strokeStyle = '#ffcc00';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.8;
    dctx.setLineDash([4 / p.zoom, 3 / p.zoom]);

    if (roi.type === 'circle') {
      dctx.beginPath();
      dctx.arc(roi.data.cx, roi.data.cy, roi.data.r, 0, Math.PI * 2);
      dctx.stroke();
    } else if (roi.type === 'rect') {
      const x = Math.min(roi.data.x1, roi.data.x2);
      const y = Math.min(roi.data.y1, roi.data.y2);
      const w = Math.abs(roi.data.x2 - roi.data.x1);
      const h = Math.abs(roi.data.y2 - roi.data.y1);
      dctx.strokeRect(x, y, w, h);
    } else if (roi.type === 'polygon') {
      dctx.beginPath();
      dctx.moveTo(roi.data.verts[0][0], roi.data.verts[0][1]);
      for (let i = 1; i < roi.data.verts.length; i++) {
        dctx.lineTo(roi.data.verts[i][0], roi.data.verts[i][1]);
      }
      dctx.closePath();
      dctx.stroke();
    } else if (roi.type === 'path') {
      // Corridor fill
      dctx.save();
      dctx.globalAlpha = 0.12;
      dctx.strokeStyle = '#ffcc00';
      dctx.lineWidth = roi.data.corridorWidth || corridorWidth;
      dctx.lineCap = 'round';
      dctx.lineJoin = 'round';
      dctx.setLineDash([]);
      dctx.beginPath();
      dctx.moveTo(roi.data.waypoints[0][0], roi.data.waypoints[0][1]);
      for (let i = 1; i < roi.data.waypoints.length; i++) {
        dctx.lineTo(roi.data.waypoints[i][0], roi.data.waypoints[i][1]);
      }
      dctx.stroke();
      dctx.restore();
      // Centerline
      dctx.beginPath();
      dctx.moveTo(roi.data.waypoints[0][0], roi.data.waypoints[0][1]);
      for (let i = 1; i < roi.data.waypoints.length; i++) {
        dctx.lineTo(roi.data.waypoints[i][0], roi.data.waypoints[i][1]);
      }
      dctx.stroke();
      // Endpoint markers: green=start(CV), red=end(PV)
      const er = 4 / p.zoom;
      dctx.globalAlpha = 1;
      dctx.setLineDash([]);
      dctx.fillStyle = '#00ff00';
      dctx.beginPath();
      dctx.arc(roi.data.waypoints[0][0], roi.data.waypoints[0][1], er, 0, Math.PI * 2);
      dctx.fill();
      dctx.fillStyle = '#ff4444';
      const last = roi.data.waypoints[roi.data.waypoints.length - 1];
      dctx.beginPath();
      dctx.arc(last[0], last[1], er, 0, Math.PI * 2);
      dctx.fill();
    }
    dctx.setLineDash([]);

    // ROI label
    const fontSize = 10 / p.zoom;
    dctx.font = fontSize + 'px system-ui';
    dctx.fillStyle = '#ffcc00';
    dctx.globalAlpha = 0.9;
    dctx.textAlign = 'left';
    dctx.textBaseline = 'top';
    let labelX, labelY;
    if (roi.type === 'circle') {
      labelX = roi.data.cx - roi.data.r;
      labelY = roi.data.cy - roi.data.r - fontSize * 1.3;
    } else if (roi.type === 'rect') {
      labelX = Math.min(roi.data.x1, roi.data.x2);
      labelY = Math.min(roi.data.y1, roi.data.y2) - fontSize * 1.3;
    } else if (roi.type === 'polygon') {
      labelX = roi.data.verts[0][0];
      labelY = roi.data.verts[0][1] - fontSize * 1.3;
    } else if (roi.type === 'path') {
      labelX = roi.data.waypoints[0][0];
      labelY = roi.data.waypoints[0][1] - fontSize * 1.3;
    }
    dctx.fillText(roi.name, labelX, labelY);
  }

  // Draw in-progress shape
  if (drawStart && drawCurrent && drawStart.panel === p) {
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.setLineDash([3 / p.zoom, 2 / p.zoom]);

    if (drawMode === 'circle') {
      const dx = drawCurrent.x - drawStart.x;
      const dy = drawCurrent.y - drawStart.y;
      const r = Math.sqrt(dx * dx + dy * dy);
      dctx.beginPath();
      dctx.arc(drawStart.x, drawStart.y, r, 0, Math.PI * 2);
      dctx.stroke();
      // Radius text
      const fontSize = 10 / p.zoom;
      dctx.font = fontSize + 'px system-ui';
      dctx.fillStyle = '#00ff88';
      dctx.textAlign = 'center';
      dctx.fillText('r=' + r.toFixed(0) + ' \u00b5m', drawStart.x, drawStart.y - r - fontSize);
    } else if (drawMode === 'rect') {
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      dctx.strokeRect(x, y, w, h);
      // Dimensions text
      const fontSize = 10 / p.zoom;
      dctx.font = fontSize + 'px system-ui';
      dctx.fillStyle = '#00ff88';
      dctx.textAlign = 'center';
      dctx.fillText(w.toFixed(0) + ' \u00d7 ' + h.toFixed(0) + ' \u00b5m', x + w / 2, y - fontSize);
    }
    dctx.setLineDash([]);
  }

  // Draw in-progress polygon/path
  if ((drawMode === 'polygon' || drawMode === 'path') && polySlideIdx === p.idx && polyVerts.length > 0) {
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.beginPath();
    dctx.moveTo(polyVerts[0][0], polyVerts[0][1]);
    for (let i = 1; i < polyVerts.length; i++) {
      dctx.lineTo(polyVerts[i][0], polyVerts[i][1]);
    }
    dctx.stroke();
    // Draw vertices
    const vr = 3 / p.zoom;
    dctx.fillStyle = '#00ff88';
    for (const v of polyVerts) {
      dctx.beginPath();
      dctx.arc(v[0], v[1], vr, 0, Math.PI * 2);
      dctx.fill();
    }
    // Vertex count
    const fontSize = 10 / p.zoom;
    dctx.font = fontSize + 'px system-ui';
    dctx.textAlign = 'left';
    dctx.fillText(polyVerts.length + ' pts', polyVerts[polyVerts.length - 1][0] + 5 / p.zoom, polyVerts[polyVerts.length - 1][1]);
  }

  dctx.restore();
}
