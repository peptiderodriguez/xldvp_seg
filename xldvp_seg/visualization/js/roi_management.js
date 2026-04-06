// Component: roi_management — extracted from generate_multi_slide_spatial_viewer.py
// ROI CRUD operations: add, delete, update list, update stats, download JSON

function addROI(slideIdx, type, data) {
  roiCounter++;
  const roi = {
    id: 'ROI_' + roiCounter,
    slideIdx,
    type,
    data,
    name: 'ROI_' + roiCounter,
    category: '',
  };
  rois.push(roi);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
  updateCorridorVisibility();
}

function deleteROI(id) {
  const idx = rois.findIndex(r => r.id === id);
  if (idx >= 0) rois.splice(idx, 1);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
  updateCorridorVisibility();
}

function updateROIList() {
  const div = document.getElementById('roi-list');
  div.innerHTML = '';
  for (const roi of rois) {
    const item = document.createElement('div');
    item.className = 'roi-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'roi-name';
    nameSpan.contentEditable = true;
    nameSpan.textContent = roi.name;
    nameSpan.title = SLIDES[roi.slideIdx].name + ' | ' + roi.type;
    nameSpan.onblur = () => { roi.name = nameSpan.textContent.trim() || roi.id; };
    nameSpan.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); nameSpan.blur(); } };

    const catSpan = document.createElement('span');
    catSpan.className = 'roi-category';
    catSpan.contentEditable = true;
    catSpan.textContent = roi.category || '';
    catSpan.title = 'Category: e.g. central_vein, portal_vein, liver';
    catSpan.onblur = () => { roi.category = catSpan.textContent.trim(); };
    catSpan.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); catSpan.blur(); } };

    const statsSpan = document.createElement('span');
    statsSpan.className = 'roi-stats';
    statsSpan.dataset.roiId = roi.id;

    const delBtn = document.createElement('span');
    delBtn.className = 'roi-del';
    delBtn.textContent = '\u00d7';
    delBtn.onclick = () => deleteROI(roi.id);

    item.appendChild(nameSpan);
    item.appendChild(catSpan);
    item.appendChild(statsSpan);
    item.appendChild(delBtn);
    div.appendChild(item);
  }
}

function updateROIStats() {
  // Count cells inside each ROI
  for (const roi of rois) {
    let count = 0;
    const slide = SLIDES[roi.slideIdx];
    const pos = slide.pos;
    const grp = slide.grp;
    for (let i = 0; i < slide.n; i++) {
      if (hidden.has(GROUP_LABELS[grp[i]])) continue;
      if (pointInROI(pos[i * 2], pos[i * 2 + 1], roi)) count++;
    }
    const el = document.querySelector('[data-roi-id="' + roi.id + '"]');
    if (el) el.textContent = count.toLocaleString();
  }

  const statsDiv = document.getElementById('roi-stats');
  if (rois.length === 0) {
    statsDiv.textContent = '';
  } else {
    statsDiv.textContent = rois.length + ' ROI(s) drawn';
  }
}

function updateCorridorVisibility() {
  const hasPath = rois.some(r => r.type === 'path');
  document.getElementById('corridor-row').style.display = hasPath ? 'flex' : 'none';
}

function downloadROIs() {
  const out = {
    rois: [],
    metadata: {
      generated: GENERATED,
      title: TITLE,
      group_field: GROUP_FIELD,
    },
  };
  for (const roi of rois) {
    const slideName = SLIDES[roi.slideIdx].name;
    const entry = { id: roi.id, slide: slideName, type: roi.type, name: roi.name };
    entry.category = roi.category || '';
    if (roi.type === 'circle') {
      entry.center_um = [roi.data.cx, roi.data.cy];
      entry.radius_um = roi.data.r;
    } else if (roi.type === 'rect') {
      entry.min_um = [Math.min(roi.data.x1, roi.data.x2), Math.min(roi.data.y1, roi.data.y2)];
      entry.max_um = [Math.max(roi.data.x1, roi.data.x2), Math.max(roi.data.y1, roi.data.y2)];
    } else if (roi.type === 'polygon') {
      entry.vertices_um = roi.data.verts;
    } else if (roi.type === 'path') {
      entry.waypoints_um = roi.data.waypoints;
      entry.corridor_um = roi.data.corridorWidth || corridorWidth;
    }
    out.rois.push(entry);
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'rois.json';
  a.click();
  URL.revokeObjectURL(url);
}
