// region_viewer.js — Shared canvas rendering for region viewers.
// Loaded via load_js("region_viewer"). Expects globals:
//   W, H — image dimensions
//   BGS — {name: "data:image/...;base64,..."} background images
//   drawRegions(ctx, zoom, selectedIdx) — caller-defined region drawing function

const _bgImgs = {};
for (const [k, s] of Object.entries(BGS)) {
    const im = new Image();
    im.src = s;
    _bgImgs[k] = im;
}

let _panX = 0, _panY = 0, _zoom = 1;
let _dragging = false, _dragX = 0, _dragY = 0;
let _curBg = Object.keys(BGS)[0] || 'fluor';
let _lwV = 1.2;

const _canvas = document.getElementById('cv');
const _ctx = _canvas.getContext('2d');
const _wrap = document.getElementById('cw');

function _resizeCanvas() {
    _canvas.width = _wrap.clientWidth;
    _canvas.height = _wrap.clientHeight;
    draw();
}
window.addEventListener('resize', _resizeCanvas);

function fitView() {
    _zoom = Math.min(_wrap.clientWidth / W, _wrap.clientHeight / H) * 0.95;
    _panX = (_wrap.clientWidth - W * _zoom) / 2;
    _panY = (_wrap.clientHeight - H * _zoom) / 2;
    draw();
}

function _initView() {
    _resizeCanvas();
    fitView();
}

// Try multiple init strategies (base64 images may load sync or async)
const _firstBg = _bgImgs[_curBg];
if (_firstBg) {
    _firstBg.onload = () => { _initView(); };
    if (_firstBg.complete) { requestAnimationFrame(_initView); }
}
window.addEventListener('load', _initView);

function draw() {
    _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
    _ctx.save();
    _ctx.translate(_panX, _panY);
    _ctx.scale(_zoom, _zoom);
    const bg = _bgImgs[_curBg];
    if (bg && bg.complete) _ctx.drawImage(bg, 0, 0, W, H);
    // Caller provides drawRegions(ctx, zoom, lwV)
    if (typeof drawRegions === 'function') drawRegions(_ctx, _zoom, _lwV);
    _ctx.restore();
    const lwLabel = document.getElementById('lwl');
    if (lwLabel) lwLabel.textContent = _lwV.toFixed(1);
}

function setBg(v) { _curBg = v; draw(); }

// Pan + zoom
_canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const r = _canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    _panX = mx - (mx - _panX) * f;
    _panY = my - (my - _panY) * f;
    _zoom *= f;
    draw();
});
_canvas.addEventListener('mousedown', e => {
    _dragging = true; _dragX = e.clientX - _panX; _dragY = e.clientY - _panY;
});
_canvas.addEventListener('mousemove', e => {
    if (_dragging) { _panX = e.clientX - _dragX; _panY = e.clientY - _dragY; draw(); }
});
_canvas.addEventListener('mouseup', () => _dragging = false);
_canvas.addEventListener('mouseleave', () => _dragging = false);

// PNG download
function dlPng(withMasks) {
    const scale = 3, ww = W * scale, hh = H * scale;
    const oc = document.createElement('canvas');
    oc.width = ww; oc.height = hh;
    const ox = oc.getContext('2d');
    const s = Math.min(ww / W, hh / H);
    const offX = (ww - W * s) / 2, offY = (hh - H * s) / 2;
    ox.save(); ox.translate(offX, offY); ox.scale(s, s);
    const bg = _bgImgs[_curBg];
    if (bg && bg.complete) ox.drawImage(bg, 0, 0, W, H);
    if (withMasks && typeof drawRegions === 'function') drawRegions(ox, s, _lwV);
    ox.restore();
    const a = document.createElement('a');
    a.download = withMasks ? 'regions_with_masks.png' : 'regions_background.png';
    a.href = oc.toDataURL('image/png');
    a.click();
}

// Point-in-polygon (for click detection)
function pointInPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

// Canvas click → image coordinates
function canvasToImage(e) {
    const r = _canvas.getBoundingClientRect();
    return [(e.clientX - r.left - _panX) / _zoom, (e.clientY - r.top - _panY) / _zoom];
}
