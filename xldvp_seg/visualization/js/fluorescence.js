// Component: fluorescence — extracted from generate_multi_slide_spatial_viewer.py
// Additive RGB fluorescence channel compositing onto canvas
// Requires globals: fluorImages, chEnabled, fluorAlpha, CH_TINTS
// Requires functions: scheduleRenderAll (for async image load callback)

function drawFluorescence(ctx, slideIdx, panZoom) {
  const fd = fluorImages[slideIdx];
  if (!fd || !fd.ready) return;

  const meta = fd.meta;
  const iw = meta.w, ih = meta.h;
  // Scale factor maps thumbnail pixels -> full-resolution pixels.
  // The viewer coordinate space is in um, so we also multiply by pixel_size.
  // thumbnail_pixel = full_res_pixel * scale
  // um = full_res_pixel * pixel_size
  // => full_res_pixel = thumbnail_pixel / scale
  // => um = (thumbnail_pixel / scale) * pixel_size
  // => thumbnail_pixel = um / pixel_size * scale
  // Draw position in um space: mosaic origin in full-res pixels -> um
  const mx_um = meta.mx * meta.pixel_size;
  const my_um = meta.my * meta.pixel_size;
  const scale_inv = 1.0 / meta.scale;  // thumbnail pixel -> full-res pixel
  const draw_w = iw * scale_inv * meta.pixel_size;  // um
  const draw_h = ih * scale_inv * meta.pixel_size;  // um

  // Rebuild composite offscreen canvas only when needed
  if (fd._dirty || !fd._canvas) {
    if (!fd._canvas) {
      fd._canvas = document.createElement('canvas');
      fd._canvas.width = iw;
      fd._canvas.height = ih;
    }
    const fctx = fd._canvas.getContext('2d', { willReadFrequently: true });
    // Additive channel compositing via pixel-level blend
    const result = new Uint8ClampedArray(iw * ih * 4);
    for (let ci = 0; ci < 3; ci++) {
      if (!chEnabled[ci] || !fd.imgs[ci]) continue;
      // Draw grayscale channel to temp canvas, read pixels
      const tmp = document.createElement('canvas');
      tmp.width = iw; tmp.height = ih;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(fd.imgs[ci], 0, 0);
      const px = tctx.getImageData(0, 0, iw, ih).data;
      const [tr, tg, tb] = CH_TINTS[ci];
      for (let i = 0; i < iw * ih; i++) {
        const v = px[i * 4] / 255;
        result[i * 4]     = Math.min(255, result[i * 4]     + tr * v);
        result[i * 4 + 1] = Math.min(255, result[i * 4 + 1] + tg * v);
        result[i * 4 + 2] = Math.min(255, result[i * 4 + 2] + tb * v);
        result[i * 4 + 3] = 255;
      }
    }
    fctx.putImageData(new ImageData(result, iw, ih), 0, 0);
    fd._dirty = false;
  }

  ctx.globalAlpha = fluorAlpha;
  ctx.drawImage(fd._canvas, mx_um, my_um, draw_w, draw_h);
  ctx.globalAlpha = 1;
}
