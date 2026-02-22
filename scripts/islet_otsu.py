import json, numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('/fs/pool/pool-mann-edwin/islet_output/2025_09_03_30610012_BS-100_20260220_211212_10pct/islet_detections.json') as f:
    dets = json.load(f)

pixel_size = 0.22

gcg_all = np.array([d.get('features',{}).get('ch2_mean', 0) for d in dets])
ins_all = np.array([d.get('features',{}).get('ch3_mean', 0) for d in dets])
sst_all = np.array([d.get('features',{}).get('ch5_mean', 0) for d in dets])

def norm(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
    return np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)

gcg_n = norm(gcg_all)
ins_n = norm(ins_all)
sst_n = norm(sst_all)
total_n = gcg_n + ins_n + sst_n

# Group by islet
islets = {}
for i, d in enumerate(dets):
    iid = d.get('islet_id')
    if iid is None or iid < 0: continue
    islets.setdefault(iid, []).append(i)

header = f"{'Islet':>6} {'Cells':>6} {'Area':>10} {'TotalSig':>10} {'Sig/Area':>10} {'Sig/Cell':>10} {'Gcg':>6} {'Ins':>6} {'Sst':>6}"
print(header)

islet_metrics = []
for iid in sorted(islets):
    idx = islets[iid]
    n = len(idx)

    xy = np.array([[dets[i].get('global_center', [0,0])[0] * pixel_size,
                     dets[i].get('global_center', [0,0])[1] * pixel_size] for i in idx])

    if n >= 3:
        try:
            hull = ConvexHull(xy)
            area = hull.volume
        except:
            area = n * 100
    else:
        area = n * 100

    total_sig = sum(total_n[i] for i in idx)
    sig_per_area = total_sig / max(area, 1)
    sig_per_cell = total_sig / n

    gcg_mean = np.mean([gcg_n[i] for i in idx])
    ins_mean = np.mean([ins_n[i] for i in idx])
    sst_mean = np.mean([sst_n[i] for i in idx])

    islet_metrics.append({
        'id': iid, 'n': n, 'area': area, 'total_sig': total_sig,
        'sig_per_area': sig_per_area, 'sig_per_cell': sig_per_cell,
        'gcg': gcg_mean, 'ins': ins_mean, 'sst': sst_mean
    })

    print(f"{iid:>6} {n:>6} {area:>10.0f} {total_sig:>10.1f} {sig_per_area:>10.4f} {sig_per_cell:>10.3f} {gcg_mean:>6.3f} {ins_mean:>6.3f} {sst_mean:>6.3f}")

sig_per_area_arr = np.array([m['sig_per_area'] for m in islet_metrics])
sig_per_cell_arr = np.array([m['sig_per_cell'] for m in islet_metrics])

print("\n=== Otsu on signal/area ===")
otsu_spa = threshold_otsu(sig_per_area_arr)
high_spa = sig_per_area_arr >= otsu_spa
print(f"Threshold: {otsu_spa:.4f}")
print(f"True islets: {[islet_metrics[i]['id'] for i in range(len(islet_metrics)) if high_spa[i]]}")
print(f"Rejected:    {[islet_metrics[i]['id'] for i in range(len(islet_metrics)) if not high_spa[i]]}")

print("\n=== Otsu on signal/cell ===")
otsu_spc = threshold_otsu(sig_per_cell_arr)
high_spc = sig_per_cell_arr >= otsu_spc
print(f"Threshold: {otsu_spc:.3f}")
print(f"True islets: {[islet_metrics[i]['id'] for i in range(len(islet_metrics)) if high_spc[i]]}")
print(f"Rejected:    {[islet_metrics[i]['id'] for i in range(len(islet_metrics)) if not high_spc[i]]}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ids = [m['id'] for m in islet_metrics]
spa = [m['sig_per_area'] for m in islet_metrics]
colors = ['#33cc33' if s >= otsu_spa else '#ff3333' for s in spa]
ax.barh(range(len(ids)), spa, color=colors)
ax.set_yticks(range(len(ids)))
ax.set_yticklabels([f"I{i} ({islet_metrics[j]['n']}c)" for j, i in enumerate(ids)])
ax.axvline(otsu_spa, color='white', ls='--', lw=2, label=f"Otsu={otsu_spa:.4f}")
ax.set_xlabel('Total signal / area')
ax.set_title('Signal density per islet')
ax.legend()

ax = axes[1]
spc = [m['sig_per_cell'] for m in islet_metrics]
colors2 = ['#33cc33' if s >= otsu_spc else '#ff3333' for s in spc]
ax.barh(range(len(ids)), spc, color=colors2)
ax.set_yticks(range(len(ids)))
ax.set_yticklabels([f"I{i} ({islet_metrics[j]['n']}c)" for j, i in enumerate(ids)])
ax.axvline(otsu_spc, color='white', ls='--', lw=2, label=f"Otsu={otsu_spc:.3f}")
ax.set_xlabel('Mean signal / cell')
ax.set_title('Mean signal per cell per islet')
ax.legend()

fig.suptitle('Islet-level Otsu: true vs spurious islets', fontsize=14)
fig.tight_layout()
out = '/fs/pool/pool-mann-edwin/islet_output/2025_09_03_30610012_BS-100_20260220_211212_10pct/islet_otsu.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
