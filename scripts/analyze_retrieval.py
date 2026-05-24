"""Analyze retrieval failure cases."""
import json
import numpy as np
import torch
import sys
from pathlib import Path
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))
from regression.moe_config import SceneType, SCENE_TYPE_NAMES

r = np.load('output/regression/retrieval_results.npz')
dists = r['distances']

with open('output/regression/test_images.json') as f:
    test_info = json.load(f)
paths = list(test_info.keys())

# Print worst cases
sorted_idx = np.argsort(dists)[::-1]
print('=== TOP 20 WORST CASES (>200km) ===')
for i in sorted_idx[:20]:
    path = paths[i]
    name = path.replace('\\', '/').split('/')[-1]
    print(f'  {dists[i]:.0f}km  {name[:80]}')

# Distribution
print(f'\n=== ERROR DISTRIBUTION (n={len(dists)}) ===')
for thresh in [10, 25, 50, 100, 200, 500, 1000, 99999]:
    count = (dists < thresh).sum() if thresh < 99999 else len(dists)
    pct = count / len(dists) * 100
    label = f'<{thresh}' if thresh < 99999 else 'total'
    print(f'  {label}km: {count} ({pct:.1f}%)')

# Per-scene
with open('output/regression/scene_labels.json') as f:
    path_labels = json.load(f)
test_labels = torch.full((len(dists),), SceneType.OTHER, dtype=torch.long)
for idx, path in enumerate(paths):
    if path in path_labels:
        test_labels[idx] = int(path_labels[path])

for c in range(4):
    mask = test_labels.numpy() == c
    if mask.any():
        c_dists = dists[mask]
        bad = (c_dists > 200)
        name = SCENE_TYPE_NAMES.get(c, f"expert{c}")
        print(f'\n  {name}: {bad.sum()}/{c_dists.shape[0]} >200km '
              f'(mean={c_dists.mean():.0f}km, median={np.median(c_dists):.0f}km, max={c_dists.max():.0f}km)')
