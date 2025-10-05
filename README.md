# RANSAC Segmentation — GPU‑accelerated plane fitting (single & batched)

A compact, production‑oriented pair of PyTorch RANSAC plane‑fitting scripts:

- `ransac_segmentation.py` — classic per‑iteration RANSAC (GPU‑aware).
- `ransac_segmentation_batched.py` — vectorized/batched RANSAC that evaluates many hypotheses in parallel for speed.

Both scripts accept 3D point clouds and return a fitted plane (coefficients), inlier indices, and outlier indices. They include SVD refinement of the best inlier set and example/demo usage.

---

## Table of contents

- Overview
- Quick highlights
- Contract (inputs / outputs / errors)
- Installation
- Quickstart (copy‑paste examples)
- API reference
- Algorithm & math
- Performance, tuning & GPU guidance
- Edge cases & robustness
- Troubleshooting
- Tests & benchmarks (quick guides)
- Recommended next steps & TODO
- License

---

## Overview
<img width="1056" height="992" alt="image" src="https://github.com/user-attachments/assets/e68ec5e2-e44f-4946-845a-d3cb1101b165" />

RANSAC Segmentation provides two complementary implementations of RANSAC plane fitting using PyTorch:

- `fit_plane_ransac(...)` — iterative RANSAC: simple, easy to understand, good for moderate point‑clouds.
- `fit_plane_ransac_batch(...)` — batched RANSAC: creates many hypotheses at once and evaluates them in parallel using tensor operations (faster on GPU; higher memory use).

Both:
- Use PyTorch tensors and will use CUDA when `torch.cuda.is_available()` returns `True`.
- Refine the best model by computing the centroid of inliers and applying SVD to estimate a precise plane normal.
- Return plane parameters as the canonical plane equation `ax + by + cz + d = 0` and lists of inlier/outlier indices.

---

## Quick highlights

- GPU‑accelerated when CUDA is available.
- Batched implementation trades memory for large speedups on modern GPUs.
- SVD refinement for numerically robust plane normals.
- Example/demo code included in both files for quick experimentation.

---

## Contract

- Inputs:
  - `points`: Nx3 array‑like (Python list, NumPy array, or PyTorch tensor). Each row is `(x, y, z)`.
  - Iteration and threshold parameters (see API reference).

- Outputs:
  - `best_plane`: `[a, b, c, d]` — plane coefficients normalized so that `[a, b, c]` is a unit normal (approximately).
  - `best_inliers`: list of indices of points considered inliers.
  - `best_outliers`: list of indices considered outliers.

- Error modes / failure cases:
  - No valid plane found: current scripts may return `None` for `best_plane` in failure cases. See Troubleshooting & Robustness for recommended handling.
  - OOM on batched mode: reduce `iterations_per_batch` or use CPU mode.

---

## Installation

Minimum dependencies:
- Python 3.8+
- PyTorch (CPU or CUDA) — version >= 1.8 recommended
- NumPy

Install using pip (example: CPU‑only PyTorch) in PowerShell:

```powershell
# CPU-only PyTorch (Windows PowerShell example)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy
```

For CUDA‑enabled PyTorch, pick the correct torch wheel from https://pytorch.org/ (matching your CUDA version). Example (replace cu118 with your CUDA tag):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy
```

If you'd like, create a `requirements.txt` with `torch` and `numpy` and run `pip install -r requirements.txt`.

---

## Quickstart — Try it now

Run the included examples in PowerShell:

```powershell
# Run single-iteration RANSAC example
python .\ransac_segmentation.py

# Run batched RANSAC example
python .\ransac_segmentation_batched.py
```

Programmatic usage from other scripts:

```python
import numpy as np
from ransac_segmentation import fit_plane_ransac
from ransac_segmentation_batched import fit_plane_ransac_batch

points = np.random.randn(1000, 3)
plane, inliers, outliers = fit_plane_ransac(points, max_iterations=500, distance_threshold=0.05, min_inliers=50)
print("Plane:", plane)

plane2, inliers2, outliers2 = fit_plane_ransac_batch(points, max_iterations=1000, distance_threshold=0.05, min_inliers=50, iterations_per_batch=64)
print("Plane (batched):", plane2)
```

---

## API Reference

All functions accept point clouds as Nx3 arrays (list/NumPy/PyTorch). They convert to `torch.float32` tensors and move to GPU automatically if available.

1) `fit_plane_ransac(points, max_iterations=1000, distance_threshold=0.01, min_inliers=3)`

- `points`: Nx3
- `max_iterations`: int — number of RANSAC trials
- `distance_threshold`: float — max perpendicular distance to count as inlier
- `min_inliers`: int — minimum inlier count to accept a model

Returns: `(best_plane, best_inliers, best_outliers)`
- `best_plane`: `[a, b, c, d]` (floats)
- `best_inliers`: `list[int]`
- `best_outliers`: `list[int]`

2) `fit_plane_ransac_batch(points, max_iterations=1000, distance_threshold=0.01, min_inliers=3, iterations_per_batch=40, epsilon=1e-8)`

- `iterations_per_batch`: how many hypotheses are generated & evaluated in parallel per loop pass.
- `epsilon`: small stabilizer to avoid divide‑by‑zero in normals normalization.

Returns same outputs. Batched version uses more GPU memory but runs faster on larger point clouds.

---

## Algorithm & Math

Plane model:

- Plane equation: `a x + b y + c z + d = 0`.
- Normalize so that sqrt(a^2 + b^2 + c^2) ≈ 1. Distance from point p=(x,y,z) to plane is:

$$
\mathrm{dist}(p, [a,b,c,d]) = \frac{|a x + b y + c z + d|}{\sqrt{a^2 + b^2 + c^2}}.
$$

RANSAC overview:
- Randomly sample minimal points (3 points define a plane in 3D).
- Compute plane from the sample using cross product:
  - v1 = p2 − p1, v2 = p3 − p1
  - normal = v1 × v2
- Use distance threshold to count inliers.
- Keep best model (largest inlier set).
- Refine best model: compute centroid of inliers and apply SVD on (inliers − centroid); the singular vector corresponding to smallest singular value is the plane normal (least squares plane fit).

Batched variant:
- Generates many (B) triples at once and computes a B×N distance matrix to evaluate inliers efficiently on GPU.

---

## Performance & Tuning

- Single‑iteration approach:
  - Memory‑light. Good when GPU memory is limited.
  - Time cost scales with `max_iterations × N` (points), but inner ops are tensorized for distance evaluation.

- Batched approach:
  - Much faster when evaluating many hypotheses on GPU since it vectorizes hypothesis generation and evaluation.
  - Memory use is O(B × N) for the distance matrix — this can be large if both `B` (`iterations_per_batch`) and `N` are big.
  - Tune `iterations_per_batch` to the largest value that fits your GPU memory. Start at 32/64 and increase if memory allows.

Practical tips:
- For `N ≈ 100k`, avoid very large `iterations_per_batch` (use ~16–64).
- When GPU memory is small, prefer the single‑iteration implementation or lower batch sizes.
- Use `with torch.no_grad():` around any loops if you integrate these functions into training pipelines (they don’t need gradients).

---

## Edge cases & robustness

- Degenerate samples: if the three sampled points are colinear or coincident, cross product norm is near zero. Batched code uses `epsilon` to stabilize; single variant should check norms > 0.
- No model found: if no sample produces enough inliers (based on `min_inliers`), `best_plane` might be `None`. Guard your callers or update the functions to return `(None, [], [])` instead of crashing.
- Reproducibility: random sampling uses Python `random` in the single version and `torch.randint` in the batched version. To reproduce:
  - `random.seed(42)` for single variant
  - `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` for the batched variant

---

## Troubleshooting

- "No CUDA available" but you expected GPU:
  - Confirm your PyTorch install supports CUDA and GPU drivers are installed. In Python:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

- Out‑of‑memory (OOM) in batched mode:
  - Lower `iterations_per_batch`.
  - Switch to CPU by forcing `device = torch.device('cpu')` (not recommended for large workloads).
- `best_plane` is `None` or code crashes converting it to a list:
  - Add a guard in your usage:

```python
if best_plane is None:
    print("No plane found. Try different parameters.")
else:
    print(best_plane)
```

---

## Tests & benchmarks

Add a minimal local benchmark to compare single vs batched:

```python
# quick_benchmark.py
import numpy as np
import time
from ransac_segmentation import fit_plane_ransac
from ransac_segmentation_batched import fit_plane_ransac_batch

# generate synthetic data similar to examples in the scripts
# then time both functions and compare inlier counts / time
```

Run:

```powershell
python .\quick_benchmark.py
```

Suggested tests:
- Unit test a synthetic planar dataset with known coefficients and assert fitted coefficients are within tolerance.
- Regression test for varying outlier ratios and batch sizes.
- Memory and time profiling for your target environment.

---

## Recommended next steps / TODO

- Add `requirements.txt` and this `README.md` to the repo (done).
- Add a safety guard returning `(None, [], [])` when no valid plane is found.
- Provide optional CLI wrappers with `argparse` for common workflows.
- Add unit tests using `pytest`:
  - happy path: synthetic plane with noise
  - edge case: degenerate / insufficient points
- Add a small benchmarking harness that logs GPU memory usage and timings.
- Add a visualizer script to plot inliers/outliers (matplotlib or open3d).

---

## License

MIT License
Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy
