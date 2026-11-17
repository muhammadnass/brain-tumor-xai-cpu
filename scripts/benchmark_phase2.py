"""
benchmark_phase2.py
====================
Measures the timing figures needed for the manuscript (items a–g).

Run from project root with venv active:
    python scripts/benchmark_phase2.py

Outputs a JSON report to results/metrics/benchmark_phase2.json
"""

import sys
import json
import time
import statistics
from pathlib import Path

import numpy as np
import torch
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MODELS_DIR, RESULTS_DIR
from ai_models.segmentation import create_segmentation_model
from xai_explainers.gradient_methods import VanillaGradient, IntegratedGradients
from xai_explainers.mesh_generator import create_mesh_generator

FEATURES = (16, 32, 64, 128, 256)
CHECKPOINT = MODELS_DIR / "segmentation" / "unet3d_cv_fold1.pth"
N_RUNS = 5
DEVICE = "cpu"

def timed_runs(fn, n=N_RUNS):
    """Run fn() n times and return (times_list, mean, std)."""
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        print(f"    run {i+1}/{n}: {times[-1]:.2f}s")
    return times, statistics.mean(times), statistics.stdev(times)

def main():
    print("=" * 60)
    print("PHASE 2 BENCHMARK — Neurosurgery XAI")
    print("=" * 60)

    # ---- Load model ----
    print("\n[1] Loading model…")
    model = create_segmentation_model(
        device=DEVICE, checkpoint_path=str(CHECKPOINT), features=FEATURES
    )
    model.eval()

    # ---- Dummy input ----
    # Random-noise input for inference/XAI timing (model will still run)
    x = torch.randn(1, 4, 128, 128, 128, device=DEVICE)
    print("    Input shape:", x.shape)

    # Synthetic binary mask for mesh timing (random noise produces no voxels
    # above threshold; we need a realistic-sized blob to benchmark marching cubes)
    rng = np.random.default_rng(42)
    synthetic_mask = np.zeros((128, 128, 128), dtype=np.float32)
    # Ellipsoid tumour, roughly 10,000 voxels — typical for BraTS at 128^3
    cx, cy, cz = 64, 64, 64
    rx, ry, rz = 14, 12, 11
    for xi in range(cx-rx, cx+rx):
        for yi in range(cy-ry, cy+ry):
            for zi in range(cz-rz, cz+rz):
                if ((xi-cx)/rx)**2 + ((yi-cy)/ry)**2 + ((zi-cz)/rz)**2 <= 1.0:
                    synthetic_mask[xi, yi, zi] = 0.8
    print(f"    Synthetic mask: {int((synthetic_mask > 0.5).sum()):,} voxels above threshold")

    results = {}

    # ---- a. Segmentation inference ----
    print(f"\n[a] Segmentation inference ({N_RUNS} runs)…")
    def _seg():
        with torch.no_grad():
            model(x)
    times, mean, std = timed_runs(_seg)
    results['a_segmentation_inference'] = {
        'mean_s': round(mean, 3), 'std_s': round(std, 3), 'runs': times
    }
    print(f"    RESULT: {mean:.2f} ± {std:.2f} s")

    # Get a real segmentation for downstream tasks
    with torch.no_grad():
        seg = model(x)
    tumor_mask = seg[0, 1].cpu().numpy()
    edema_mask = seg[0, 2].cpu().numpy()

    # ---- Vanilla Gradient (needed for XAI + mesh) ----
    print("\n[VG] Vanilla Gradient (single run for downstream)…")
    vg = VanillaGradient(model)
    attn_vg = vg.generate(x, target_class=1)
    print(f"    attn_vg shape: {attn_vg.shape}")

    # ---- b. Preview mesh step_size=4 ----
    print(f"\n[b] Preview mesh (step_size=4, {N_RUNS} runs) — synthetic mask…")
    mg = create_mesh_generator(target_vertex_count=50_000)
    # Use synthetic mask and a uniform attention map (all-zero attention still
    # exercises the full marching-cubes + glTF pipeline)
    synthetic_attn = np.zeros((128, 128, 128), dtype=np.float32)
    def _preview_mesh():
        mg.generate_full_segmentation_3d(
            tumor_mask=synthetic_mask,
            edema_mask=synthetic_mask * 0.6,  # smaller edema blob
            attention_map=synthetic_attn,
            step_size=4,
        )
    times, mean, std = timed_runs(_preview_mesh)
    results['b_preview_mesh_step4'] = {
        'mean_s': round(mean, 3), 'std_s': round(std, 3), 'runs': times,
        'note': 'Measured on synthetic ellipsoid mask (~10k voxels); random-noise input skips marching cubes'
    }
    print(f"    RESULT: {mean:.2f} ± {std:.2f} s")

    # ---- c. Full mesh step_size=2 ----
    print(f"\n[c] Full mesh (step_size=2, {N_RUNS} runs) — synthetic mask…")
    def _full_mesh():
        mg.generate_full_segmentation_3d(
            tumor_mask=synthetic_mask,
            edema_mask=synthetic_mask * 0.6,
            attention_map=synthetic_attn,
            step_size=2,
        )
    times, mean, std = timed_runs(_full_mesh)
    results['c_full_mesh_step2'] = {
        'mean_s': round(mean, 3), 'std_s': round(std, 3), 'runs': times,
        'note': 'Measured on synthetic ellipsoid mask (~10k voxels)'
    }
    print(f"    RESULT: {mean:.2f} ± {std:.2f} s")

    # ---- d. Integrated Gradients ----
    print(f"\n[d] Integrated Gradients steps=20 ({N_RUNS} runs)…")
    ig = IntegratedGradients(model, steps=20)
    def _ig():
        ig.generate(x, target_class=1)
    times, mean, std = timed_runs(_ig)
    results['d_integrated_gradients_steps20'] = {
        'mean_s': round(mean, 3), 'std_s': round(std, 3), 'runs': times
    }
    print(f"    RESULT: {mean:.2f} ± {std:.2f} s")

    # ---- f. Peak RAM during segmentation (psutil RSS) ----
    print("\n[f] Peak RAM during segmentation…")
    if PSUTIL_AVAILABLE:
        proc = psutil.Process()
        rss_before = proc.memory_info().rss / 1e6
        with torch.no_grad():
            model(x)
        rss_after = proc.memory_info().rss / 1e6
        peak_mb = rss_after  # RSS at peak (right after forward pass)
        delta_mb = rss_after - rss_before
        results['f_peak_ram_mb'] = {
            'rss_before_mb': round(rss_before, 1),
            'rss_after_mb':  round(rss_after, 1),
            'delta_mb':      round(delta_mb, 1),
            'note': 'psutil RSS — includes PyTorch tensor memory; measured immediately after forward pass'
        }
        print(f"    RSS before: {rss_before:.1f} MB  |  after: {rss_after:.1f} MB  |  delta: {delta_mb:.1f} MB")
    else:
        print("    psutil not installed — install with: pip install psutil")
        results['f_peak_ram_mb'] = {'note': 'psutil not available — install and rerun'}
        peak_mb = None

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (for manuscript)")
    print("=" * 60)
    print(f"  a. Segmentation inference:  {results['a_segmentation_inference']['mean_s']:.2f} ± {results['a_segmentation_inference']['std_s']:.2f} s")
    print(f"  b. Preview mesh (step=4):   {results['b_preview_mesh_step4']['mean_s']:.2f} ± {results['b_preview_mesh_step4']['std_s']:.2f} s  [synthetic mask]")
    print(f"  c. Full mesh   (step=2):    {results['c_full_mesh_step2']['mean_s']:.2f} ± {results['c_full_mesh_step2']['std_s']:.2f} s  [synthetic mask]")
    print(f"  d. IG (steps=20):           {results['d_integrated_gradients_steps20']['mean_s']:.2f} ± {results['d_integrated_gradients_steps20']['std_s']:.2f} s")
    ram = results['f_peak_ram_mb']
    if 'rss_after_mb' in ram:
        print(f"  f. Peak RAM (psutil RSS):   {ram['rss_after_mb']:.1f} MB total process  (Δ {ram['delta_mb']:.1f} MB during inference)")
    else:
        print(f"  f. Peak RAM:                {ram.get('note', 'unavailable')}")
    print()
    print("  e. End-to-end (upload→3D preview visible):")
    print("     Measure with browser DevTools Network waterfall, 5 runs.")
    print("     (Cannot be measured server-side — requires browser + tunnel.)")
    print()
    print("  g. Static asset sizes:")
    print("     Measure in browser DevTools → Network → filter JS/CSS,")
    print("     compare before/after enabling Cloudflare Auto Minify.")
    print()
    print("  h. Cloudflare Tunnel URL: printed by start_tunnel.sh at startup.")
    print("  i. Admin portal: will be at /admin path (Phase 2, item 2).")
    print("  j. DB column names: defined when evaluation module is built.")

    out = RESULTS_DIR / "metrics" / "benchmark_phase2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out}")

if __name__ == '__main__':
    main()
