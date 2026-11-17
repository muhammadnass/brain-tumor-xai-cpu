"""
Flask API server — Phase 2 optimised build
Implements optimisation items 1–7, 11–12 from the performance spec.

Key changes from Phase 1:
  1. Flask-Compress: gzip all responses > 1 kB
  2. ThreadPoolExecutor: segmentation pipeline is fully non-blocking
  3. /api/health: sub-5 ms, no model calls
  4. SSE /api/progress/<job_id>: real-time progress stream
  5. File-hash result cache: skip re-inference for identical uploads
  6. Static-asset cache headers set in Flask config
  7. Model loaded once at startup; pre-warm run at import time
  8. Dual-LOD mesh: preview at step_size=4 in main thread,
     full at step_size=2 in background (see mesh_generator)
 11. Integrated Gradients computed in background immediately after
     segmentation completes; endpoint signals ig_ready=true when done
 12. XAI slice images served as WebP (quality=85, ~40-70 kB each)

Architecture
-----------
POST /api/predict/segment
    → saves files to tmpdir, submits _run_pipeline() to executor
    → returns {job_id, status:"queued"} immediately

GET  /api/jobs/<job_id>/status
    → returns {status, progress, message, preview_ready, ig_ready, full_mesh_ready}

GET  /api/jobs/<job_id>/result
    → returns full result once preview_ready; includes ig/full_mesh if done

GET  /api/progress/<job_id>       (SSE)
    → streams {status, progress, message, …} every 2 s until done/error

GET  /api/xai/slice/<job_id>/<slice_idx>?method=vg|ig
    → returns WebP-encoded overlay for axial slice (item 12, 13)

All previous endpoints (/api/plan/surgical, /api/comparison/*) unchanged.
"""

import io
import os
import json
import uuid
import time
import base64
import hashlib
import shutil
import tempfile
import threading
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_compress import Compress
from PIL import Image
from werkzeug.utils import secure_filename

from config import APIConfig, XAIConfig, MODELS_DIR, RESULTS_DIR
from ai_models.segmentation import create_segmentation_model
from xai_explainers.attention_visualizer import FeatureImportanceVisualizer
from xai_explainers.gradient_methods import VanillaGradient, IntegratedGradients
from data_processing.preprocessor import preprocess_mri_files, MRIPreprocessor
from evaluation.routes import eval_bp
from evaluation.database import init_db
from surgical_planning.cc_filter import apply_cc_filter
try:
    from surgical_planning.risk_calculator import compute_risk_regions
    RISK_CALC_AVAILABLE = True
except Exception:
    RISK_CALC_AVAILABLE = False

try:
    from xai_explainers.mesh_generator import (
        create_mesh_generator, generate_skull_surface,
        generate_brain_surface, generate_ventricles,
        align_head_to_brain,
    )
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False
    generate_skull_surface = None
    generate_brain_surface = None
    generate_ventricles = None
    align_head_to_brain = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# FLASK + GZIP COMPRESSION (item 1)
# ===========================================================================
app = Flask(__name__)
app.config.from_object(APIConfig)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

compress = Compress()
app.config['COMPRESS_MIN_SIZE'] = 1024       # compress anything > 1 kB
app.config['COMPRESS_LEVEL'] = 6             # level 6 = good ratio, fast enough
app.config['COMPRESS_ALGORITHM'] = ['gzip']  # gzip first; brotli if installed
compress.init_app(app)

# ===========================================================================
# CORS — allow browser JS (Streamlit iframe) to call the Flask API
# ===========================================================================
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Static asset cache headers (item 6)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000

# Register Phase 2 evaluation blueprint and initialise study database
app.register_blueprint(eval_bp, url_prefix='/eval')
init_db()

# ===========================================================================
# MODEL LOADING — ONCE AT STARTUP (item 7)
# ===========================================================================
logger.info("=" * 60)
logger.info("Starting model initialisation…")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURES = (16, 32, 64, 128, 256)
CHECKPOINT = MODELS_DIR / "segmentation" / "unet3d_cv_fold1.pth"
THRESHOLD = 0.5

segmentation_model = None
try:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    segmentation_model = create_segmentation_model(
        device=DEVICE,
        checkpoint_path=str(CHECKPOINT),
        features=FEATURES,
    )
    n = sum(p.numel() for p in segmentation_model.parameters())
    if n > 30_000_000:
        raise RuntimeError(f"Wrong architecture: {n:,} params, expected ~22.6M")
    logger.info(f"Model loaded — ready.  [{n:,} params | device={DEVICE}]")
except Exception as exc:
    logger.error(f"❌ Model load failed: {exc}")
    segmentation_model = None

# Pre-warm in a separate try so a failure here never clobbers the loaded model.
# The U-Net has 5 encoder stages × MaxPool3d(2) = 32× spatial downsampling.
# Minimum safe input: 64^3 → bottleneck 2^3 (>1 per channel for BatchNorm).
if segmentation_model is not None:
    try:
        _dummy = torch.zeros(1, 4, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            _ = segmentation_model(_dummy)
        del _dummy
        logger.info("Pre-warm pass complete — first inference will be fast.")
    except Exception as _pw_exc:
        logger.warning(
            f"⚠️ Pre-warm failed ({_pw_exc}); model still loaded and usable."
        )

mesh_generator = None
if MESH_AVAILABLE:
    try:
        mesh_generator = create_mesh_generator(target_vertex_count=50_000)
        logger.info("✅ Mesh generator ready")
    except Exception as exc:
        logger.warning(f"⚠️ Mesh generator unavailable: {exc}")
logger.info("=" * 60)

# ===========================================================================
# THREAD POOL (item 2)
# ===========================================================================
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="seg")

# ===========================================================================
# JOB REGISTRY (items 2, 4)
# ===========================================================================
_jobs: dict = {}
_jobs_lock = threading.Lock()
JOB_TTL = 3600  # seconds

def _update_job(job_id: str, **kw):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)

def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))

def _cleanup_old_jobs():
    cutoff = datetime.utcnow() - timedelta(seconds=JOB_TTL)
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items()
                 if j.get('_created', datetime.utcnow()) < cutoff]
        for jid in stale:
            _wipe_tmpdir(_jobs[jid].get('_tmpdir'))
            del _jobs[jid]
    if stale:
        logger.info(f"Pruned {len(stale)} expired jobs")

def _wipe_tmpdir(tmpdir):
    if tmpdir and Path(tmpdir).exists():
        shutil.rmtree(tmpdir, ignore_errors=True)

# ===========================================================================
# FILE-HASH RESULT CACHE (item 5)
# ===========================================================================
_cache: dict = {}          # hash → result_preview dict
_cache_lock = threading.Lock()
_CACHE_MAX = 5

def _cache_get(key: str):
    with _cache_lock:
        return _cache.get(key)

def _cache_put(key: str, val):
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            del _cache[next(iter(_cache))]
        _cache[key] = val

def _hash_files(byte_list: list) -> str:
    h = hashlib.sha256()
    for b in byte_list:
        h.update(b)
    return h.hexdigest()

# ===========================================================================
# PATHS
# ===========================================================================
COMPARISONS_DIR = RESULTS_DIR / "expert_evaluations"
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_FILE = COMPARISONS_DIR / "blind_comparisons.jsonl"

# ===========================================================================
# UTILITIES
# ===========================================================================

def allowed_file(fn: str) -> bool:
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def ok(data: dict = None, code: int = 200):
    r = {'status': 'success'}
    if data:
        r.update(data)
    return jsonify(r), code

def err(msg: str, code: int = 500):
    return jsonify({'status': 'error', 'error': msg}), code

def _norm_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)

def _webp(image_bgr: np.ndarray, quality: int = 85) -> tuple[str, str]:
    """Encode BGR image as WebP; return (base64_str, mime_type). (item 12)"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='WEBP', quality=quality, method=4)
    return base64.b64encode(buf.getvalue()).decode(), 'image/webp'

def _make_overlay(attention_map: np.ndarray, background_vol: np.ndarray,
                  slice_idx: int) -> np.ndarray:
    """Build a single axial overlay image (BGR uint8)."""
    n_slices = background_vol.shape[0]
    slice_idx = max(0, min(slice_idx, n_slices - 1))
    bg = _norm_uint8(background_vol[slice_idx])
    attn = attention_map[slice_idx]
    return FeatureImportanceVisualizer.create_overlay(bg, attn, alpha=XAIConfig.overlay_alpha)

# ===========================================================================
# BACKGROUND PIPELINE (items 2, 8, 11)
# ===========================================================================

def _run_pipeline(job_id: str, file_paths: dict,
                  display_path: str | None, file_hash: str):
    """
    Full pipeline running in a background thread.
    Stages: preprocess → segment → VG XAI → preview mesh
    Then background stages: IG XAI → full-LOD mesh
    """
    try:
        # ---- Stage 1: Preprocess ----
        _update_job(job_id, status='preprocessing', progress=0.05,
                    message='Preprocessing MRI data…')
        mri_vol = preprocess_mri_files(file_paths)   # numpy [4,D,H,W]
        if mri_vol is None:
            _update_job(job_id, status='error', error='Preprocessing failed', progress=0)
            return

        # ---- Stage 2: Inference ----
        _update_job(job_id, status='segmenting', progress=0.20,
                    message='Running segmentation model…')
        dev = next(segmentation_model.parameters()).device
        tensor = torch.from_numpy(mri_vol).float().unsqueeze(0).to(dev)

        t0 = time.time()
        with torch.no_grad():
            seg = segmentation_model.predict(tensor)
        t_seg = time.time() - t0
        logger.info(f"[{job_id}] Inference {t_seg:.2f}s")

        tumor_mask = seg[0, 1].cpu().numpy()
        edema_mask = seg[0, 2].cpu().numpy()

        # Connected-component post-processing: remove scattered false positives
        tumor_mask = apply_cc_filter(tumor_mask, THRESHOLD)
        edema_mask = apply_cc_filter(edema_mask, THRESHOLD)

        tumor_vox = int((tumor_mask > THRESHOLD).sum())
        edema_vox = int((edema_mask > THRESHOLD).sum())

        # Background volume for overlays (use skull-inclusive scan if provided)
        if display_path:
            pp = MRIPreprocessor(target_shape=(128, 128, 128))
            raw = pp.load_nifti(display_path)
            bg_vol = pp.resize(raw) if raw is not None else mri_vol[0]
            # Register the head scan into BraTS space so the skull surface
            # and slice-navigator background line up with the segmentation.
            if raw is not None and align_head_to_brain is not None:
                try:
                    _update_job(job_id, message='Registering head scan to brain…')
                    bg_vol = align_head_to_brain(bg_vol, mri_vol[0])
                except Exception as reg_exc:
                    logger.warning(f"[{job_id}] Head registration failed: {reg_exc}")
        else:
            bg_vol = mri_vol[0]   # T1 channel (FLAIR channel 0)

        # ---- Stage 3: Vanilla Gradient ----
        _update_job(job_id, status='xai_vg', progress=0.45,
                    message='Generating Vanilla Gradient map…')
        attn_vg = None
        xai_vg = {}
        try:
            t0 = time.time()
            vg = VanillaGradient(segmentation_model)
            attn_vg = vg.generate(tensor, target_class=1)  # [D,H,W]
            t_vg = time.time() - t0
            logger.info(f"[{job_id}] VG {t_vg:.2f}s")

            mid = attn_vg.shape[0] // 2
            overlay_vg = _make_overlay(attn_vg, bg_vol, mid)
            img_b64, mime = _webp(overlay_vg)
            xai_vg = {
                'attention_image_base64': img_b64,
                'attention_image_mime': mime,
                'explanation_text': FeatureImportanceVisualizer.generate_explanation_text(attn_vg),
                'attention_statistics': FeatureImportanceVisualizer.get_attention_statistics(attn_vg),
                'vg_time_s': t_vg,
            }
        except Exception as exc:
            logger.error(f"[{job_id}] VG failed: {exc}")
            xai_vg = {'explanation_text': '⚠️ XAI unavailable for this case.'}

        # ---- Stage 4: Preview mesh (step_size=4) + skull surface (item 8 + Phase 2 item 1) ----
        _update_job(job_id, status='mesh_preview', progress=0.65,
                    message='Generating 3-D preview…')
        geo_preview = None
        if mesh_generator is not None and attn_vg is not None:
            try:
                t0 = time.time()
                geo_preview = mesh_generator.generate_full_segmentation_3d(
                    tumor_mask=tumor_mask, edema_mask=edema_mask,
                    attention_map=attn_vg,
                    tumor_threshold=THRESHOLD, edema_threshold=THRESHOLD,
                    step_size=4,
                )

                # Skull / scalp surface from non-skull-stripped T1 (Phase 2 item 1)
                # --- Skull surface (non-skull-stripped T1 = outer head) ---
                if display_path and generate_skull_surface and geo_preview:
                    try:
                        # bg_vol is the head scan already registered to BraTS space
                        skull_vol = bg_vol
                        if skull_vol is not None:
                            skull_glb = generate_skull_surface(
                                skull_vol, step_size=3
                            )
                            if skull_glb:
                                geo_preview['skull_mesh_gltf'] = \
                                    base64.b64encode(skull_glb).decode()
                                logger.info(
                                    f"[{job_id}] Skull surface added "
                                    f"({len(skull_glb):,} bytes)"
                                )
                    except Exception as skull_exc:
                        logger.warning(
                            f"[{job_id}] Skull surface failed: {skull_exc}"
                        )

                # --- Brain parenchyma surface (skull-stripped T1 ch0) ---
                if geo_preview and generate_brain_surface:
                    try:
                        brain_t1 = mri_vol[0]  # skull-stripped T1/FLAIR ch0
                        brain_glb = generate_brain_surface(
                            brain_t1, step_size=3
                        )
                        if brain_glb:
                            geo_preview['brain_mesh_gltf'] = \
                                base64.b64encode(brain_glb).decode()
                            logger.info(
                                f"[{job_id}] Brain surface added "
                                f"({len(brain_glb):,} bytes)"
                            )
                    except Exception as brain_exc:
                        logger.warning(
                            f"[{job_id}] Brain surface failed: {brain_exc}"
                        )

                # --- Ventricles (dark CSF regions in skull-stripped T1) ---
                if geo_preview and generate_ventricles:
                    try:
                        brain_t1 = mri_vol[0]
                        # brain_mask=None → derived from corner background
                        # (works with z-score normalised input)
                        vent_glb = generate_ventricles(
                            brain_t1, brain_mask=None, step_size=2
                        )
                        if vent_glb:
                            geo_preview['ventricle_mesh_gltf'] = \
                                base64.b64encode(vent_glb).decode()
                            logger.info(
                                f"[{job_id}] Ventricles added "
                                f"({len(vent_glb):,} bytes)"
                            )
                    except Exception as vent_exc:
                        logger.warning(
                            f"[{job_id}] Ventricles failed: {vent_exc}"
                        )

                logger.info(f"[{job_id}] Preview mesh {time.time()-t0:.2f}s")
            except Exception as exc:
                logger.error(f"[{job_id}] Preview mesh failed: {exc}")

        # ---- Build preview result ----
        result_preview = {
            'segmentation': {
                'tumor_volume_voxels': tumor_vox,
                'edema_volume_voxels': edema_vox,
            },
            'quality': {
                'tumor_confidence': float(tumor_mask.max()),
                'edema_confidence': float(edema_mask.max()),
                'quality_score': 'high' if tumor_mask.max() > 0.8 else 'medium',
            },
            'statistics': {
                'tumor_mean': float(tumor_mask.mean()),
                'tumor_std':  float(tumor_mask.std()),
                'tumor_max':  float(tumor_mask.max()),
            },
            'xai_vg': xai_vg,
            'xai_explanation': xai_vg,   # backward-compat alias
            '3d_preview': geo_preview,
            '3d_preview_available': geo_preview is not None,
            '3d_geometry': geo_preview,
            '3d_available': geo_preview is not None,
            'used_skull_display': display_path is not None,
            'inference_time_s': t_seg,
        }

        # Store raw arrays in job for slice serving (item 13)
        # _display_path is used by the /xai/slice endpoint to set skull_bg flag
        _update_job(job_id,
                    status='preview_ready',
                    progress=0.75,
                    message='Preview ready — computing full XAI…',
                    result_preview=result_preview,
                    _attn_vg=attn_vg,
                    _bg_vol=bg_vol,
                    _mri_vol=mri_vol,
                    _tumor_mask=tumor_mask,
                    _edema_mask=edema_mask,
                    _display_path=display_path)

        _cache_put(file_hash, {
            'result_preview': result_preview,
            '_attn_vg': attn_vg, '_bg_vol': bg_vol, '_mri_vol': mri_vol,
            '_tumor_mask': tumor_mask, '_edema_mask': edema_mask,
            '_display_path': display_path,
        })

        # ---- Stage 5: Integrated Gradients (background, item 11) ----
        _update_job(job_id, status='xai_ig', progress=0.78,
                    message='Computing Integrated Gradients…')
        xai_ig = _compute_ig(job_id, mri_vol, bg_vol, dev)
        if xai_ig:
            _update_job(job_id, xai_ig=xai_ig)

        # ---- Stage 6: Full-LOD mesh step_size=2 (item 8) ----
        _update_job(job_id, status='mesh_full', progress=0.90,
                    message='Generating full-resolution 3-D model…')
        geo_full = _compute_full_mesh(job_id, tumor_mask, edema_mask,
                                      xai_ig.get('_attn_map') if xai_ig else attn_vg)
        if geo_full:
            # Carry anatomical context meshes over from the preview
            for k in ('brain_mesh_gltf', 'ventricle_mesh_gltf', 'skull_mesh_gltf'):
                if geo_preview and geo_preview.get(k):
                    geo_full[k] = geo_preview[k]
            _update_job(job_id, mesh_full=geo_full)

        _update_job(job_id, status='done', progress=1.0, message='Complete.')
        logger.info(f"[{job_id}] Pipeline done")

    except Exception as exc:
        logger.error(f"[{job_id}] Pipeline crash: {exc}\n{traceback.format_exc()}")
        _update_job(job_id, status='error', error=str(exc), progress=0)
    finally:
        _wipe_tmpdir(_get_job(job_id).get('_tmpdir'))


def _compute_ig(job_id: str, mri_vol: np.ndarray,
                bg_vol: np.ndarray, device) -> dict | None:
    """Integrated Gradients computation (item 11)."""
    try:
        tensor = torch.from_numpy(mri_vol).float().unsqueeze(0).to(device)
        t0 = time.time()
        ig = IntegratedGradients(segmentation_model, steps=20)
        attn_ig = ig.generate(tensor, target_class=1)
        t_ig = time.time() - t0
        logger.info(f"[{job_id}] IG {t_ig:.2f}s")

        mid = attn_ig.shape[0] // 2
        overlay = _make_overlay(attn_ig, bg_vol, mid)
        img_b64, mime = _webp(overlay)
        return {
            'attention_image_base64': img_b64,
            'attention_image_mime': mime,
            'explanation_text': FeatureImportanceVisualizer.generate_explanation_text(attn_ig),
            'attention_statistics': FeatureImportanceVisualizer.get_attention_statistics(attn_ig),
            'ig_time_s': t_ig,
            '_attn_map': attn_ig,   # keep for full mesh colouring; stripped before JSON
        }
    except Exception as exc:
        logger.error(f"[{job_id}] IG failed: {exc}")
        return None


def _compute_full_mesh(job_id: str, tumor_mask: np.ndarray,
                       edema_mask: np.ndarray,
                       attn_map: np.ndarray | None) -> dict | None:
    """Full-LOD mesh at step_size=2 (item 8)."""
    if mesh_generator is None or attn_map is None:
        return None
    try:
        t0 = time.time()
        geo = mesh_generator.generate_full_segmentation_3d(
            tumor_mask=tumor_mask, edema_mask=edema_mask,
            attention_map=attn_map,
            tumor_threshold=THRESHOLD, edema_threshold=THRESHOLD,
            step_size=2,
        )
        logger.info(f"[{job_id}] Full mesh {time.time()-t0:.2f}s")
        return geo
    except Exception as exc:
        logger.error(f"[{job_id}] Full mesh failed: {exc}")
        return None


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Sub-5 ms health check (item 3). Used as Cloudflare health-check target."""
    return jsonify({'status': 'ok', 'model': segmentation_model is not None}), 200


@app.route('/api/predict/segment', methods=['POST'])
def predict_segment():
    """
    Async segmentation endpoint.
    Returns job_id immediately; client polls /api/jobs/<id>/status.
    """
    if segmentation_model is None:
        return err('Model not loaded', 503)

    required = ['T1', 'T2', 'FLAIR', 'T1c']
    for mod in required:
        if mod not in request.files:
            return err(f'Missing modality: {mod}', 400)

    # Read bytes for hashing before saving to disk (item 5)
    raw = {}
    for mod in required:
        f = request.files[mod]
        raw[mod] = f.read()
        f.seek(0)

    # Also read DisplayScan bytes for cache key — without this, submitting
    # the same 4 modalities + a new skull T1 would hit the old (skull-free)
    # cached result and never run skull surface generation. (#fix-skull-cache)
    display_raw = b''
    if 'DisplayScan' in request.files and request.files['DisplayScan'].filename:
        df_pre = request.files['DisplayScan']
        display_raw = df_pre.read()
        df_pre.seek(0)  # rewind so df.save() below still works

    file_hash = _hash_files([raw[m] for m in required] + [display_raw])
    cached = _cache_get(file_hash)
    if cached:
        logger.info(f"Cache hit {file_hash[:8]}")
        jid = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[jid] = {
                'status': 'done', 'progress': 1.0, 'message': 'From cache.',
                '_created': datetime.utcnow(),
                **cached,   # result_preview + slice arrays for the navigator
            }
        return ok({'job_id': jid, 'status': 'queued', 'cached': True})

    # Save to temp dir
    tmpdir = tempfile.mkdtemp()
    file_paths = {}
    for mod in required:
        f = request.files[mod]
        fp = os.path.join(tmpdir, secure_filename(f'{mod}_{f.filename}'))
        with open(fp, 'wb') as out:
            out.write(raw[mod])
        file_paths[mod] = fp

    display_path = None
    if 'DisplayScan' in request.files and request.files['DisplayScan'].filename:
        df = request.files['DisplayScan']
        dp = os.path.join(tmpdir, secure_filename(f'display_{df.filename}'))
        df.save(dp)
        display_path = dp

    jid = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[jid] = {
            'status': 'queued', 'progress': 0.0, 'message': 'Queued.',
            'result_preview': None, 'xai_ig': None, 'mesh_full': None,
            'error': None, '_created': datetime.utcnow(), '_tmpdir': tmpdir,
        }

    executor.submit(_run_pipeline, jid, file_paths, display_path, file_hash)
    _cleanup_old_jobs()
    return ok({'job_id': jid, 'status': 'queued'})


@app.route('/api/jobs/<job_id>/status', methods=['GET'])
def job_status(job_id):
    j = _get_job(job_id)
    if not j:
        return err('Job not found', 404)
    return ok({
        'job_id': job_id,
        'status':            j.get('status'),
        'progress':          j.get('progress', 0),
        'message':           j.get('message', ''),
        'error':             j.get('error'),
        'preview_ready':     j.get('result_preview') is not None,
        'ig_ready':          j.get('xai_ig') is not None,
        'full_mesh_ready':   j.get('mesh_full') is not None,
    })


@app.route('/api/jobs/<job_id>/result', methods=['GET'])
def job_result(job_id):
    j = _get_job(job_id)
    if not j:
        return err('Job not found', 404)
    if j.get('status') == 'error':
        return err(j.get('error', 'Unknown error'), 500)
    preview = j.get('result_preview')
    if preview is None:
        return jsonify({'status': 'processing', 'progress': j.get('progress', 0)}), 202

    result = dict(preview)

    # Merge IG if ready
    ig = j.get('xai_ig')
    if ig:
        ig_clean = {k: v for k, v in ig.items() if not k.startswith('_')}
        result['xai_ig'] = ig_clean

    # Merge full mesh if ready
    if j.get('mesh_full'):
        result['3d_full'] = j['mesh_full']
        result['3d_full_available'] = True
        # Upgrade 3d_geometry to full LOD
        result['3d_geometry'] = j['mesh_full']
        result['3d_available'] = True

    result['status'] = 'success'
    return jsonify(result), 200


@app.route('/api/progress/<job_id>', methods=['GET'])
def progress_sse(job_id):
    """SSE stream emitting progress every 2 s (item 4)."""
    def generate():
        for _ in range(180):          # max 6 min
            j = _get_job(job_id)
            if not j:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            payload = {
                'status':          j.get('status'),
                'progress':        j.get('progress', 0),
                'message':         j.get('message', ''),
                'preview_ready':   j.get('result_preview') is not None,
                'ig_ready':        j.get('xai_ig') is not None,
                'full_mesh_ready': j.get('mesh_full') is not None,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if payload['status'] in ('done', 'error'):
                return
            time.sleep(2)

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/xai/slice/<job_id>/<int:slice_idx>', methods=['GET'])
def xai_slice(job_id, slice_idx):
    """
    Return a WebP-encoded axial overlay slice (items 12, 13).
    method=vg (default) or method=ig
    """
    j = _get_job(job_id)
    if not j:
        return err('Job not found', 404)

    method = request.args.get('method', 'vg')

    if method == 'ig':
        ig = j.get('xai_ig')
        if ig is None:
            return jsonify({'status': 'processing', 'message': 'IG not yet ready'}), 202
        attn = ig.get('_attn_map')
    else:
        attn = j.get('_attn_vg')

    bg_vol = j.get('_bg_vol')

    if attn is None or bg_vol is None:
        return err('Slice data not available (job may have expired)', 410)

    n = attn.shape[0]
    slice_idx = max(0, min(slice_idx, n - 1))
    overlay = _make_overlay(attn, bg_vol, slice_idx)
    img_b64, mime = _webp(overlay)

    # skull_bg=True when the background was loaded from a separate DisplayScan
    # (non-skull-stripped T1) rather than the first modality channel.
    skull_bg = bool(j.get('_display_path'))

    return ok({'image_base64': img_b64, 'mime': mime,
               'slice_idx': slice_idx, 'n_slices': n, 'skull_bg': skull_bg})


# ===========================================================================
# SURGICAL PLAN (unchanged logic)
# ===========================================================================
@app.route('/api/plan/surgical', methods=['POST'])
def plan_surgical():
    """
    Volume-based complexity + atlas-based precise risk regions.
    Accepts optional tumor_mask_128 (base64 numpy) and t1_path for
    patient-specific risk computation; falls back to model-space centroid
    approximation when not provided.
    """
    try:
        data       = request.get_json()
        vol        = data.get('tumor_volume', 0)
        confidence = data.get('segmentation_confidence')
        job_id     = data.get('job_id')         # optional: fetch mask from job

        if vol < 8000:
            cx, mm, re, hr = 'simple',   3,  '95–100%', 'low'
        elif vol < 20000:
            cx, mm, re, hr = 'moderate', 5,  '85–95%',  'moderate'
        else:
            cx, mm, re, hr = 'complex',  10, '70–90%',  'moderate-high'

        # ---- Precise risk regions ----
        risk_data = {'high_risk_structures': [], 'low_risk_structures': [],
                     'method': 'unavailable', 'note': ''}
        if RISK_CALC_AVAILABLE and job_id:
            job = _get_job(job_id)
            tumor_mask = job.get('_tumor_mask')
            t1_path    = None          # skull-stripped T1 not yet stored per-job
            if tumor_mask is not None:
                try:
                    risk_data = compute_risk_regions(
                        t1_path=t1_path,
                        tumor_mask_128=tumor_mask,
                    )
                    logger.info(f"Risk regions computed via {risk_data['method']}")
                except Exception as re_exc:
                    logger.error(f"Risk calc failed: {re_exc}")

        return ok({
            'surgical_plan': {
                'tumor_volume_voxels':       vol,
                'complexity':                cx,
                'complexity_basis':          'tumor volume only',
                'recommended_margin_mm':     mm,
                'estimated_resection_extent': re,
            },
            'risk_assessment': {
                'hemorrhage_risk':           hr,
                'neurological_deficit_risk': 'low' if cx == 'simple' else 'moderate',
                'recurrence_risk':           'standard',
                'recommendation':            'Consult with neurosurgery team.',
            },
            'risk_regions': risk_data,
            'requires_neurosurgeon_review': True,
            'confidence': confidence,
        })
    except Exception as exc:
        return err(str(exc))


# ===========================================================================
# COMPARISON SAVE / LIST (unchanged)
# ===========================================================================
@app.route('/api/comparison/save', methods=['POST'])
def save_comparison():
    try:
        data = request.get_json()
        for f in ['evaluator_name', 'case_id', 'human_estimate', 'ai_result']:
            if f not in data:
                return err(f'Missing field: {f}', 400)
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            **{k: data.get(k) for k in [
                'evaluator_name', 'case_id', 'human_estimate', 'ai_result',
                'concordance', 'evaluator_notes', 'visualization_feedback',
            ]},
        }
        with open(COMPARISONS_FILE, 'a') as fh:
            fh.write(json.dumps(record) + '\n')
        return ok({'saved': True})
    except Exception as exc:
        return err(str(exc))


@app.route('/api/comparison/list', methods=['GET'])
def list_comparisons():
    try:
        if not COMPARISONS_FILE.exists():
            return ok({'comparisons': [], 'count': 0})
        records = []
        with open(COMPARISONS_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return ok({'comparisons': records, 'count': len(records)})
    except Exception as exc:
        return err(str(exc))


# ===========================================================================
# ERROR HANDLERS
# ===========================================================================
@app.errorhandler(404)
def not_found(_): return err('Not found', 404)

@app.errorhandler(500)
def internal(_): return err('Internal server error', 500)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == '__main__':
    logger.info("🧠 NEUROSURGERY XAI API — Phase 2 (optimised)")
    app.run(host=APIConfig.HOST, port=APIConfig.PORT,
            debug=APIConfig.DEBUG, use_reloader=False, threaded=True)
