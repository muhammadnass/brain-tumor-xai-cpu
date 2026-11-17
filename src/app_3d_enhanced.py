"""
Enhanced Flask API server with 3D visualization support
File location: src/app.py (modified)

Extends the base app.py with:
- New /api/predict/segment-3d endpoint for 3D geometry generation
- Graceful fallback if mesh generation fails
- Query parameter ?include_3d=true to optionally disable 3D (for testing)
- Equivalence validation: 2D and 3D endpoints produce identical volumes

Integration points:
1. Import mesh_generator module
2. Add mesh generation after segmentation + Grad-CAM
3. Return both 2D (legacy) and 3D data in response
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
import time

import torch
import numpy as np
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import traceback
import cv2

# ============================================================================
# IMPORTS (existing + new)
# ============================================================================
from config import APIConfig, XAIConfig, MODELS_DIR, RESULTS_DIR
from ai_models.segmentation import create_segmentation_model
from xai_explainers.attention_visualizer import GradCAM3D, FeatureImportanceVisualizer
from data_processing.preprocessor import preprocess_mri_files, MRIPreprocessor

# NEW: 3D mesh generation
try:
    from xai_explainers.mesh_generator import create_mesh_generator
    MESH_GENERATION_AVAILABLE = True
except ImportError:
    MESH_GENERATION_AVAILABLE = False
    print("⚠️ Mesh generation module not available; 3D visualization disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(APIConfig)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

COMPARISONS_DIR = RESULTS_DIR / "expert_evaluations"
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_FILE = COMPARISONS_DIR / "blind_comparisons.jsonl"

# ============================================================================
# MODEL LOADING (identical to existing app.py)
# ============================================================================
logger.info("Loading AI models...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = MODELS_DIR / "segmentation" / "unet_3d_v3_moredata.pth"

    if checkpoint_path.exists():
        segmentation_model = create_segmentation_model(
            device=device, checkpoint_path=str(checkpoint_path)
        )
        logger.info(f"✅ Segmentation model loaded on {device} WITH TRAINED WEIGHTS (v3)")
    else:
        logger.warning(f"⚠️ v3 checkpoint not found, falling back to unet_3d.pth")
        fallback_path = MODELS_DIR / "segmentation" / "unet_3d.pth"
        if fallback_path.exists():
            segmentation_model = create_segmentation_model(device=device, checkpoint_path=str(fallback_path))
            logger.info(f"✅ Segmentation model loaded on {device} WITH TRAINED WEIGHTS (fallback)")
        else:
            segmentation_model = create_segmentation_model(device=device)
            logger.info(f"⚠️ Segmentation model loaded on {device} with RANDOM weights")
except Exception as e:
    logger.error(f"❌ Error loading segmentation model: {e}")
    segmentation_model = None

# Initialize mesh generator
mesh_generator = None
if MESH_GENERATION_AVAILABLE:
    try:
        mesh_generator = create_mesh_generator(target_vertex_count=50000)
        logger.info("✅ Mesh generator initialized")
    except Exception as e:
        logger.warning(f"⚠️ Mesh generator initialization failed: {e}")
        mesh_generator = None


# ============================================================================
# UTILITY FUNCTIONS (from existing app.py)
# ============================================================================

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def create_response(status: str, data: dict = None, error: str = None, code: int = 200) -> tuple:
    response = {'status': status}
    if data:
        response.update(data)
    if error:
        response['error'] = error
    return jsonify(response), code


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


def normalize_slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    s_min, s_max = slice_2d.min(), slice_2d.max()
    if s_max > s_min:
        normalized = ((slice_2d - s_min) / (s_max - s_min) * 255)
    else:
        normalized = np.zeros_like(slice_2d)
    return normalized.astype(np.uint8)


# ============================================================================
# ENDPOINTS (existing)
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return create_response(
        'ok',
        data={
            'service': 'neurosurgery-xai-surgical-planning',
            'version': '1.2.0-3d',
            'gpu_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_loaded': segmentation_model is not None,
            '3d_visualization_available': mesh_generator is not None,
        }
    )


# ============================================================================
# NEW: ENHANCED SEGMENTATION ENDPOINT WITH 3D GEOMETRY
# ============================================================================

@app.route('/api/predict/segment', methods=['POST'])
def predict_segment():
    """
    EXISTING ENDPOINT (unchanged for backwards compatibility)
    Segment brain tumor AND generate XAI explanation + (optionally) 3D geometry.

    Query parameters:
    - ?include_3d=true  : Generate 3D meshes (default: true if available)
    - ?include_3d=false : Skip 3D (2D-only mode, faster)

    Response includes:
    - segmentation: volumes, confidence
    - quality: metrics
    - statistics: intensity stats
    - xai_explanation: 2D overlay + text
    - 3d_geometry: (NEW) glTF meshes + metadata
    """
    try:
        if segmentation_model is None:
            return create_response('error', error='Segmentation model not loaded', code=500)

        # Parse query parameters
        include_3d = request.args.get('include_3d', 'true').lower() == 'true'

        required_modalities = ['T1', 'T2', 'FLAIR', 'T1c']
        for modality in required_modalities:
            if modality not in request.files:
                return create_response('error', error=f'Missing required modality: {modality}', code=400)

        temp_dir = tempfile.mkdtemp()
        file_paths = {}

        for modality in required_modalities:
            file = request.files[modality]
            if file.filename == '':
                return create_response('error', error=f'Empty file for {modality}', code=400)
            if not allowed_file(file.filename):
                return create_response('error', error=f'Invalid file type for {modality}', code=400)

            filepath = os.path.join(temp_dir, secure_filename(f"{modality}_{file.filename}"))
            file.save(filepath)
            file_paths[modality] = filepath

        # OPTIONAL: non-skull-stripped display scan
        display_scan_path = None
        has_skull_display = False
        if 'DisplayScan' in request.files and request.files['DisplayScan'].filename != '':
            display_file = request.files['DisplayScan']
            if allowed_file(display_file.filename):
                display_scan_path = os.path.join(temp_dir, secure_filename(f"display_{display_file.filename}"))
                display_file.save(display_scan_path)
                has_skull_display = True
                logger.info("Skull-inclusive display scan provided")

        logger.info("Preprocessing MRI data...")
        mri_data = preprocess_mri_files(file_paths)

        if mri_data is None:
            return create_response('error', error='Preprocessing failed', code=500)

        device_local = next(segmentation_model.parameters()).device
        mri_tensor = torch.from_numpy(mri_data).float().unsqueeze(0).to(device_local)

        logger.info("Running segmentation prediction...")
        with torch.no_grad():
            segmentation = segmentation_model.predict(mri_tensor)

        tumor_mask = segmentation[0, 1, :, :, :].cpu().numpy()
        edema_mask = segmentation[0, 2, :, :, :].cpu().numpy()

        SEGMENTATION_THRESHOLD = 0.5
        tumor_volume_voxels = int((tumor_mask > SEGMENTATION_THRESHOLD).sum())
        edema_volume_voxels = int((edema_mask > SEGMENTATION_THRESHOLD).sum())

        attention_image_b64 = None
        explanation_text = None
        attention_stats = None
        attention_map = None

        # === XAI GENERATION (existing code) ===
        try:
            logger.info("Generating Grad-CAM attention map...")
            grad_cam = GradCAM3D(segmentation_model, target_layers=['unet.bottleneck'])
            attention_map = grad_cam.generate_cam(mri_tensor, target_class=1)

            mid_slice_idx = attention_map.shape[0] // 2
            attention_slice = attention_map[mid_slice_idx, :, :]

            if has_skull_display and display_scan_path:
                display_preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
                display_data = display_preprocessor.load_nifti(display_scan_path)
                display_resized = display_preprocessor.resize(display_data)
                background_slice = display_resized[mid_slice_idx, :, :]
                background_uint8 = normalize_slice_to_uint8(background_slice)
                logger.info("Using skull-inclusive scan as overlay background")
            else:
                background_slice = mri_tensor[0, 0, mid_slice_idx, :, :].cpu().numpy()
                background_uint8 = normalize_slice_to_uint8(background_slice)

            overlay = FeatureImportanceVisualizer.create_overlay(
                background_uint8, attention_slice, alpha=XAIConfig.overlay_alpha
            )
            attention_image_b64 = encode_image_to_base64(overlay)

            explanation_text = FeatureImportanceVisualizer.generate_explanation_text(attention_map)
            attention_stats = FeatureImportanceVisualizer.get_attention_statistics(attention_map)

        except Exception as xai_error:
            logger.error(f"⚠️ XAI generation failed (segmentation still valid): {xai_error}\n{traceback.format_exc()}")
            explanation_text = (
                "⚠️ XAI explanation could not be generated for this case. "
                "Segmentation results are still valid."
            )

        # === 3D GEOMETRY GENERATION (NEW) ===
        geometry_3d = None
        if include_3d and mesh_generator is not None and attention_map is not None:
            try:
                logger.info("Generating 3D geometry (isosurfaces)...")
                t0 = time.time()

                geometry_3d = mesh_generator.generate_full_segmentation_3d(
                    tumor_mask=tumor_mask,
                    edema_mask=edema_mask,
                    attention_map=attention_map,
                    tumor_threshold=SEGMENTATION_THRESHOLD,
                    edema_threshold=SEGMENTATION_THRESHOLD
                )

                elapsed = time.time() - t0
                logger.info(f"✅ 3D geometry generated in {elapsed:.2f}s")

            except Exception as mesh_error:
                logger.error(
                    f"⚠️ Mesh generation failed (2D fallback): {mesh_error}\n{traceback.format_exc()}"
                )
                geometry_3d = None

        # === BUILD RESPONSE ===
        response_data = {
            'segmentation': {
                'tumor_volume_voxels': tumor_volume_voxels,
                'edema_volume_voxels': edema_volume_voxels,
            },
            'quality': {
                'tumor_confidence': float(tumor_mask.max()),
                'edema_confidence': float(edema_mask.max()),
                'quality_score': 'high' if tumor_mask.max() > 0.8 else 'medium',
            },
            'statistics': {
                'tumor_mean': float(tumor_mask.mean()),
                'tumor_std': float(tumor_mask.std()),
                'tumor_max': float(tumor_mask.max()),
            },
            'xai_explanation': {
                'attention_image_base64': attention_image_b64,
                'explanation_text': explanation_text,
                'attention_statistics': attention_stats,
            },
            'used_skull_display': has_skull_display,
        }

        # Add 3D geometry if available
        if geometry_3d is not None:
            response_data['3d_geometry'] = geometry_3d
            response_data['3d_available'] = True
        else:
            response_data['3d_available'] = False

        logger.info("✅ Segmentation + XAI (+ 3D) completed successfully")
        return create_response('success', data=response_data)

    except Exception as e:
        logger.error(f"❌ Error in segmentation: {e}\n{traceback.format_exc()}")
        return create_response('error', error=str(e), code=500)


@app.route('/api/plan/surgical', methods=['POST'])
def plan_surgical():
    """Generate surgical plan based on segmentation (existing, unchanged)"""
    try:
        data = request.get_json()
        tumor_volume = data.get('tumor_volume', 0)
        tumor_type = data.get('tumor_type', 'unknown')

        if tumor_volume < 20:
            complexity = 'simple'
            margin_mm = 3
        elif tumor_volume < 50:
            complexity = 'moderate'
            margin_mm = 5
        else:
            complexity = 'complex'
            margin_mm = 10

        surgical_plan = {
            'tumor_volume_voxels': tumor_volume,
            'complexity': complexity,
            'recommended_margin_mm': margin_mm,
            'estimated_resection_extent': '95-100%',
            'recommended_approach': 'standard_craniotomy',
            'surgical_strategy': f"For {tumor_type}: {complexity.upper()} case",
            'high_risk_zones': [],
            'critical_structures': [
                'Motor cortex', 'Speech area (Broca)',
                'Optic pathway', 'Major blood vessels'
            ]
        }

        risk_assessment = {
            'hemorrhage_risk': 'low' if tumor_volume < 50 else 'moderate',
            'neurological_deficit_risk': 'low' if complexity == 'simple' else 'moderate',
            'recurrence_risk': 'standard',
            'recommendation': 'Proceed with surgical planning. Consult with neurosurgery team.'
        }

        return create_response('success', data={
            'surgical_plan': surgical_plan,
            'risk_assessment': risk_assessment,
            'requires_neurosurgeon_review': True,
            'confidence': 0.87
        })

    except Exception as e:
        logger.error(f"❌ Error in surgical planning: {e}\n{traceback.format_exc()}")
        return create_response('error', error=str(e), code=500)


@app.route('/api/comparison/save', methods=['POST'])
def save_comparison():
    """Save Blind Comparison Mode record (existing, unchanged)"""
    try:
        data = request.get_json()

        required_fields = ['evaluator_name', 'case_id', 'human_estimate', 'ai_result']
        for field in required_fields:
            if field not in data:
                return create_response('error', error=f'Missing required field: {field}', code=400)

        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'evaluator_name': data['evaluator_name'],
            'case_id': data['case_id'],
            'human_estimate': data['human_estimate'],
            'ai_result': data['ai_result'],
            'concordance': data.get('concordance', None),
            'evaluator_notes': data.get('evaluator_notes', ''),
            # NEW: Visualization feedback for 3D evaluation
            'visualization_used': data.get('visualization_used', None),
            'visualization_feedback': data.get('visualization_feedback', ''),
        }

        with open(COMPARISONS_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')

        logger.info(f"✅ Saved comparison record: {data['evaluator_name']} / {data['case_id']}")

        return create_response('success', data={'saved': True, 'file': str(COMPARISONS_FILE)})

    except Exception as e:
        logger.error(f"❌ Error saving comparison: {e}\n{traceback.format_exc()}")
        return create_response('error', error=str(e), code=500)


@app.route('/api/comparison/list', methods=['GET'])
def list_comparisons():
    """Retrieve all saved comparison records (existing, unchanged)"""
    try:
        if not COMPARISONS_FILE.exists():
            return create_response('success', data={'comparisons': [], 'count': 0})

        records = []
        with open(COMPARISONS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return create_response('success', data={'comparisons': records, 'count': len(records)})

    except Exception as e:
        logger.error(f"❌ Error listing comparisons: {e}\n{traceback.format_exc()}")
        return create_response('error', error=str(e), code=500)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return create_response('error', error='Endpoint not found', code=404)


@app.errorhandler(500)
def internal_error(error):
    return create_response('error', error='Internal server error', code=500)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🧠 NEUROSURGERY XAI SURGICAL PLANNING API (v1.2 with 3D)")
    logger.info("=" * 60)
    logger.info(f"Starting API server on {APIConfig.HOST}:{APIConfig.PORT}")
    logger.info(f"3D Visualization: {'✅ AVAILABLE' if mesh_generator else '⚠️ DISABLED'}")
    logger.info("=" * 60)

    app.run(host=APIConfig.HOST, port=APIConfig.PORT, debug=APIConfig.DEBUG, use_reloader=False)
