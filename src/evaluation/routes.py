"""
evaluation/routes.py
====================
Flask Blueprint: /eval/*

All clinical evaluation endpoints. Mounted in src/app.py as:
    from evaluation.routes import eval_bp
    app.register_blueprint(eval_bp, url_prefix='/eval')

Auth model
----------
Each request must include a valid session token in the
X-Eval-Token header (issued by POST /eval/auth/login).
Tokens are simple UUID strings stored in a server-side dict
(in-memory; lost on restart — evaluators re-login per session,
which is acceptable for a longitudinal study with infrequent
sessions).  A proper JWT/cookie auth layer can replace this
before any multi-user deployment.

Endpoints
---------
POST /eval/auth/login           → {token, evaluator_id, display_name}
POST /eval/auth/logout          → 204
GET  /eval/cases                → [case list for current evaluator]
POST /eval/cases                → create/resume case, returns case dict
GET  /eval/cases/<id>           → single case
POST /eval/cases/<id>/evaluate  → save evaluation form (SUS + NASA-TLX + concordance)
POST /eval/cases/<id>/disagree  → log model vs clinician disagreement
GET  /eval/cases/<id>/exports   → list exports for this case
POST /eval/export/retrain       → export flagged cases as structured JSON dataset
GET  /eval/admin/summary        → aggregate summary (hospital MD view)
GET  /eval/admin/export_csv     → full de-identified dataset as CSV
"""

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, request, jsonify, Response

from evaluation.database import (
    verify_evaluator, create_evaluator, list_evaluators,
    get_or_create_case, list_cases, get_evaluation, save_evaluation,
    log_disagreement, get_flagged_for_retrain,
    record_export, list_exports,
    get_study_summary, export_all_deidentified, export_evaluator_session,
)
from evaluation.case_ref import suggest_case_ref

try:
    from evaluation.pdf_export import generate_evaluation_pdf, save_segmentation_nifti
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

eval_bp = Blueprint('eval', __name__)

# ---------------------------------------------------------------------------
# In-memory session store  {token: {evaluator_id, username, display_name}}
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_auth(f):
    """Decorator: require valid X-Eval-Token header."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('X-Eval-Token', '')
        session = _sessions.get(token)
        if not session:
            return jsonify({'error': 'Unauthorised — login required'}), 401
        request.eval_session = session
        return f(*args, **kwargs)
    return wrapper


def _ok(data=None, code=200):
    r = {'status': 'success'}
    if data:
        r.update(data)
    return jsonify(r), code


def _err(msg, code=400):
    return jsonify({'status': 'error', 'error': msg}), code


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------

@eval_bp.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return _err('username and password required')

    evaluator = verify_evaluator(username, password)
    if not evaluator:
        return _err('Invalid credentials', 401)

    token = str(uuid.uuid4())
    _sessions[token] = {
        'evaluator_id':  evaluator['id'],
        'username':      evaluator['username'],
        'display_name':  evaluator['display_name'],
    }
    return _ok({
        'token':        token,
        'evaluator_id': evaluator['id'],
        'display_name': evaluator['display_name'],
    })


@eval_bp.route('/auth/logout', methods=['POST'])
@_token_auth
def logout():
    token = request.headers.get('X-Eval-Token', '')
    _sessions.pop(token, None)
    return '', 204


@eval_bp.route('/auth/register', methods=['POST'])
def register():
    """
    Create a new evaluator account.
    In production this should be gated behind admin credentials.
    For now it is open so the study coordinator can add accounts easily;
    restrict by checking request.remote_addr == '127.0.0.1' if needed.
    """
    data = request.get_json() or {}
    username     = data.get('username', '').strip()
    password     = data.get('password', '')
    display_name = data.get('display_name', '').strip()
    if not username or len(password) < 8:
        return _err('username required; password must be ≥ 8 characters')
    try:
        eid = create_evaluator(username, password, display_name)
        return _ok({'evaluator_id': eid}, 201)
    except ValueError as exc:
        return _err(str(exc), 409)


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

@eval_bp.route('/cases', methods=['GET'])
@_token_auth
def get_cases():
    eid = request.eval_session['evaluator_id']
    return _ok({'cases': list_cases(eid)})


@eval_bp.route('/cases', methods=['POST'])
@_token_auth
def create_case():
    """
    Create or resume a case.
    Body: {t1_filename, modality_hash, case_ref (optional override)}

    case_ref is derived from t1_filename (option C) unless the caller
    provides an explicit override (evaluator edited the pre-filled field).
    """
    data = request.get_json() or {}
    eid           = request.eval_session['evaluator_id']
    t1_filename   = data.get('t1_filename', '')
    modality_hash = data.get('modality_hash', '')
    case_ref      = data.get('case_ref', '').strip()

    if not modality_hash:
        return _err('modality_hash required')

    # Derive case_ref from filename if not explicitly supplied
    if not case_ref:
        existing = [c['case_ref'] for c in list_cases(eid)]
        case_ref = suggest_case_ref(t1_filename, existing)

    case = get_or_create_case(eid, case_ref, modality_hash)
    # Include any existing evaluation if this is a resumed case
    existing_eval = get_evaluation(case['id'], eid)
    return _ok({'case': case, 'existing_evaluation': existing_eval}, 201)


@eval_bp.route('/cases/<int:case_id>', methods=['GET'])
@_token_auth
def get_case(case_id):
    eid = request.eval_session['evaluator_id']
    cases = list_cases(eid)
    case = next((c for c in cases if c['id'] == case_id), None)
    if not case:
        return _err('Case not found or not owned by this evaluator', 404)
    existing_eval = get_evaluation(case_id, eid)
    return _ok({'case': case, 'existing_evaluation': existing_eval})


# ---------------------------------------------------------------------------
# EVALUATION FORM
# ---------------------------------------------------------------------------

@eval_bp.route('/cases/<int:case_id>/evaluate', methods=['POST'])
@_token_auth
def submit_evaluation(case_id):
    """
    Save the full evaluation form: concordance + SUS (10 items) + NASA-TLX (6 dims).
    All SUS items must be integers 1–5.
    All NASA-TLX items must be integers 0–100.
    """
    eid  = request.eval_session['evaluator_id']
    data = request.get_json() or {}

    # Validate SUS items
    for i in range(1, 11):
        key = f'sus_{i}'
        val = data.get(key)
        if val is None or not isinstance(val, int) or not (1 <= val <= 5):
            return _err(f'{key} must be an integer 1–5')

    # Validate NASA-TLX items
    for dim in ['nasa_mental_demand', 'nasa_physical_demand',
                'nasa_temporal_demand', 'nasa_performance',
                'nasa_effort', 'nasa_frustration']:
        val = data.get(dim)
        if val is None or not isinstance(val, int) or not (0 <= val <= 100):
            return _err(f'{dim} must be an integer 0–100')

    row_id = save_evaluation(case_id, eid, data)
    return _ok({'evaluation_id': row_id})


# ---------------------------------------------------------------------------
# DISAGREEMENT LOGGING  (Phase 2 item 3)
# ---------------------------------------------------------------------------

@eval_bp.route('/cases/<int:case_id>/disagree', methods=['POST'])
@_token_auth
def log_case_disagreement(case_id):
    """
    Record a model–clinician disagreement for this case.
    Body: {region_notes, model_volume_vox, confidence_score,
           clinician_override, flag_for_retrain}
    """
    eid  = request.eval_session['evaluator_id']
    data = request.get_json() or {}
    data.setdefault('flag_for_retrain', 0)
    row_id = log_disagreement(case_id, eid, data)
    return _ok({'disagreement_id': row_id}, 201)


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

@eval_bp.route('/cases/<int:case_id>/exports', methods=['GET'])
@_token_auth
def case_exports(case_id):
    return _ok({'exports': list_exports(case_id)})


@eval_bp.route('/cases/<int:case_id>/export_pdf', methods=['POST'])
@_token_auth
def export_case_pdf(case_id):
    """
    Generate a de-identified PDF report + NIfTI segmentation for a case.

    Body: {
      segmentation:    {tumor_volume_voxels, tumor_confidence, quality_score},
      risk_regions:    {…} from risk_calculator (optional),
      xai_image_b64:   base64 VG overlay image (optional),
      tumor_mask_b64:  base64 numpy array for NIfTI export (optional),
      t1_path:         server-side path to reference T1 (optional)
    }
    Returns: {pdf_url, nifti_url, export_id}
    """
    if not PDF_AVAILABLE:
        return _err('ReportLab not installed — run: pip install reportlab', 503)

    eid  = request.eval_session['evaluator_id']
    data = request.get_json() or {}

    # Fetch evaluation record from DB
    ev = get_evaluation(case_id, eid)
    if ev is None:
        return _err('No evaluation found for this case — complete the form first', 404)

    # Fetch case record
    from evaluation.database import get_connection
    with get_connection() as conn:
        case = conn.execute(
            'SELECT * FROM cases WHERE id = ? AND evaluator_id = ?',
            (case_id, eid)
        ).fetchone()
    if not case:
        return _err('Case not found', 404)

    # Evaluator code (E01, E02 …)
    evaluator_code = f'E{eid:02d}'
    case_ref       = case['case_ref']

    # Output paths
    from config import RESULTS_DIR
    exports_dir = RESULTS_DIR / 'expert_evaluations' / 'exports'
    exports_dir.mkdir(parents=True, exist_ok=True)

    timestamp  = _utcnow()[:10]   # YYYY-MM-DD
    pdf_path   = str(exports_dir / f'{case_ref}_{evaluator_code}_{timestamp}.pdf')
    nifti_path = str(exports_dir / f'{case_ref}_{evaluator_code}_{timestamp}_mask.nii.gz')

    try:
        # PDF
        generate_evaluation_pdf(
            case_ref       = case_ref,
            evaluator_code = evaluator_code,
            evaluation     = dict(ev),
            segmentation   = data.get('segmentation', {}),
            risk_regions   = data.get('risk_regions'),
            xai_image_b64  = data.get('xai_image_b64'),
            output_path    = pdf_path,
        )
    except Exception as exc:
        return _err(f'PDF generation failed: {exc}', 500)

    # NIfTI segmentation mask (optional — only if mask data provided)
    nifti_saved = False
    mask_b64    = data.get('tumor_mask_b64')
    t1_path     = data.get('t1_path')
    if mask_b64 and t1_path:
        try:
            import numpy as np, base64
            mask_bytes = base64.b64decode(mask_b64)
            mask       = np.frombuffer(mask_bytes, dtype=np.float32).reshape(128, 128, 128)
            save_segmentation_nifti(mask, t1_path, nifti_path)
            nifti_saved = True
        except Exception as exc:
            nifti_path = None
            import logging
            logging.getLogger(__name__).warning(f'NIfTI export failed: {exc}')

    export_id = record_export(
        case_id,
        pdf_path   = pdf_path,
        nifti_path = nifti_path if nifti_saved else None,
    )

    return _ok({
        'export_id':   export_id,
        'pdf_path':    pdf_path,
        'nifti_path':  nifti_path if nifti_saved else None,
        'nifti_saved': nifti_saved,
    }, 201)


@eval_bp.route('/cases/export_session_csv', methods=['GET'])
@_token_auth
def export_session_csv():
    """
    Per-evaluator session CSV.  Returns only rows that belong to the
    authenticated evaluator — safe to download on the clinical device,
    suitable for a clinical utility paper's supplementary data.
    Includes surgical outcome comparison fields added in Phase 2.
    """
    evaluator_id = request.eval_session['evaluator_id']
    rows = export_evaluator_session(evaluator_id)
    if not rows:
        return _err('No completed evaluations yet for this evaluator', 404)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    evaluator_code = f"E{evaluator_id:02d}"
    return Response(
        buf.getvalue(),
        content_type='text/csv',
        headers={
            'Content-Disposition':
                f'attachment; filename=session_{evaluator_code}_{_utcnow()[:10]}.csv'
        },
    )


@eval_bp.route('/export/retrain', methods=['GET', 'POST'])
@_token_auth
def export_retrain_dataset():
    """
    Return all disagreements flagged for retraining as a structured JSON
    dataset (Phase 2 item 3 — model improvement loop).
    """
    flagged = get_flagged_for_retrain()
    payload = {
        'exported_at': _utcnow(),
        'n_cases':     len(flagged),
        'cases':       flagged,
    }
    return Response(
        json.dumps(payload, indent=2),
        content_type='application/json',
        headers={'Content-Disposition': 'attachment; filename=retrain_dataset.json'},
    )


# ---------------------------------------------------------------------------
# ADMIN  (hospital MD view — de-identified)
# ---------------------------------------------------------------------------

@eval_bp.route('/admin/summary', methods=['GET'])
def admin_summary():
    """
    Aggregate study summary — no per-patient data, no evaluator names.
    No auth required here: the hospital MD accesses this view via the
    admin portal which has its own login layer (see admin/routes.py).
    """
    return _ok(get_study_summary())


@eval_bp.route('/admin/export_csv', methods=['GET'])
def admin_export_csv():
    """
    Full de-identified evaluation dataset as CSV.
    Column names are the exact DB column names — reported in manuscript
    Methods section (item j of the performance spec).
    """
    rows = export_all_deidentified()
    if not rows:
        return _err('No evaluation data yet', 404)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    return Response(
        buf.getvalue(),
        content_type='text/csv',
        headers={
            'Content-Disposition':
                f'attachment; filename=study_export_{_utcnow()[:10]}.csv'
        },
    )
