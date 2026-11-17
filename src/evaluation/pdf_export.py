"""
evaluation/pdf_export.py
========================
Generates a de-identified clinical evaluation PDF report using ReportLab.

Report contents
---------------
  Page 1 — Study header, case reference, evaluation date, evaluator code
  Page 2 — AI segmentation summary: tumour volume, confidence, quality
  Page 2 — XAI overlay image (Vanilla Gradient, mid-axial slice, WebP/PNG)
  Page 3 — Clinical assessment: concordance, XAI influence, plausibility
  Page 3 — Risk structures (from atlas proximity calculator)
  Page 4 — SUS (10 items + computed score + grade)
  Page 5 — NASA-TLX (6 dimensions + raw mean + profile bar chart)
  Page 5 — Free-text notes
  Footer — "Research prototype — not for clinical use" on every page

Privacy
-------
  - Case reference is filename-derived (never a patient name)
  - Evaluator shown as an anonymous code (E01, E02 …)
  - No date of birth, hospital number, or NHS number appears anywhere
  - The PDF is written to the local exports/ directory only

Dependencies
------------
  pip install reportlab Pillow
"""

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (matches admin portal navy/white scheme)
# ---------------------------------------------------------------------------
NAVY   = (0.059, 0.137, 0.251)      # #0f2340
STEEL  = (0.118, 0.227, 0.373)      # #1e3a5f
LIGHT  = (0.941, 0.965, 0.988)      # #f0f7fc
RED    = (0.937, 0.267, 0.267)      # #ef4444
AMBER  = (0.961, 0.620, 0.043)      # #f59e0b
GREEN  = (0.133, 0.773, 0.369)      # #22c55e
GREY   = (0.392, 0.455, 0.545)      # #647488
WHITE  = (1.0,   1.0,   1.0)
BLACK  = (0.0,   0.0,   0.0)


def _try_import_rl():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm, mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage, PageBreak,
        )
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.graphics.charts.barcharts import HorizontalBarChart
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# SUS METADATA
# ---------------------------------------------------------------------------
SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine most people would learn to use this system quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going.",
]

NASA_LABELS = [
    ("Mental Demand",   "nasa_mental_demand"),
    ("Physical Demand", "nasa_physical_demand"),
    ("Temporal Demand", "nasa_temporal_demand"),
    ("Performance",     "nasa_performance"),
    ("Effort",          "nasa_effort"),
    ("Frustration",     "nasa_frustration"),
]


def _sus_grade(score: float) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "OK / Marginal"
    return "Poor"


def _rl_colour(rgb_tuple):
    from reportlab.lib.colors import Color
    return Color(*rgb_tuple)


# ---------------------------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------------------------

def generate_evaluation_pdf(
    case_ref: str,
    evaluator_code: str,
    evaluation: dict,
    segmentation: dict,
    risk_regions: dict | None,
    xai_image_b64: str | None,
    output_path: str,
) -> str:
    """
    Generate a de-identified evaluation PDF.

    Args:
        case_ref:        Study case reference (filename-derived, no PHI).
        evaluator_code:  Anonymous code, e.g. "E01".
        evaluation:      Dict of evaluation record from SQLite.
        segmentation:    Dict with tumor_volume_voxels, tumor_confidence, etc.
        risk_regions:    Output from risk_calculator.compute_risk_regions().
        xai_image_b64:   Base64-encoded WebP/PNG of the VG overlay image.
        output_path:     Full path where the PDF should be written.

    Returns:
        output_path (str)
    """
    if not _try_import_rl():
        raise ImportError(
            "ReportLab is not installed. Run: pip install reportlab"
        )

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak,
    )
    from reportlab.lib.colors import Color, white, black, HexColor

    W, H = A4
    MARGIN = 2.0 * cm

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=2.5 * cm,
        title=f"Neurosurgery XAI Evaluation — {case_ref}",
        author="Neurosurgery XAI Study",
    )

    styles = getSampleStyleSheet()

    def _style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    S_TITLE  = _style("Title",  fontSize=18, textColor=_rl_colour(NAVY),
                      fontName="Helvetica-Bold", spaceAfter=4)
    S_H1     = _style("H1",    fontSize=13, textColor=_rl_colour(NAVY),
                      fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    S_H2     = _style("H2",    fontSize=10, textColor=_rl_colour(STEEL),
                      fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
    S_BODY   = _style("Body",  fontSize=9,  textColor=_rl_colour(BLACK),
                      spaceAfter=3)
    S_SMALL  = _style("Small", fontSize=7.5, textColor=_rl_colour(GREY))
    S_WARN   = _style("Warn",  fontSize=8,  textColor=_rl_colour(RED),
                      fontName="Helvetica-Bold")
    S_CENTER = _style("Ctr",   fontSize=9,  alignment=TA_CENTER)

    NAVY_C  = _rl_colour(NAVY)
    STEEL_C = _rl_colour(STEEL)
    LIGHT_C = _rl_colour(LIGHT)
    RED_C   = _rl_colour(RED)
    AMBER_C = _rl_colour(AMBER)
    GREEN_C = _rl_colour(GREEN)

    def _hr():
        return HRFlowable(width="100%", thickness=0.5, color=LIGHT_C,
                          spaceAfter=6, spaceBefore=6)

    now_str  = datetime.now(timezone.utc).strftime("%d %B %Y %H:%M UTC")
    sus_score = evaluation.get("sus_score")
    nasa_score = evaluation.get("nasa_raw_score")

    story = []

    # ================================================================
    # HEADER BLOCK
    # ================================================================
    header_data = [[
        Paragraph("🧠 Neurosurgery XAI", S_TITLE),
        Paragraph(
            f"<font color='#647488' size=8>Research prototype · Not for clinical use</font>",
            _style("HR", alignment=TA_CENTER, fontSize=8,
                   textColor=_rl_colour(GREY))
        ),
    ]]
    header_table = Table(header_data, colWidths=[12 * cm, 5 * cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY_C),
        ("TEXTCOLOR",  (0, 0), (-1, -1), white),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # Case / evaluator / date
    meta_data = [
        ["Case reference",  case_ref],
        ["Evaluator code",  evaluator_code],
        ["Evaluation date", now_str],
        ["Report generated", now_str],
    ]
    meta_table = Table(meta_data, colWidths=[5 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), STEEL_C),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_C]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(_hr())

    # ================================================================
    # SEGMENTATION SUMMARY
    # ================================================================
    story.append(Paragraph("1. AI Segmentation Summary", S_H1))

    tv  = segmentation.get("tumor_volume_voxels", 0)
    conf = segmentation.get("tumor_confidence", 0)
    qual = segmentation.get("quality_score", "—").upper()

    seg_data = [
        ["Tumour volume",     f"{tv:,} voxels"],
        ["Model confidence",  f"{conf:.1%}"],
        ["Quality score",     qual],
        ["Segmentation model", "3D U-Net, fold 1 (val Dice = 0.7667)"],
        ["XAI methods",       "Vanilla Gradient (primary) + Integrated Gradients (background)"],
    ]
    seg_table = Table(seg_data, colWidths=[6 * cm, 11 * cm])
    seg_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), STEEL_C),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_C]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 10))

    # XAI overlay image
    if xai_image_b64:
        try:
            img_bytes = base64.b64decode(xai_image_b64)
            img_buf   = io.BytesIO(img_bytes)
            # Convert WebP → PNG in memory for ReportLab compatibility
            from PIL import Image as PILImage
            pil_img   = PILImage.open(img_buf).convert("RGB")
            png_buf   = io.BytesIO()
            pil_img.save(png_buf, format="PNG")
            png_buf.seek(0)
            story.append(Paragraph("Vanilla Gradient XAI Overlay (mid-axial slice)", S_H2))
            rl_img = RLImage(png_buf, width=10 * cm, height=10 * cm)
            story.append(rl_img)
            story.append(Paragraph(
                "Red/yellow = high model attention · Blue = low attention · "
                "This map shows model sensitivity, not diagnostic probability.",
                S_SMALL,
            ))
        except Exception as exc:
            logger.warning(f"XAI image embed failed: {exc}")
            story.append(Paragraph(f"[XAI image unavailable: {exc}]", S_SMALL))

    story.append(_hr())

    # ================================================================
    # CLINICAL ASSESSMENT
    # ================================================================
    story.append(Paragraph("2. Clinical Assessment", S_H1))

    concord_map = {
        "fully_agree":     "✅ Fully agree",
        "partially_agree": "⚠️ Partially agree",
        "disagree":        "❌ Disagree",
    }
    xai_map = {
        "reinforced": "🔵 Reinforced my reading",
        "altered":    "🟡 Altered my reading",
        "no_effect":  "⚪ No effect on my reading",
    }

    clin_data = [
        ["Concordance with AI",      concord_map.get(
            evaluation.get("concordance", ""), evaluation.get("concordance", "—"))],
        ["XAI influence on reading", xai_map.get(
            evaluation.get("xai_altered_reading", ""),
            evaluation.get("xai_altered_reading", "—"))],
        ["Anatomical plausibility",  f"{evaluation.get('anatomical_plausibility','—')} / 5"],
        ["Tumour location correct",  "Yes" if evaluation.get("tumour_location_correct") else "No"],
        ["Margin accurate",          "Yes" if evaluation.get("margin_accurate") else "No"],
    ]
    clin_table = Table(clin_data, colWidths=[6 * cm, 11 * cm])
    clin_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), STEEL_C),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_C]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(clin_table)

    notes = evaluation.get("free_text_notes", "")
    if notes:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Clinical notes:", S_H2))
        story.append(Paragraph(notes, S_BODY))

    # Risk structures
    if risk_regions:
        high = risk_regions.get("high_risk_structures", [])
        mod  = risk_regions.get("low_risk_structures", [])
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Risk Regions ({risk_regions.get('method','—')})", S_H2))
        if high:
            for s in high:
                story.append(Paragraph(
                    f"🔴 <b>{s['name']}</b> — {s['min_distance_mm']} mm from boundary",
                    S_WARN,
                ))
        else:
            story.append(Paragraph(
                "No critical structures within 8 mm of tumour boundary.", S_BODY
            ))
        if mod:
            story.append(Paragraph(
                "Moderate proximity (8–20 mm): " +
                ", ".join(f"{s['name']} ({s['min_distance_mm']} mm)" for s in mod),
                S_SMALL,
            ))
        if risk_regions.get("note"):
            story.append(Paragraph(f"Note: {risk_regions['note']}", S_SMALL))

    story.append(PageBreak())

    # ================================================================
    # SUS
    # ================================================================
    story.append(Paragraph("3. System Usability Scale (SUS)", S_H1))
    story.append(Paragraph(
        "Brooke, J. (1996). SUS — a quick and dirty usability scale. "
        "In P. Jordan et al. (Eds.), Usability evaluation in industry (pp. 189–194).",
        S_SMALL,
    ))
    story.append(Spacer(1, 6))

    sus_rows = [["#", "Question", "Response (1–5)"]]
    for i, q in enumerate(SUS_QUESTIONS, start=1):
        val = evaluation.get(f"sus_{i}", "—")
        sus_rows.append([str(i), q, str(val)])

    sus_table = Table(sus_rows, colWidths=[0.8 * cm, 13.2 * cm, 3 * cm])
    sus_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_C),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_C]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",  (2, 0), (2, -1), "CENTER"),
        ("GRID",   (0, 0), (-1, -1), 0.3, _rl_colour(LIGHT)),
    ]))
    story.append(sus_table)
    story.append(Spacer(1, 8))

    # SUS score
    if sus_score is not None:
        grade = _sus_grade(sus_score)
        grade_colour = (GREEN_C if sus_score >= 85 else
                        AMBER_C if sus_score >= 70 else RED_C)
        score_data = [
            [Paragraph(f"<b>SUS Score</b>", S_BODY),
             Paragraph(f"<b>{sus_score:.1f} / 100</b>", S_BODY),
             Paragraph(f"<b>{grade}</b>", S_BODY)],
        ]
        score_table = Table(score_data, colWidths=[5 * cm, 5 * cm, 7 * cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_C),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",  (1, 0), (1, 0), NAVY_C),
            ("TEXTCOLOR",  (2, 0), (2, 0), grade_colour),
        ]))
        story.append(score_table)

    story.append(PageBreak())

    # ================================================================
    # NASA-TLX
    # ================================================================
    story.append(Paragraph("4. NASA Task Load Index (NASA-TLX)", S_H1))
    story.append(Paragraph(
        "Hart & Staveland (1988). Development of NASA-TLX: Results of empirical "
        "and theoretical research. Advances in Psychology, 52, 139–183.",
        S_SMALL,
    ))
    story.append(Spacer(1, 6))

    nasa_rows = [["Dimension", "Score (0–100)", "Low end", "High end"]]
    for label, key in NASA_LABELS:
        val = evaluation.get(key, "—")
        lo, hi = {
            "nasa_mental_demand":   ("Low",     "High"),
            "nasa_physical_demand": ("Low",     "High"),
            "nasa_temporal_demand": ("Low",     "High"),
            "nasa_performance":     ("Failure", "Perfect"),
            "nasa_effort":          ("Low",     "High"),
            "nasa_frustration":     ("Low",     "High"),
        }.get(key, ("Low", "High"))
        nasa_rows.append([label, str(val), lo, hi])

    nasa_table = Table(nasa_rows, colWidths=[5 * cm, 4 * cm, 3 * cm, 5 * cm])
    nasa_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_C),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_C]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",  (1, 0), (1, -1), "CENTER"),
        ("GRID",   (0, 0), (-1, -1), 0.3, _rl_colour(LIGHT)),
    ]))
    story.append(nasa_table)
    story.append(Spacer(1, 8))

    if nasa_score is not None:
        workload = ("Low" if nasa_score < 40 else
                    "Moderate" if nasa_score < 65 else "High")
        story.append(Paragraph(
            f"<b>NASA-TLX Raw Mean: {nasa_score:.1f} / 100</b> "
            f"— Workload: {workload}",
            S_H2,
        ))

    story.append(Spacer(1, 12))
    story.append(_hr())

    # ================================================================
    # DISCLAIMER
    # ================================================================
    disclaimer = (
        "DISCLAIMER: This report is generated by a research prototype and is "
        "intended for research purposes only. The AI segmentation and XAI "
        "outputs have not been validated for clinical use and must not be used "
        "as a substitute for qualified medical judgement. The system has been "
        "evaluated with Dice = 0.746 (95% CI 0.725–0.766) on 484 BraTS / "
        "Decathlon cases. Risk region distances are estimates based on atlas "
        "proximity and do not replace intraoperative neuronavigation."
    )
    story.append(Paragraph(disclaimer, S_SMALL))

    # ================================================================
    # PAGE FOOTER (added via doc template canvas)
    # ================================================================
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColorRGB(*GREY)
        canvas.drawString(
            MARGIN,
            1.2 * cm,
            f"Neurosurgery XAI — Research prototype · Not for clinical use · "
            f"Case {case_ref} · Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    logger.info(f"PDF written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# NIfTI DE-IDENTIFICATION
# ---------------------------------------------------------------------------

def deidentify_nifti_header(nifti_path: str, output_path: str) -> str:
    """
    Strip PHI from a NIfTI header and save to output_path.

    Fields cleared:
      - descrip, aux_file, intent_name (may contain free-text PHI)
      - db_name (legacy NIfTI-1 field)

    The image data and spatial transform are preserved exactly.
    """
    try:
        import nibabel as nib
        img    = nib.load(nifti_path)
        header = img.header.copy()

        # Clear all text fields that might carry PHI
        for field in ["descrip", "aux_file", "intent_name", "db_name"]:
            try:
                header[field] = b""
            except (KeyError, ValueError):
                pass

        clean_img = nib.Nifti1Image(
            np.array(img.dataobj), img.affine, header=header
        )
        nib.save(clean_img, output_path)
        logger.info(f"De-identified NIfTI saved to {output_path}")
        return output_path
    except Exception as exc:
        logger.error(f"NIfTI de-identification failed: {exc}")
        raise


def save_segmentation_nifti(
    tumor_mask: np.ndarray,
    reference_nifti_path: str,
    output_path: str,
) -> str:
    """
    Save the segmentation mask as a de-identified NIfTI file, using the
    spatial affine from the reference T1 scan.

    Args:
        tumor_mask:           [128,128,128] binary or probability array.
        reference_nifti_path: Path to the patient's T1 scan (for affine).
        output_path:          Where to write the .nii.gz file.

    Returns:
        output_path
    """
    try:
        import nibabel as nib
        ref    = nib.load(reference_nifti_path)
        binary = (tumor_mask > 0.5).astype(np.uint8)

        # Re-use the header from the reference but clear PHI fields
        header = ref.header.copy()
        for field in ["descrip", "aux_file", "intent_name", "db_name"]:
            try:
                header[field] = b""
            except (KeyError, ValueError):
                pass
        header.set_data_dtype(np.uint8)

        mask_img = nib.Nifti1Image(binary, ref.affine, header=header)
        nib.save(mask_img, output_path)
        logger.info(f"Segmentation NIfTI saved to {output_path}")
        return output_path
    except Exception as exc:
        logger.error(f"NIfTI save failed: {exc}")
        raise
