"""
evaluation_form.py
==================
Renders the full Phase 2 clinical evaluation form in Streamlit.

Call from the main app:
    from evaluation_form import render_evaluation_tab
    render_evaluation_tab(api_url, segmentation_result, job_id)

Form structure (4 steps, single-page with st.tabs):
  Step 1 — Login        (evaluator credentials)
  Step 2 — Case setup   (case_ref derived from T1 filename, editable)
  Step 3 — Clinical     (concordance, XAI reading, plausibility, notes)
  Step 4 — SUS          (10 validated items, 1–5 Likert)
  Step 5 — NASA-TLX     (6 dimensions, 0–100 sliders)
  Submit  — saves to DB via /eval/*, triggers export option

Session-state keys used (all prefixed eval_):
  eval_token, eval_id, eval_display_name
  eval_case  (dict from API)
  eval_step  (1–5)
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# SUS validated question text (Brooke, 1996)
# ---------------------------------------------------------------------------
SUS_QUESTIONS = [
    (1, "I think that I would like to use this system frequently."),
    (2, "I found the system unnecessarily complex."),
    (3, "I thought the system was easy to use."),
    (4, "I think that I would need the support of a technical person to be able to use this system."),
    (5, "I found the various functions in this system were well integrated."),
    (6, "I thought there was too much inconsistency in this system."),
    (7, "I would imagine that most people would learn to use this system very quickly."),
    (8, "I found the system very cumbersome to use."),
    (9, "I felt very confident using the system."),
    (10, "I needed to learn a lot of things before I could get going with this system."),
]

NASA_DIMENSIONS = [
    ("nasa_mental_demand",   "Mental Demand",   "Low", "High",
     "How much mental and perceptual activity was required? "
     "Was the task easy or demanding, simple or complex?"),
    ("nasa_physical_demand", "Physical Demand", "Low", "High",
     "How much physical activity was required? "
     "Was the task easy or demanding, slack or strenuous?"),
    ("nasa_temporal_demand", "Temporal Demand", "Low", "High",
     "How much time pressure did you feel due to the pace at which tasks occurred?"),
    ("nasa_performance",     "Performance",     "Failure", "Perfect",
     "How successful do you think you were in accomplishing the goals set by the evaluator? "
     "How satisfied were you with your performance?"),
    ("nasa_effort",          "Effort",          "Low", "High",
     "How hard did you have to work (mentally and physically) to accomplish your level of performance?"),
    ("nasa_frustration",     "Frustration",     "Low", "High",
     "How insecure, discouraged, irritated, stressed, and annoyed were you during the task?"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(token: str) -> dict:
    return {"X-Eval-Token": token, "Content-Type": "application/json"}


def _post(url: str, token: str, body: dict) -> requests.Response:
    return requests.post(url, json=body, headers=_headers(token), timeout=15)


def _get(url: str, token: str) -> requests.Response:
    return requests.get(url, headers=_headers(token), timeout=15)


def _init_state():
    for k, v in [
        ("eval_token", None), ("eval_id", None), ("eval_display_name", ""),
        ("eval_case", None), ("eval_step", 1),
        ("eval_clinical_done", False), ("eval_sus_done", False),
        ("eval_nasa_done", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


def _logout(api_url: str):
    token = st.session_state.get("eval_token")
    if token:
        try:
            requests.post(f"{api_url}/eval/auth/logout",
                          headers=_headers(token), timeout=5)
        except Exception:
            pass
    for k in ["eval_token", "eval_id", "eval_display_name",
               "eval_case", "eval_step",
               "eval_clinical_done", "eval_sus_done", "eval_nasa_done"]:
        st.session_state[k] = None if "token" in k or "id" in k or "case" in k \
            else ("" if "name" in k else (1 if "step" in k else False))


# ---------------------------------------------------------------------------
# STEP 1 — LOGIN
# ---------------------------------------------------------------------------

def _render_login(api_url: str):
    st.subheader("🔐 Evaluator Login")
    st.caption("Contact the study coordinator to receive your credentials.")

    with st.form("eval_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if not username or not password:
            st.error("Enter both username and password.")
            return
        try:
            r = requests.post(
                f"{api_url}/eval/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                st.session_state["eval_token"]        = data["token"]
                st.session_state["eval_id"]           = data["evaluator_id"]
                st.session_state["eval_display_name"] = data["display_name"]
                st.session_state["eval_step"]         = 2
                st.rerun()
            else:
                st.error(f"❌ {r.json().get('error', 'Login failed')}")
        except Exception as exc:
            st.error(f"❌ Could not reach API: {exc}")

    st.markdown("---")
    with st.expander("Register a new evaluator account"):
        with st.form("eval_register_form"):
            new_user = st.text_input("New username", key="reg_user")
            new_disp = st.text_input("Display name (e.g. Dr. Aliyu)", key="reg_disp")
            new_pass = st.text_input("Password (≥ 8 characters)", type="password",
                                      key="reg_pass")
            reg_sub  = st.form_submit_button("Create account")
        if reg_sub:
            if len(new_pass) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    rr = requests.post(
                        f"{api_url}/eval/auth/register",
                        json={"username": new_user, "password": new_pass,
                              "display_name": new_disp},
                        timeout=10,
                    )
                    if rr.status_code == 201:
                        st.success(f"✅ Account created for '{new_user}'. You can now log in.")
                    else:
                        st.error(rr.json().get("error", "Registration failed"))
                except Exception as exc:
                    st.error(f"❌ {exc}")


# ---------------------------------------------------------------------------
# STEP 2 — CASE SETUP
# ---------------------------------------------------------------------------

def _render_case_setup(api_url: str, result: dict, t1_filename: str,
                        modality_hash: str):
    st.subheader("📁 Case Setup")

    token = st.session_state["eval_token"]

    # Derive suggested case_ref from T1 filename
    from pathlib import Path
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from evaluation.case_ref import derive_case_ref
    suggested = derive_case_ref(t1_filename)

    st.markdown(f"**Evaluator:** {st.session_state['eval_display_name']}")
    st.markdown(f"**T1 filename:** `{t1_filename}`")

    case_ref = st.text_input(
        "Case reference ID",
        value=suggested,
        help="Auto-derived from the filename. Edit if needed. "
             "Never enter the patient's name or hospital ID here.",
    )
    st.caption(
        "ℹ️ This ID is the only identifier stored — no patient name, "
        "date of birth, or hospital number is retained."
    )

    if st.button("Confirm case and proceed", type="primary"):
        if not case_ref.strip():
            st.error("Case reference cannot be empty.")
            return
        try:
            r = _post(
                f"{api_url}/eval/cases",
                token,
                {"t1_filename": t1_filename,
                 "modality_hash": modality_hash,
                 "case_ref": case_ref.strip()},
            )
            if r.status_code in (200, 201):
                data = r.json()
                st.session_state["eval_case"] = data["case"]
                existing = data.get("existing_evaluation")
                if existing:
                    st.info(
                        f"ℹ️ A previous evaluation exists for **{case_ref}** "
                        f"(submitted {existing.get('submitted_at','?')[:10]}). "
                        "Submitting again will overwrite it."
                    )
                st.session_state["eval_step"] = 3
                st.rerun()
            else:
                st.error(r.json().get("error", "Case creation failed"))
        except Exception as exc:
            st.error(f"❌ {exc}")


# ---------------------------------------------------------------------------
# STEP 3 — CLINICAL ASSESSMENT
# ---------------------------------------------------------------------------

def _render_clinical(api_url: str, result: dict):
    case_ref = st.session_state["eval_case"]["case_ref"]
    st.subheader(f"🏥 Clinical Assessment — {case_ref}")
    st.caption("Answer based on what you observe in the AI output shown above.")

    with st.form("eval_clinical_form"):
        concordance = st.radio(
            "**1. Concordance** — Does the AI segmentation agree with your clinical assessment?",
            options=["fully_agree", "partially_agree", "disagree"],
            format_func=lambda x: {
                "fully_agree":     "✅ Fully agree",
                "partially_agree": "⚠️ Partially agree",
                "disagree":        "❌ Disagree",
            }[x],
        )

        xai_reading = st.radio(
            "**2. XAI influence** — Did the explanation overlay alter or reinforce your reading?",
            options=["reinforced", "altered", "no_effect"],
            format_func=lambda x: {
                "reinforced": "🔵 Reinforced my reading",
                "altered":    "🟡 Altered my reading",
                "no_effect":  "⚪ No effect on my reading",
            }[x],
        )

        plausibility = st.slider(
            "**3. Anatomical plausibility** — How plausible is the highlighted tumour region anatomically?",
            min_value=1, max_value=5, value=3,
            help="1 = Not at all plausible  ·  5 = Completely plausible",
        )
        st.caption(
            "1 = Not at all plausible · 2 = Unlikely · "
            "3 = Neutral · 4 = Plausible · 5 = Completely plausible"
        )

        loc_correct = st.checkbox(
            "**4.** The highlighted tumour location is consistent with the source images."
        )
        margin_ok = st.checkbox(
            "**5.** The indicated surgical margin / tumour boundary appears accurate."
        )

        notes = st.text_area(
            "**6. Free-text notes** (optional)",
            placeholder="Describe any discrepancies, concerns, or observations…",
            height=100,
        )

        # Disagreement section — only shown if disagree or partially agree
        st.markdown("---")
        st.markdown("**7. Disagreement details** *(fill if you selected Disagree or Partially agree)*")
        disagree_notes   = st.text_area("Where did the model go wrong?",
                                         placeholder="e.g. Over-segmented in right frontal lobe…",
                                         height=80)
        clinician_override = st.selectbox(
            "Clinician correction",
            ["", "larger", "smaller", "different_location", "absent"],
            format_func=lambda x: {
                "":                   "— select if applicable —",
                "larger":             "Tumour is larger than predicted",
                "smaller":            "Tumour is smaller than predicted",
                "different_location": "Tumour is in a different location",
                "absent":             "No tumour present",
            }.get(x, x),
        )
        flag_retrain = st.checkbox(
            "🚩 Flag this case for model retraining dataset",
            help="Adds this case to the structured retrain export (Phase 2 item 3)."
        )

        # ------------------------------------------------------------------ #
        # SURGICAL OUTCOME COMPARISON                                          #
        # For retrospective evaluation: compare AI outputs to documented      #
        # intraoperative findings. Required for a clinical utility paper.     #
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.markdown("### 🔪 Surgical Outcome Comparison")
        st.caption(
            "Complete this section for retrospective cases where the documented "
            "intraoperative or pathological findings are available. "
            "Leave *'No surgical data available'* selected if this is a prospective case."
        )

        has_surgical_data = st.checkbox(
            "I have documented surgical/pathological findings for this case",
            key="has_surgical_data_cb",
            help="Check this only for operated cases where ground-truth outcomes are known.",
        )

        surgical_extent = "not_available"
        margin_agreement = "not_assessed"
        risk_category_match = "not_assessed"
        intraop_notes = ""

        if has_surgical_data:
            surgical_extent = st.selectbox(
                "**8. Actual surgical extent achieved** (operative report)",
                options=["gross_total", "near_total", "subtotal", "biopsy_only"],
                format_func=lambda x: {
                    "gross_total":  "Gross total resection (>95 % tumour removed)",
                    "near_total":   "Near-total resection (90–95 %)",
                    "subtotal":     "Subtotal resection (<90 %, e.g. eloquent area)",
                    "biopsy_only":  "Stereotactic biopsy only",
                }[x],
                key="surgical_extent_sel",
            )

            margin_agreement = st.radio(
                "**9. AI margin vs. actual surgical boundary** — "
                "How did the AI-predicted resection boundary compare with what was resected?",
                options=["underestimates", "matches", "overestimates"],
                format_func=lambda x: {
                    "underestimates": "⬇️ AI under-estimated tumour extent (residual tumour found beyond AI boundary)",
                    "matches":        "✅ AI margin matched the intraoperative tumour boundary",
                    "overestimates":  "⬆️ AI over-estimated tumour extent (AI boundary exceeded actual tumour)",
                }[x],
                key="margin_agreement_radio",
            )

            risk_category_match = st.radio(
                "**10. Risk category agreement** — "
                "Did the AI complexity rating (Simple / Moderate / Complex) match "
                "the intraoperative complexity experienced?",
                options=["matches", "underrated", "overrated"],
                format_func=lambda x: {
                    "matches":    "✅ AI complexity matched intraoperative complexity",
                    "underrated": "⬇️ AI under-rated complexity (surgery was harder than predicted)",
                    "overrated":  "⬆️ AI over-rated complexity (surgery was easier than predicted)",
                }[x],
                key="risk_category_match_radio",
            )

            intraop_notes = st.text_area(
                "**11. Intraoperative / pathological notes** (optional)",
                placeholder=(
                    "e.g. IDH-wildtype GBM confirmed on frozen section; "
                    "gross total achieved but AI over-delineated oedema into motor cortex…"
                ),
                height=90,
                key="intraop_notes_ta",
            )

        submitted = st.form_submit_button("Save clinical assessment →", type="primary")

    if submitted:
        st.session_state["_eval_clinical"] = {
            "concordance":             concordance,
            "xai_altered_reading":     xai_reading,
            "anatomical_plausibility": plausibility,
            "tumour_location_correct": int(loc_correct),
            "margin_accurate":         int(margin_ok),
            "free_text_notes":         notes,
            # disagreement fields
            "_disagree_notes":         disagree_notes,
            "_clinician_override":     clinician_override,
            "_flag_retrain":           int(flag_retrain),
            # surgical outcome comparison (new — for clinical utility paper)
            "surgical_extent":         surgical_extent,
            "margin_agreement":        margin_agreement,
            "risk_category_match":     risk_category_match,
            "intraop_notes":           intraop_notes,
        }
        st.session_state["eval_clinical_done"] = True
        st.session_state["eval_step"] = 4
        st.rerun()


# ---------------------------------------------------------------------------
# STEP 4 — SUS
# ---------------------------------------------------------------------------

def _render_sus():
    st.subheader("📊 Usability — System Usability Scale (SUS)")
    st.caption(
        "Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree). "
        "Respond based on your experience with this specific system today."
    )

    likert_labels = {
        1: "1 — Strongly disagree",
        2: "2 — Disagree",
        3: "3 — Neutral",
        4: "4 — Agree",
        5: "5 — Strongly agree",
    }

    with st.form("eval_sus_form"):
        responses = {}
        for i, question in SUS_QUESTIONS:
            st.markdown(f"**{i}.** {question}")
            responses[f"sus_{i}"] = st.radio(
                f"sus_{i}",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: likert_labels[x],
                horizontal=True,
                index=2,   # default: Neutral
                label_visibility="collapsed",
            )
            st.markdown("")  # spacing

        submitted = st.form_submit_button("Save SUS responses →", type="primary")

    if submitted:
        st.session_state["_eval_sus"] = responses
        st.session_state["eval_sus_done"] = True
        st.session_state["eval_step"] = 5
        st.rerun()


# ---------------------------------------------------------------------------
# STEP 5 — NASA-TLX
# ---------------------------------------------------------------------------

def _render_nasa():
    st.subheader("🧠 Workload — NASA Task Load Index (NASA-TLX)")
    st.caption(
        "Rate each dimension from 0 to 100 based on your experience "
        "completing this evaluation. Move the slider to your rating."
    )

    with st.form("eval_nasa_form"):
        responses = {}
        for key, label, lo, hi, description in NASA_DIMENSIONS:
            st.markdown(f"**{label}**")
            st.caption(description)
            col_lo, col_slider, col_hi = st.columns([1, 8, 1])
            with col_lo:
                st.caption(lo)
            with col_slider:
                responses[key] = st.slider(
                    label, min_value=0, max_value=100, value=50, step=5,
                    label_visibility="collapsed",
                )
            with col_hi:
                st.caption(hi)
            st.markdown("")

        submitted = st.form_submit_button("Review and submit →", type="primary")

    if submitted:
        st.session_state["_eval_nasa"] = responses
        st.session_state["eval_nasa_done"] = True
        st.session_state["eval_step"] = 6
        st.rerun()


# ---------------------------------------------------------------------------
# STEP 6 — REVIEW & SUBMIT
# ---------------------------------------------------------------------------

def _render_submit(api_url: str, result: dict):
    st.subheader("✅ Review and Submit")

    token   = st.session_state["eval_token"]
    case    = st.session_state["eval_case"]
    case_id = case["id"]
    eid     = st.session_state["eval_id"]

    clinical = st.session_state.get("_eval_clinical", {})
    sus      = st.session_state.get("_eval_sus", {})
    nasa     = st.session_state.get("_eval_nasa", {})

    # SUS score preview (standard formula)
    if sus:
        odd  = sum(sus.get(f"sus_{i}", 3) - 1 for i in (1, 3, 5, 7, 9))
        even = sum(5 - sus.get(f"sus_{i}", 3) for i in (2, 4, 6, 8, 10))
        sus_score = (odd + even) * 2.5
    else:
        sus_score = None

    nasa_vals  = [nasa.get(d[0], 50) for d in NASA_DIMENSIONS] if nasa else []
    nasa_score = sum(nasa_vals) / len(nasa_vals) if nasa_vals else None

    # Summary table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Clinical assessment**")
        st.write(f"Concordance: **{clinical.get('concordance','—')}**")
        st.write(f"XAI influence: **{clinical.get('xai_altered_reading','—')}**")
        st.write(f"Anatomical plausibility: **{clinical.get('anatomical_plausibility','—')}/5**")
        # Surgical outcome fields (only show if data was entered)
        if clinical.get('surgical_extent', 'not_available') != 'not_available':
            st.markdown("**Surgical outcomes**")
            st.write(f"Extent: **{clinical.get('surgical_extent','—')}**")
            st.write(f"Margin: **{clinical.get('margin_agreement','—')}**")
            st.write(f"Risk match: **{clinical.get('risk_category_match','—')}**")
    with col2:
        st.markdown("**Scores**")
        if sus_score is not None:
            grade = ("Excellent" if sus_score >= 85 else
                     "Good" if sus_score >= 70 else
                     "OK" if sus_score >= 50 else "Poor")
            st.metric("SUS Score", f"{sus_score:.1f} / 100", delta=grade)
        if nasa_score is not None:
            st.metric("NASA-TLX (raw mean)", f"{nasa_score:.1f} / 100")

    st.markdown("---")
    if st.button("📤 Submit evaluation", type="primary", use_container_width=True):
        payload = {
            **{k: v for k, v in clinical.items() if not k.startswith("_")},
            **sus,
            **nasa,
        }

        try:
            # Save main evaluation
            r = _post(f"{api_url}/eval/cases/{case_id}/evaluate", token, payload)
            if r.status_code != 200:
                st.error(f"❌ Save failed: {r.json().get('error')}")
                return

            # Log disagreement if applicable
            disagree_notes    = clinical.get("_disagree_notes", "")
            clinician_override = clinical.get("_clinician_override", "")
            flag_retrain      = clinical.get("_flag_retrain", 0)
            concordance       = clinical.get("concordance", "fully_agree")

            if concordance != "fully_agree" or disagree_notes or clinician_override:
                seg = result.get("segmentation", {})
                qual = result.get("quality", {})
                _post(f"{api_url}/eval/cases/{case_id}/disagree", token, {
                    "region_notes":       disagree_notes,
                    "model_volume_vox":   seg.get("tumor_volume_voxels", 0),
                    "confidence_score":   qual.get("tumor_confidence", 0),
                    "clinician_override": clinician_override,
                    "flag_for_retrain":   flag_retrain,
                })

            st.success(
                f"✅ Evaluation saved for case **{case['case_ref']}**  ·  "
                f"SUS: {sus_score:.1f}  ·  NASA-TLX: {nasa_score:.1f}"
            )
            st.balloons()
            st.session_state["_export_case_id"] = case_id
            st.session_state["_export_case_ref"] = case["case_ref"]

        except Exception as exc:
            st.error(f"❌ Submission error: {exc}")
            return

    # ---- PDF / NIfTI export (shown after save) ----
    if st.session_state.get("_export_case_id") == case_id:
        st.markdown("---")
        st.subheader("📄 Export Report & Segmentation")
        col_pdf, col_new = st.columns(2)

        with col_pdf:
            if st.button("⬇️ Generate PDF + NIfTI export",
                         type="primary", use_container_width=True,
                         key="gen_pdf_btn"):
                xai = (result.get("xai_vg") or
                       result.get("xai_explanation") or {})
                export_payload = {
                    "segmentation": result.get("segmentation", {}),
                    "risk_regions": result.get("risk_regions"),
                    "xai_image_b64": xai.get("attention_image_base64"),
                }
                with st.spinner("⏳ Generating PDF report…"):
                    er = _post(
                        f"{api_url}/eval/cases/{case_id}/export_pdf",
                        token, export_payload,
                    )
                if er.status_code == 201:
                    edata    = er.json()
                    pdf_path = edata.get("pdf_path", "")
                    try:
                        with open(pdf_path, "rb") as fh:
                            pdf_bytes = fh.read()
                        st.download_button(
                            "📅 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"{case['case_ref']}_evaluation.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.info(f"✅ PDF saved on server: {pdf_path}")
                    if edata.get("nifti_saved"):
                        st.success("✅ Segmentation NIfTI also exported.")
                elif er.status_code == 503:
                    st.warning(
                        "⚠️ ReportLab not installed.\n"
                        "Run: `pip install reportlab` then restart Flask."
                    )
                else:
                    st.error(f"❌ {er.json().get('error','Export failed')}")

        with col_new:
            if st.button("🔄 Start next case",
                         use_container_width=True, key="next_case_btn"):
                for k in ["eval_case", "eval_step", "eval_clinical_done",
                           "eval_sus_done", "eval_nasa_done",
                           "_eval_clinical", "_eval_sus", "_eval_nasa",
                           "_export_case_id", "_export_case_ref"]:
                    st.session_state.pop(k, None)
                st.session_state["eval_step"] = 2
                st.rerun()

        # Session-level summary export — all cases evaluated this session
        st.markdown("---")
        st.subheader("📊 Session Summary Export")
        st.caption(
            "Download a de-identified CSV of all cases you have evaluated "
            "in this session — suitable for inclusion in a clinical utility paper."
        )
        col_csv, col_json = st.columns(2)
        with col_csv:
            if st.button("⬇️ Export session CSV",
                         use_container_width=True, key="session_csv_btn"):
                try:
                    token = st.session_state.get("eval_token", "")
                    cr = _get(f"{eval_api}/eval/cases/export_session_csv", token)
                    if cr.status_code == 200:
                        evaluator_id = st.session_state.get("eval_id", "evaluator")
                        st.download_button(
                            "📥 Download CSV",
                            data=cr.content,
                            file_name=f"session_summary_{evaluator_id}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="dl_session_csv",
                        )
                    else:
                        st.error(
                            f"Export failed ({cr.status_code}): "
                            f"{cr.json().get('error', 'unknown error')}"
                        )
                except Exception as exc:
                    st.error(f"Session export error: {exc}")
        with col_json:
            if st.button("⬇️ Export retrain dataset (JSON)",
                         use_container_width=True, key="retrain_json_btn"):
                try:
                    token = st.session_state.get("eval_token", "")
                    rr = _post(
                        f"{eval_api}/eval/export/retrain", token, {}
                    )
                    if rr.status_code == 200:
                        st.download_button(
                            "📥 Download JSON",
                            data=rr.content,
                            file_name="retrain_flagged_cases.json",
                            mime="application/json",
                            use_container_width=True,
                            key="dl_retrain_json",
                        )
                    else:
                        st.warning("No cases flagged for retraining yet."
                                   if rr.status_code == 404 else
                                   f"Export failed: {rr.status_code}")
                except Exception as exc:
                    st.error(f"Retrain export error: {exc}")

    if st.button("← Edit responses", key="back_to_edit"):
        st.session_state["eval_step"] = 3
        st.rerun()


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def render_evaluation_tab(api_url: str, result: dict | None = None,
                           t1_filename: str = "", modality_hash: str = ""):
    """
    Render the full evaluation form inside a Streamlit tab.

    Args:
        api_url:        Flask API base URL (e.g. http://localhost:5000/api)
        result:         Segmentation result dict (from job polling)
        t1_filename:    Original T1 upload filename (for case_ref derivation)
        modality_hash:  SHA-256 of the four modality files (for dedup)
    """
    _init_state()

    eval_api = api_url.rstrip("/api").rstrip("/")

    # ---- Header + logout ----
    col_title, col_logout = st.columns([6, 1])
    with col_title:
        st.header("📋 Clinical Evaluation")
    if st.session_state.get("eval_token"):
        with col_logout:
            if st.button("Logout", key="eval_logout"):
                _logout(eval_api)
                st.rerun()

    # ---- Progress indicator ----
    step = st.session_state.get("eval_step", 1)
    steps = ["Login", "Case", "Clinical", "SUS", "NASA-TLX", "Submit"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps), start=1):
        with col:
            if i < step:
                st.success(f"✅ {label}", icon=None)
            elif i == step:
                st.info(f"▶ **{label}**", icon=None)
            else:
                st.caption(f"○ {label}")

    st.markdown("---")

    # ---- Guard: segmentation must be done first (steps 3+) ----
    if step >= 3 and result is None:
        st.warning(
            "⚠️ Run segmentation first (Upload & Segmentation tab) "
            "before completing the evaluation form."
        )
        return

    # ---- Route to current step ----
    if step == 1:
        _render_login(eval_api)

    elif step == 2:
        _render_case_setup(eval_api, result or {}, t1_filename, modality_hash)

    elif step == 3:
        _render_clinical(eval_api, result)

    elif step == 4:
        _render_sus()

    elif step == 5:
        _render_nasa()

    elif step >= 6:
        _render_submit(eval_api, result or {})

    # ---- Prior cases (sidebar shortcut) ----
    if step >= 2 and st.session_state.get("eval_token"):
        with st.expander("📂 My previous cases"):
            try:
                r = _get(f"{eval_api}/eval/cases", st.session_state["eval_token"])
                if r.status_code == 200:
                    cases = r.json().get("cases", [])
                    if cases:
                        for c in cases[:10]:
                            status_icon = {"pending": "⏳", "evaluated": "✅",
                                           "exported": "📤"}.get(c["status"], "?")
                            st.write(
                                f"{status_icon} **{c['case_ref']}** · "
                                f"{c['created_at'][:10]} · "
                                f"{c['eval_count']} evaluation(s)"
                            )
                    else:
                        st.caption("No cases yet.")
            except Exception:
                st.caption("Could not load case history.")
