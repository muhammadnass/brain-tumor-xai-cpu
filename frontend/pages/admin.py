"""
Admin portal — accessible at {streamlit_url}/admin
Full study management for the hospital MD / study coordinator.

Login: sunusinasir@gmail.com / %Password1 (seeded at DB init)

Capabilities
------------
  Dashboard    — n cases, n evaluations, mean SUS/NASA-TLX, concordance chart
  Evaluators   — list accounts, create new evaluator, password reset
  Cases        — full case browser across all evaluators
  Exports      — de-identified CSV download, column names for stats section
  Retrain      — cases flagged for model improvement loop (Phase 2 item 3)
"""

import sys
import os
import requests
import streamlit as st
from pathlib import Path

# Make src importable
_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="XAI Admin — Neurosurgery Study",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.environ.get("FLASK_API_URL", "http://localhost:5000")

# ---------------------------------------------------------------------------
# STYLING — clinical, subdued, Q1 professional
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
body, .stApp { background: #f0f4f8; font-family: 'Inter', system-ui, sans-serif; }

.admin-bar {
    background: linear-gradient(135deg, #0f2340 0%, #1e3a5f 100%);
    color: #fff; padding: 1.2rem 2rem; border-radius: 10px;
    margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;
}
.admin-bar h1 { font-size: 1.35rem; font-weight: 700; margin: 0; }
.admin-bar p  { font-size: 0.82rem; margin: 0; opacity: 0.7; }

.metric-row {
    display: flex; flex-wrap: wrap; gap: 0.9rem; margin-bottom: 1.5rem;
}
.mcard {
    background: white; border: 1px solid #dde3ed; border-radius: 10px;
    padding: 1.1rem 1.4rem; min-width: 130px; flex: 1;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.mcard .v { font-size: 2rem; font-weight: 800; color: #0f2340; line-height: 1; }
.mcard .l { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
            letter-spacing: .05em; margin-top: .3rem; }

.card {
    background: white; border: 1px solid #dde3ed; border-radius: 10px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.card h3 { font-size: .95rem; font-weight: 700; color: #0f2340;
           margin: 0 0 .9rem; padding-bottom: .5rem;
           border-bottom: 1px solid #f1f5f9; }

.badge {
    display:inline-block; padding:2px 8px; border-radius:20px;
    font-size:.7rem; font-weight:600;
}
.bg { background:#dcfce7; color:#166534; }
.by { background:#fef9c3; color:#854d0e; }
.br { background:#fee2e2; color:#991b1b; }
.bb { background:#dbeafe; color:#1e40af; }

.login-box {
    max-width:420px; margin:5rem auto; background:white;
    border:1px solid #dde3ed; border-radius:14px; padding:2.5rem;
    box-shadow:0 6px 24px rgba(0,0,0,.09);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
for k, v in [("adm_token", None), ("adm_id", None),
              ("adm_name", ""), ("adm_pg", "dashboard")]:
    if k not in st.session_state:
        st.session_state[k] = v


def _h():
    return {"X-Eval-Token": st.session_state["adm_token"],
            "Content-Type": "application/json"}


def _get(path):
    return requests.get(f"{API_URL}/eval{path}", headers=_h(), timeout=15)


def _post(path, body):
    return requests.post(f"{API_URL}/eval{path}", json=body,
                         headers=_h(), timeout=15)


# ---------------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------------
if not st.session_state["adm_token"]:
    st.markdown("""
    <div class="login-box">
      <div style="text-align:center;margin-bottom:1.8rem">
        <div style="font-size:3rem">🔑</div>
        <h2 style="font-size:1.25rem;font-weight:700;color:#0f2340;margin:.4rem 0 .2rem">
          Admin Portal</h2>
        <p style="color:#64748b;font-size:.82rem">
          Neurosurgery XAI — Study Management</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("adm_login"):
        uname = st.text_input("Email / Username",
                               placeholder="sunusinasir@gmail.com")
        pw    = st.text_input("Password", type="password")
        btn   = st.form_submit_button("Sign in →", type="primary",
                                       use_container_width=True)

    if btn:
        try:
            r = requests.post(
                f"{API_URL}/eval/auth/login",
                json={"username": uname, "password": pw}, timeout=10,
            )
            if r.status_code == 200:
                data  = r.json()
                ev_id = data["evaluator_id"]
                from evaluation.database import is_admin
                if not is_admin(ev_id):
                    st.error("❌ This account does not have admin privileges.")
                else:
                    st.session_state.update({
                        "adm_token": data["token"],
                        "adm_id":    ev_id,
                        "adm_name":  data["display_name"],
                    })
                    st.rerun()
            else:
                st.error(f"❌ {r.json().get('error','Login failed')}")
        except Exception as exc:
            st.error(f"❌ API unreachable: {exc} — is Flask running on port 5000?")
    st.stop()

# ---------------------------------------------------------------------------
# ADMIN SHELL — header + nav
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="admin-bar">
  <div style="font-size:2.2rem">🔑</div>
  <div>
    <h1>Admin Portal — Neurosurgery XAI Study</h1>
    <p>Signed in as <strong>{st.session_state['adm_name']}</strong>
       &nbsp;·&nbsp; {API_URL}</p>
  </div>
</div>
""", unsafe_allow_html=True)

_pages = {
    "📊 Dashboard":  "dashboard",
    "👥 Evaluators": "evaluators",
    "📁 Cases":      "cases",
    "📤 Exports":    "exports",
    "🚩 Retrain":    "retrain",
}
nav_cols = st.columns(len(_pages) + 1)
for col, (lbl, pg) in zip(nav_cols, _pages.items()):
    with col:
        if st.button(lbl, use_container_width=True,
                     type="primary" if st.session_state["adm_pg"] == pg
                     else "secondary"):
            st.session_state["adm_pg"] = pg
            st.rerun()
with nav_cols[-1]:
    if st.button("🔓 Sign out", use_container_width=True):
        try:
            requests.post(f"{API_URL}/eval/auth/logout",
                          headers=_h(), timeout=5)
        except Exception:
            pass
        for k in ["adm_token", "adm_id", "adm_name"]:
            st.session_state[k] = None if k != "adm_name" else ""
        st.session_state["adm_pg"] = "dashboard"
        st.rerun()

st.markdown("---")
pg = st.session_state["adm_pg"]

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
if pg == "dashboard":
    try:
        r = _get("/admin/summary")
        if r.status_code != 200:
            st.error("Could not load summary from Flask API.")
            st.stop()
        s = r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        st.stop()

    avg_sus  = s.get("avg_sus_score")
    avg_nasa = s.get("avg_nasa_score")
    concord  = s.get("concordance", {})

    st.markdown(f"""
    <div class="metric-row">
      <div class="mcard"><div class="v">{s['n_evaluators']}</div>
        <div class="l">Evaluators</div></div>
      <div class="mcard"><div class="v">{s['n_cases']}</div>
        <div class="l">Cases</div></div>
      <div class="mcard"><div class="v">{s['n_evaluations']}</div>
        <div class="l">Evaluations</div></div>
      <div class="mcard"><div class="v">{f"{avg_sus:.1f}" if avg_sus else "—"}</div>
        <div class="l">Mean SUS</div></div>
      <div class="mcard"><div class="v">{f"{avg_nasa:.1f}" if avg_nasa else "—"}</div>
        <div class="l">Mean NASA-TLX</div></div>
      <div class="mcard"><div class="v">{s['flagged_retrain']}</div>
        <div class="l">Flagged retrain</div></div>
    </div>
    """, unsafe_allow_html=True)

    # SUS interpretation
    if avg_sus:
        grade = ("🟢 Excellent (≥85)" if avg_sus >= 85 else
                 "🟡 Good (70–84)"    if avg_sus >= 70 else
                 "🔴 Marginal (<70)")
        st.info(f"SUS grade: **{grade}** — Brooke (1996) scale")

    # Concordance chart
    if concord:
        st.markdown('<div class="card"><h3>Concordance breakdown</h3>',
                    unsafe_allow_html=True)
        total = max(sum(concord.values()), 1)
        rows = [
            ("✅ Fully agree",     "fully_agree",     "#22c55e"),
            ("⚠️ Partially agree", "partially_agree", "#f59e0b"),
            ("❌ Disagree",        "disagree",        "#ef4444"),
        ]
        for lbl, key, colour in rows:
            n   = concord.get(key, 0)
            pct = n / total * 100
            st.markdown(f"**{lbl}** — {n} cases ({pct:.0f}%)")
            st.progress(pct / 100)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EVALUATORS
# ---------------------------------------------------------------------------
elif pg == "evaluators":
    from evaluation.database import list_evaluators, is_admin

    st.markdown('<div class="card"><h3>👥 Registered Evaluators</h3>',
                unsafe_allow_html=True)
    evs = list_evaluators()
    if evs:
        for ev in evs:
            badge = ('<span class="badge bb">ADMIN</span>'
                     if is_admin(ev["id"])
                     else '<span class="badge bg">EVALUATOR</span>')
            st.markdown(
                f"**{ev['display_name']}** &nbsp;`{ev['username']}`"
                f"&nbsp; {badge} &nbsp; joined {ev['created_at'][:10]}",
                unsafe_allow_html=True,
            )
    else:
        st.info("No evaluators yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>➕ Add Evaluator</h3>',
                unsafe_allow_html=True)
    with st.form("add_eval"):
        disp = st.text_input("Display name", placeholder="Dr. Aminu Bello")
        user = st.text_input("Username / Email")
        pw   = st.text_input("Password (≥ 8 chars)", type="password")
        ok   = st.form_submit_button("Create account", type="primary")
    if ok:
        if len(pw) < 8:
            st.error("Password must be ≥ 8 characters.")
        else:
            try:
                r = _post("/auth/register", {
                    "username": user, "password": pw, "display_name": disp
                })
                if r.status_code == 201:
                    st.success(f"✅ Account created for {disp}")
                    st.rerun()
                else:
                    st.error(r.json().get("error", "Failed"))
            except Exception as exc:
                st.error(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------
elif pg == "cases":
    from evaluation.database import get_connection

    st.markdown('<div class="card"><h3>📁 All Cases</h3>',
                unsafe_allow_html=True)
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT c.id, c.case_ref, c.status, c.created_at,
                       e.display_name AS evaluator,
                       COUNT(ev.id)   AS n_evals
                FROM cases c
                JOIN evaluators e ON e.id = c.evaluator_id
                LEFT JOIN evaluations ev ON ev.case_id = c.id
                GROUP BY c.id ORDER BY c.created_at DESC
            """).fetchall()

        _badge = {
            "pending":   '<span class="badge by">PENDING</span>',
            "evaluated": '<span class="badge bg">EVALUATED</span>',
            "exported":  '<span class="badge bb">EXPORTED</span>',
        }
        if rows:
            for row in rows:
                b = _badge.get(row["status"],
                               f'<span class="badge">{row["status"]}</span>')
                st.markdown(
                    f"**{row['case_ref']}** &nbsp; {b} &nbsp; "
                    f"evaluator: *{row['evaluator']}* &nbsp; "
                    f"{row['n_evals']} eval(s) &nbsp; "
                    f"created {row['created_at'][:10]}",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No cases recorded yet.")
    except Exception as exc:
        st.error(f"Database error: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------
elif pg == "exports":
    st.markdown('<div class="card"><h3>📤 De-identified Study Export</h3>',
                unsafe_allow_html=True)
    st.info(
        "All exports are de-identified. Evaluator names are replaced with "
        "anonymous codes (E01, E02 …). Case references are filename-derived "
        "and never contain patient names, dates of birth, or hospital numbers."
    )
    if st.button("⬇️ Download full dataset (CSV)", type="primary"):
        try:
            r = requests.get(
                f"{API_URL}/eval/admin/export_csv",
                headers=_h(), timeout=30,
            )
            if r.status_code == 200:
                st.download_button(
                    "💾 Save study_export.csv",
                    data=r.content,
                    file_name="study_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.error(f"Export failed ({r.status_code})")
        except Exception as exc:
            st.error(str(exc))

    st.markdown("---")
    st.markdown("**CSV column names** (for statistical analysis section of manuscript):")
    st.code(
        "evaluation_id, evaluator_code, case_ref, case_created_at, case_status,\n"
        "concordance, xai_altered_reading, anatomical_plausibility,\n"
        "tumour_location_correct, margin_accurate,\n"
        "sus_1, sus_2, sus_3, sus_4, sus_5,\n"
        "sus_6, sus_7, sus_8, sus_9, sus_10, sus_score,\n"
        "nasa_mental_demand, nasa_physical_demand, nasa_temporal_demand,\n"
        "nasa_performance, nasa_effort, nasa_frustration, nasa_raw_score,\n"
        "submitted_at, free_text_notes",
        language=None,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# RETRAIN
# ---------------------------------------------------------------------------
elif pg == "retrain":
    from evaluation.database import get_flagged_for_retrain

    st.markdown('<div class="card"><h3>🚩 Model Improvement Loop</h3>',
                unsafe_allow_html=True)
    st.info(
        "Cases where the neurosurgeon disagreed with the AI and flagged "
        "them for the next retraining round."
    )
    try:
        flagged = get_flagged_for_retrain()
        if flagged:
            st.metric("Cases flagged for retraining", len(flagged))
            for f in flagged:
                with st.expander(
                    f"📌 {f['case_ref']} — {f.get('clinician_override','?')} "
                    f"(conf {f.get('confidence_score', 0):.2f})"
                ):
                    st.write(f"**Evaluator:** {f['username']}")
                    st.write(f"**Region notes:** {f.get('region_notes') or '—'}")
                    st.write(f"**Model volume:** "
                             f"{f.get('model_volume_vox',0):,} voxels")
                    st.write(f"**Correction:** {f.get('clinician_override','—')}")
                    st.write(f"**Created:** {f.get('created_at','')[:10]}")

            if st.button("⬇️ Download retrain dataset (JSON)", type="primary"):
                r = requests.get(
                    f"{API_URL}/eval/export/retrain",
                    headers=_h(), timeout=15,
                )
                if r.status_code == 200:
                    st.download_button(
                        "💾 Save retrain_dataset.json",
                        data=r.content,
                        file_name="retrain_dataset.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                else:
                    st.error(f"Export failed ({r.status_code})")
        else:
            st.success("✅ No cases flagged for retraining yet.")
    except Exception as exc:
        st.error(f"Could not load retrain cases: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)
