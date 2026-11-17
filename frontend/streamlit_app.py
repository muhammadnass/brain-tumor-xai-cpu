"""
Streamlit front-end — Phase 2 (redesigned)
Prestigious neurosurgery / AI research aesthetic.
Real-time slice navigator via custom HTML/JS component.
"""

import time
import base64
import hashlib
import logging
import sys
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
from streamlit_3d_viewer import render_3d_viewer
from evaluation_form import render_evaluation_tab

st.set_page_config(
    page_title="Neurosurgery XAI — Clinical Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# GLOBAL CSS — Prestigious Neurosurgery / AI Research theme
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg-primary:    #080D18;
    --bg-secondary:  #0F1729;
    --bg-card:       #131E35;
    --bg-elevated:   #1A2845;
    --accent-blue:   #00BFFF;
    --accent-purple: #9B72FF;
    --accent-teal:   #00E5CC;
    --accent-warn:   #FF6B6B;
    --accent-gold:   #FFB547;
    --text-primary:  #E8EDF5;
    --text-secondary:#8FA3C0;
    --text-muted:    #4A5C78;
    --border:        rgba(0,191,255,0.15);
    --border-strong: rgba(0,191,255,0.35);
    --glow-blue:     0 0 20px rgba(0,191,255,0.2);
    --glow-purple:   0 0 20px rgba(155,114,255,0.2);
    --radius:        10px;
}

/* ── Body / App shell ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {
    background: var(--bg-primary) !important;
    padding-top: 1.5rem !important;
}
section[data-testid="stMain"] { background: var(--bg-primary) !important; }

/* ── Header / Title ── */
h1, .stTitle {
    font-family: 'Space Grotesk', sans-serif !important;
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
h4, h5, h6, p, label, span, div {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border-strong) !important;
    gap: 4px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--accent-blue) !important;
    border-color: var(--border) !important;
    background: var(--bg-card) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    background: var(--bg-card) !important;
    border-color: var(--border-strong) !important;
    border-bottom-color: var(--bg-card) !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="stTabContent"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1.5rem !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
    box-shadow: var(--glow-blue) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 0 28px rgba(0,191,255,0.3) !important;
    border-color: var(--border-strong) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: var(--glow-blue) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px rgba(0,191,255,0.4) !important;
}
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-elevated) !important;
    box-shadow: var(--glow-blue) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    color: var(--accent-teal) !important;
    font-weight: 500 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
}
.stSuccess { border-left-color: var(--accent-teal) !important; background: rgba(0,229,204,0.07) !important; }
.stWarning { border-left-color: var(--accent-gold) !important; background: rgba(255,181,71,0.07) !important; }
.stError   { border-left-color: var(--accent-warn) !important; background: rgba(255,107,107,0.07) !important; }
.stInfo    { border-left-color: var(--accent-blue) !important; background: rgba(0,191,255,0.07) !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #fff !important;
    box-shadow: 0 0 8px var(--accent-blue) !important;
}

/* ── Radio / Checkbox ── */
[data-testid="stRadio"] label { color: var(--text-primary) !important; }
[data-testid="stCheckbox"] label { color: var(--text-primary) !important; }

/* ── Text inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue) !important;
    outline: none !important;
}

/* ── Tables ── */
[data-testid="stTable"] table,
.stDataFrame table {
    background: var(--bg-elevated) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTable"] th { background: var(--bg-card) !important; color: var(--accent-blue) !important; }
[data-testid="stTable"] td { color: var(--text-primary) !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
}

/* ── Horizontal rule ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Sidebar components ── */
[data-testid="stSidebar"] .stTextInput input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary) !important; }

/* ── Footer ── */
.xai-footer {
    text-align: center;
    padding: 16px;
    color: var(--text-muted);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.xai-footer strong { color: var(--accent-blue); }

/* ── Section header decoration ── */
.section-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,191,255,0.15), rgba(155,114,255,0.15));
    border: 1px solid var(--border-strong);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 10px;
}

/* ═══════════════════════════════════════════════════════
   RESPONSIVE — tablet (≤ 1024 px) & mobile (≤ 640 px)
   ═══════════════════════════════════════════════════════ */

/* Tablet: tighten padding, shrink tab labels */
@media (max-width: 1024px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stTabContent"] { padding: 1rem !important; }
    button[data-baseweb="tab"] {
        font-size: 0.76rem !important;
        padding: 7px 10px !important;
    }
    [data-testid="stMetric"] { padding: 10px 12px !important; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
}

/* Mobile: single-column, larger tap targets */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding: 0.5rem 0.75rem !important;
        max-width: 100vw !important;
    }
    [data-testid="stTabContent"] {
        padding: 0.75rem !important;
        border-radius: 0 0 8px 8px !important;
    }
    /* Tabs: horizontal scroll, no wrap */
    [data-testid="stTabs"] > div:first-child {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 2px !important;
    }
    [data-testid="stTabs"] > div:first-child::-webkit-scrollbar { display: none !important; }
    button[data-baseweb="tab"] {
        font-size: 0.72rem !important;
        padding: 8px 10px !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }
    /* Columns: stack vertically */
    [data-testid="stColumns"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Metrics: full-width stacked */
    [data-testid="stMetric"] {
        width: 100% !important;
        padding: 12px 14px !important;
        margin-bottom: 0.4rem !important;
    }
    /* Typography */
    h1 { font-size: 1.25rem !important; letter-spacing: -0.3px !important; }
    h2 { font-size: 1.05rem !important; }
    h3 { font-size: 0.95rem !important; }
    .stCaption, [data-testid="stCaptionContainer"] { font-size: 0.78rem !important; }
    .section-tag { font-size: 0.66rem !important; padding: 2px 10px !important; }
    /* Buttons: full width, bigger tap area */
    .stButton > button {
        width: 100% !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.9rem !important;
    }
    /* Alerts */
    [data-testid="stAlert"] { font-size: 0.82rem !important; }
    /* Sidebar full-width when open */
    [data-testid="stSidebar"] {
        width: 100vw !important;
        min-width: 100vw !important;
    }
    .xai-footer { font-size: 0.7rem !important; padding: 12px 8px !important; }
}

/* Slice navigator iframe always full width */
iframe { width: 100% !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# SESSION KEEPALIVE
# ===========================================================================
components.html("""
<script>
(function() {
  function ping() { fetch(window.location.origin, { method: 'HEAD' }).catch(()=>{}); }
  setInterval(ping, 240000);
})();
</script>
""", height=0)

# ===========================================================================
# ACCESS GATE + PRIVACY NOTICE
# ===========================================================================
import os

def _get_access_code():
    code = os.environ.get("APP_ACCESS_CODE")
    if code:
        return code
    try:
        return st.secrets.get("APP_ACCESS_CODE", None)
    except Exception:
        return None

ACCESS_CODE = _get_access_code()

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False
if "privacy_acknowledged" not in st.session_state:
    st.session_state.privacy_acknowledged = False

if ACCESS_CODE and not st.session_state.access_granted:
    st.markdown('<div class="section-tag">Access Control</div>', unsafe_allow_html=True)
    st.title("🧠 Neurosurgery XAI Surgical Planning Assistant")
    st.markdown("### 🔒 Restricted Access")
    entered = st.text_input("Enter access code", type="password")
    if st.button("Enter", type="primary"):
        if entered == ACCESS_CODE:
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("❌ Incorrect access code")
    st.stop()

if not st.session_state.privacy_acknowledged:
    st.title("🧠 Neurosurgery XAI Surgical Planning Assistant")
    st.error("""
### ⚠️ IMPORTANT — Please read before continuing

**This is a research prototype, not a validated clinical tool.**

Model performance: Dice = 0.746, 95% CI 0.725–0.766, 3-fold CV / 484 cases.

**Do NOT upload real, identifiable patient data.**

AI predictions must support, never replace, clinician interpretation.
""")
    ack = st.checkbox("I understand and will not upload identifiable patient data.")
    if st.button("Continue", disabled=not ack, type="primary"):
        st.session_state.privacy_acknowledged = True
        st.rerun()
    st.stop()

# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.markdown("""
<div style="padding: 4px 0 12px 0;">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;
              background:linear-gradient(135deg,#00BFFF,#9B72FF);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:-0.3px;">
    ⚙️ Configuration
  </div>
</div>
""", unsafe_allow_html=True)

api_url = st.sidebar.text_input(
    "API URL", value="http://localhost:5000/api",
    help="Flask API base URL"
)
show_xai = st.sidebar.checkbox("🔍 Show XAI overlays", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background:rgba(0,191,255,0.06);border:1px solid rgba(0,191,255,0.2);
            border-radius:10px;padding:12px 14px;font-size:0.8rem;line-height:1.7;">
  <div style="color:#00BFFF;font-weight:600;font-size:0.78rem;
              text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
    Model Specs
  </div>
  <div style="color:#8FA3C0;">
    <b style="color:#E8EDF5;">Architecture</b><br>3D U-Net · 22.6M params<br><br>
    <b style="color:#E8EDF5;">Training</b><br>484 BraTS/Decathlon · 3-fold CV<br><br>
    <b style="color:#00E5CC;">Dice Score</b><br>0.746 (95% CI 0.725–0.766)<br><br>
    <span style="color:#FF6B6B;font-size:0.74rem;">⚠️ Research prototype — not clinically validated</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# HELPERS
# ===========================================================================

def _poll_job(job_id: str, api_base: str, timeout_s: int = 300) -> dict | None:
    bar = st.progress(0.0)
    msg = st.empty()
    t0  = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{api_base}/jobs/{job_id}/status", timeout=10)
            if r.status_code != 200:
                msg.error(f"❌ Status check failed ({r.status_code})")
                return None
            s = r.json()
        except requests.RequestException as exc:
            msg.warning(f"⚠️ Poll error: {exc} — retrying…")
            time.sleep(3)
            continue

        bar.progress(float(s.get('progress', 0)))
        msg.markdown(f"⏳ **{s.get('message', 'Processing…')}**")

        if s.get('status') == 'error':
            msg.error(f"❌ Pipeline error: {s.get('error')}")
            bar.empty()
            return None

        if s.get('preview_ready'):
            try:
                rr = requests.get(f"{api_base}/jobs/{job_id}/result", timeout=30)
                if rr.status_code == 200:
                    bar.progress(1.0)
                    msg.success("✅ Segmentation complete")
                    return rr.json()
            except requests.RequestException:
                pass

        time.sleep(2)

    msg.error("❌ Timeout waiting for result")
    bar.empty()
    return None


def _display_xai(xai_dict: dict, label: str):
    img_b64 = xai_dict.get('attention_image_base64')
    if img_b64:
        st.image(base64.b64decode(img_b64),
                 caption=f"{label} overlay — red/yellow = high attention",
                 width=480)
    text = xai_dict.get('explanation_text')
    if text:
        with st.expander("Explanation text"):
            st.text(text)


def _render_slice_navigator(api_url: str, job_id: str):
    """
    Real-time slice navigator implemented as a custom HTML/JS component.
    Fetches slice images directly from Flask without Streamlit page reruns.
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0F1729;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E8EDF5;
    padding: 16px;
    user-select: none;
  }}
  .nav-wrapper {{
    background: #131E35;
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 0 24px rgba(0,191,255,0.08);
  }}
  .nav-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
  }}
  .nav-title {{
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #00BFFF, #9B72FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .slice-badge {{
    margin-left: auto;
    background: rgba(0,191,255,0.12);
    border: 1px solid rgba(0,191,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #00BFFF;
    font-weight: 600;
    min-width: 80px;
    text-align: center;
  }}
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }}
  .slider-wrap {{
    flex: 1;
    min-width: 180px;
  }}
  .slider-label {{
    font-size: 0.73rem;
    color: #8FA3C0;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
  }}
  input[type=range] {{
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00BFFF var(--pct, 50%), #1A2845 var(--pct, 50%));
    outline: none;
    cursor: pointer;
    transition: background 0.05s;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 0 8px #00BFFF, 0 0 2px rgba(0,0,0,0.5);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }}
  input[type=range]::-webkit-slider-thumb:hover {{
    box-shadow: 0 0 14px #00BFFF;
  }}
  input[type=range]::-moz-range-thumb {{
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 0 8px #00BFFF;
    cursor: pointer;
    border: none;
  }}
  .method-wrap {{
    display: flex;
    gap: 6px;
    align-items: center;
  }}
  .method-label {{
    font-size: 0.73rem;
    color: #8FA3C0;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    white-space: nowrap;
  }}
  .method-btn {{
    background: transparent;
    border: 1px solid rgba(0,191,255,0.25);
    border-radius: 6px;
    color: #8FA3C0;
    font-family: inherit;
    font-size: 0.76rem;
    font-weight: 500;
    padding: 5px 12px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  .method-btn:hover {{
    color: #00BFFF;
    border-color: #00BFFF;
    background: rgba(0,191,255,0.06);
  }}
  .method-btn.active {{
    background: linear-gradient(135deg, rgba(0,191,255,0.2), rgba(155,114,255,0.2));
    border-color: rgba(0,191,255,0.5);
    color: #00BFFF;
    box-shadow: 0 0 10px rgba(0,191,255,0.2);
  }}
  .img-container {{
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: #080D18;
    border: 1px solid rgba(0,191,255,0.12);
    height: 460px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .img-container img {{
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 8px;
    transition: opacity 0.12s;
    object-fit: contain;
  }}
  .img-container img.loading {{
    opacity: 0.4;
  }}
  .overlay-status {{
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(8,13,24,0.85);
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #8FA3C0;
    backdrop-filter: blur(4px);
  }}
  .overlay-status.skull {{ color: #00E5CC; border-color: rgba(0,229,204,0.3); }}
  .overlay-status.loading-dot::after {{
    content: '…';
    animation: dots 1s infinite;
  }}
  @keyframes dots {{ 0%{{opacity:1}} 50%{{opacity:0.3}} 100%{{opacity:1}} }}
  .caption-bar {{
    margin-top: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.73rem;
    color: #4A5C78;
  }}
  .caption-hint {{ color: #4A5C78; font-style: italic; }}
  .error-msg {{
    color: #FF6B6B;
    font-size: 0.8rem;
    padding: 12px;
    text-align: center;
  }}
  .placeholder-msg {{
    color: #4A5C78;
    font-size: 0.82rem;
    padding: 24px;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="nav-wrapper">
  <div class="nav-header">
    <span style="font-size:1.1rem;">🔬</span>
    <span class="nav-title">Slice Navigator</span>
    <span class="slice-badge" id="slice-badge">Slice 64 / 127</span>
  </div>
  <div class="controls-row">
    <div class="slider-wrap">
      <div class="slider-label">Axial slice position</div>
      <input type="range" id="slice-slider" min="0" max="127" value="64"
             oninput="onSlide(this.value)">
    </div>
    <div class="method-wrap">
      <span class="method-label">Overlay</span>
      <button class="method-btn active" id="btn-vg" onclick="setMethod('vg')">VG</button>
      <button class="method-btn"        id="btn-ig" onclick="setMethod('ig')">IG</button>
    </div>
  </div>
  <div class="img-container" id="img-container">
    <div class="placeholder-msg" id="placeholder">Loading slice…</div>
    <img id="slice-img" src="" alt="" style="display:none;">
    <div class="overlay-status" id="overlay-status" style="display:none;"></div>
  </div>
  <div class="caption-bar">
    <span id="caption-left" class="caption-hint">Drag slider for real-time navigation</span>
    <span id="caption-right"></span>
  </div>
</div>
<script>
const API_URL  = "{api_url}";
const JOB_ID   = "{job_id}";
let   method   = "vg";
let   lastFetch = 0;
let   pendingSlice = null;
let   fetchTimer   = null;

// update slider gradient fill
function updateFill(val) {{
  const pct = (val / 127 * 100).toFixed(1);
  document.getElementById('slice-slider').style.setProperty('--pct', pct + '%');
  document.getElementById('slice-badge').textContent = 'Slice ' + val + ' / 127';
}}

function setMethod(m) {{
  method = m;
  document.getElementById('btn-vg').classList.toggle('active', m === 'vg');
  document.getElementById('btn-ig').classList.toggle('active', m === 'ig');
  fetchSlice(parseInt(document.getElementById('slice-slider').value));
}}

function onSlide(val) {{
  updateFill(val);
  // Throttle: fire at most every 80 ms while dragging
  pendingSlice = parseInt(val);
  if (!fetchTimer) {{
    fetchTimer = setTimeout(() => {{
      fetchTimer = null;
      if (pendingSlice !== null) {{
        fetchSlice(pendingSlice);
        pendingSlice = null;
      }}
    }}, 80);
  }}
}}

async function fetchSlice(idx) {{
  const img    = document.getElementById('slice-img');
  const status = document.getElementById('overlay-status');
  const ph     = document.getElementById('placeholder');

  img.classList.add('loading');
  status.style.display = 'block';
  status.className     = 'overlay-status loading-dot';
  status.textContent   = '';

  const url = API_URL + '/xai/slice/' + JOB_ID + '/' + idx + '?method=' + method;
  try {{
    const resp = await fetch(url, {{ signal: AbortSignal.timeout(15000) }});
    if (resp.status === 200) {{
      const data = await resp.json();
      const src  = 'data:' + (data.mime || 'image/png') + ';base64,' + data.image_base64;
      img.onload = () => img.classList.remove('loading');
      img.src    = src;
      img.style.display = 'block';
      ph.style.display  = 'none';
      status.className  = 'overlay-status' + (data.skull_bg ? ' skull' : '');
      status.textContent = data.skull_bg
        ? '🦴 skull · ' + method.toUpperCase()
        : method.toUpperCase();
      document.getElementById('caption-right').textContent =
        'n_slices: ' + (data.n_slices || 128);
    }} else if (resp.status === 202) {{
      status.className   = 'overlay-status';
      status.textContent = '⏳ IG computing…';
      img.classList.remove('loading');
    }} else {{
      status.className   = 'overlay-status';
      status.textContent = 'unavailable';
      img.classList.remove('loading');
    }}
  }} catch(e) {{
    status.className   = 'overlay-status';
    status.textContent = e.name === 'TimeoutError' ? 'timeout' : ('err: ' + e.message.slice(0,30));
    img.classList.remove('loading');
    console.error('Slice fetch failed:', e);
  }}
}}

// initial load
updateFill(64);
fetchSlice(64);
</script>
</body>
</html>
"""

    components.html(html, height=740, scrolling=False)


# ===========================================================================
# HEADER + TABS
# ===========================================================================
st.markdown('<div class="section-tag">Research Prototype</div>', unsafe_allow_html=True)
st.title("🧠 Neurosurgery XAI Surgical Planning Assistant")
st.caption("AI-assisted tumour segmentation · 3D U-Net · Explainable overlays · Phase 2")

tab_upload, tab_plan, tab_analysis, tab_3d, tab_eval, tab_instructions = st.tabs([
    "📤 Upload & Segmentation",
    "🎯 Surgical Planning",
    "📊 Detailed Analysis",
    "🧊 3D Model",
    "📋 Clinical Evaluation",
    "ℹ️ Instructions",
])

# ===========================================================================
# TAB 1 — UPLOAD & SEGMENTATION
# ===========================================================================
with tab_upload:
    st.markdown('<div class="section-tag">Step 1</div>', unsafe_allow_html=True)
    st.subheader("Upload MRI Data")
    st.caption("Upload skull-stripped NIfTI scans (T1, T1c, T2, FLAIR). All four modalities required.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t1_file = st.file_uploader("T1", type=['nii', 'gz', 'dcm'])
    with col2:
        t2_file = st.file_uploader("T2", type=['nii', 'gz', 'dcm'])
    with col3:
        flair_file = st.file_uploader("FLAIR", type=['nii', 'gz', 'dcm'])
    with col4:
        t1c_file = st.file_uploader("T1c", type=['nii', 'gz', 'dcm'])

    with st.expander("🦴 Optional: skull-inclusive T1 for 3D anatomical context"):
        skull_file = st.file_uploader(
            "Non-skull-stripped scan (e.g. skull_T1_brainweb.nii.gz)",
            type=['nii', 'gz', 'dcm'], key="skull_upload"
        )

    if st.button("🚀 Run Segmentation", key="seg_btn", type="primary"):
        if not all([t1_file, t2_file, flair_file, t1c_file]):
            st.warning("⚠️ Upload all four required scans.")
        else:
            st.info("⏳ Uploading and starting segmentation… preview in ~30–90 s on CPU.")
            files = {'T1': t1_file, 'T2': t2_file, 'FLAIR': flair_file, 'T1c': t1c_file}
            if skull_file:
                files['DisplayScan'] = skull_file

            h = hashlib.sha256()
            for f in [t1_file, t2_file, flair_file, t1c_file]:
                f.seek(0); h.update(f.read()); f.seek(0)
            st.session_state['_t1_filename']   = t1_file.name
            st.session_state['_modality_hash'] = h.hexdigest()

            try:
                resp = requests.post(f"{api_url}/predict/segment", files=files, timeout=60)
                if resp.status_code != 200:
                    st.error(f"❌ Submit failed ({resp.status_code}): {resp.text[:200]}")
                    st.stop()

                job_data = resp.json()
                job_id   = job_data.get('job_id')
                if not job_id:
                    st.error("❌ No job_id returned from API")
                    st.stop()

                st.session_state['job_id'] = job_id
                result = _poll_job(job_id, api_url)
                if result:
                    st.session_state['segmentation_result'] = result

            except requests.exceptions.Timeout:
                st.error("❌ Upload timed out — check the API server is running.")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

    # Results
    if 'segmentation_result' in st.session_state:
        result  = st.session_state['segmentation_result']
        seg     = result.get('segmentation', {})
        quality = result.get('quality', {})

        st.success("✅ Segmentation complete")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tumour Volume",  f"{seg.get('tumor_volume_voxels', 0):,} voxels")
        c2.metric("Confidence",     f"{quality.get('tumor_confidence', 0):.2%}")
        c3.metric("Quality Score",  quality.get('quality_score', '—').upper())

        if seg.get('tumor_volume_voxels', 0) == 0:
            st.warning("⚠️ 0 voxels predicted — no tumour detected above threshold.")

        if show_xai:
            xai_vg = result.get('xai_vg') or result.get('xai_explanation', {})
            xai_ig = result.get('xai_ig')
            if xai_vg and xai_vg.get('attention_image_base64'):
                st.subheader("🔍 XAI — Vanilla Gradient")
                _display_xai(xai_vg, "Vanilla Gradient")
            if xai_ig and xai_ig.get('attention_image_base64'):
                st.subheader("🔍 XAI — Integrated Gradients")
                _display_xai(xai_ig, "Integrated Gradients")
            elif 'job_id' in st.session_state:
                st.info("⏳ Integrated Gradients computing in background — check back in ~4 min.")

        # ------------------------------------------------------------------ #
        # REAL-TIME SLICE NAVIGATOR (custom HTML/JS component)               #
        # ------------------------------------------------------------------ #
        if show_xai and 'job_id' in st.session_state:
            st.markdown("---")
            _render_slice_navigator(api_url, st.session_state['job_id'])

        st.info("💡 Go to **🧊 3D Model** tab for the interactive view, "
                "then **📋 Clinical Evaluation** to complete the assessment.")

# ===========================================================================
# TAB 2 — SURGICAL PLANNING
# ===========================================================================
with tab_plan:
    st.markdown('<div class="section-tag">AI Planning</div>', unsafe_allow_html=True)
    st.subheader("Surgical Planning")
    if 'segmentation_result' not in st.session_state:
        st.info("📤 Run segmentation first.")
    else:
        result = st.session_state['segmentation_result']
        vol    = result['segmentation']['tumor_volume_voxels']
        conf   = result.get('quality', {}).get('tumor_confidence')
        jid    = st.session_state.get('job_id', '')
        try:
            pr = requests.post(
                f"{api_url}/plan/surgical",
                json={'tumor_volume': vol, 'tumor_type': 'glioblastoma',
                      'segmentation_confidence': conf, 'job_id': jid},
                timeout=30,
            )
            if pr.status_code == 200:
                pr_data = pr.json()
                plan    = pr_data['surgical_plan']
                risk    = pr_data['risk_assessment']
                regions = pr_data.get('risk_regions', {})

                st.caption(
                    "ℹ️ Complexity based on tumour volume. "
                    "Risk regions calculated from actual tumour proximity "
                    f"({regions.get('method','—')})."
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity",        plan['complexity'].upper())
                c2.metric("Safety Margin",     f"{plan['recommended_margin_mm']} mm")
                c3.metric("Target Resection",  plan['estimated_resection_extent'])

                st.write(
                    f"Haemorrhage risk: **{risk['hemorrhage_risk']}** &nbsp;·&nbsp; "
                    f"Neurological deficit: **{risk['neurological_deficit_risk']}** &nbsp;·&nbsp; "
                    f"Recurrence: **{risk['recurrence_risk']}**"
                )
                st.info(risk['recommendation'])

                high = regions.get('high_risk_structures', [])
                mod  = regions.get('low_risk_structures',  [])

                if high:
                    st.subheader("🔴 High-Risk Structures (< 8 mm from tumour boundary)")
                    for s in high:
                        st.error(
                            f"🚨 **{s['name']}** — "
                            f"{s['min_distance_mm']} mm from boundary"
                            + (f" ({s['affected_voxels']:,} voxels at risk)"
                               if 'affected_voxels' in s else "")
                        )
                elif regions.get('method') != 'unavailable':
                    st.success("✅ No critical structures within 8 mm of the tumour boundary.")

                if mod:
                    with st.expander(
                        f"🟡 Moderate proximity ({len(mod)} structure(s) within 8–20 mm)"
                    ):
                        for s in mod:
                            st.warning(f"🟡 **{s['name']}** — {s['min_distance_mm']} mm")

                if regions.get('note'):
                    st.caption(f"ℹ️ {regions['note']}")

                st.caption(
                    "⚠️ This plan requires neurosurgeon review. "
                    "Risk regions are proximity estimates and do not replace "
                    "intraoperative neuronavigation."
                )
        except Exception as exc:
            st.error(f"Could not generate plan: {exc}")

# ===========================================================================
# TAB 3 — DETAILED ANALYSIS
# ===========================================================================
with tab_analysis:
    st.markdown('<div class="section-tag">Metrics</div>', unsafe_allow_html=True)
    st.subheader("Detailed Analysis")
    if 'segmentation_result' not in st.session_state:
        st.info("📤 Run segmentation first.")
    else:
        result  = st.session_state['segmentation_result']
        seg     = result['segmentation']
        quality = result.get('quality', {})
        stats   = result.get('statistics', {})
        c1, c2  = st.columns(2)
        with c1:
            st.subheader("Segmentation Volumes")
            st.metric("Tumour Volume",     f"{seg['tumor_volume_voxels']:,} voxels")
            st.metric("Oedema Volume",     f"{seg.get('edema_volume_voxels',0):,} voxels")
            st.metric("Tumour Confidence", f"{quality.get('tumor_confidence',0):.3f}")
            st.metric("Oedema Confidence", f"{quality.get('edema_confidence',0):.3f}")
        with c2:
            st.subheader("Intensity Statistics")
            st.metric("Tumour Mean", f"{stats.get('tumor_mean',0):.4f}")
            st.metric("Tumour Std",  f"{stats.get('tumor_std',0):.4f}")
            st.metric("Tumour Max",  f"{stats.get('tumor_max',0):.4f}")

        jid = st.session_state.get('job_id')
        if jid and st.button("🔄 Refresh (check for IG / full mesh)"):
            try:
                rr = requests.get(f"{api_url}/jobs/{jid}/result", timeout=15)
                if rr.status_code == 200:
                    st.session_state['segmentation_result'] = rr.json()
                    st.rerun()
            except Exception:
                pass

# ===========================================================================
# TAB 4 — 3D MODEL
# ===========================================================================
with tab_3d:
    st.markdown('<div class="section-tag">3D Visualisation</div>', unsafe_allow_html=True)
    st.subheader("Interactive 3D Surgical Model")
    if 'segmentation_result' not in st.session_state:
        st.info("📤 Run segmentation first.")
    else:
        result = st.session_state['segmentation_result']
        geo    = result.get('3d_full') or result.get('3d_preview') or result.get('3d_geometry')

        if geo and geo.get('tumor_mesh_gltf'):
            lod = geo.get('metadata', {}).get('lod', 'preview')
            st.caption(f"Mesh LOD: **{lod}** · step_size={geo.get('metadata',{}).get('step_size','?')}")
            st.markdown("🖱️ Drag = rotate · 📜 Scroll = zoom · ☑️ Checkboxes = toggle visibility")
            render_3d_viewer(
                geometry_data=geo,
                height=700,
            )
            if geo.get('skull_mesh_gltf'):
                st.caption(
                    "🦴 Skull surface rendered · Upload non-skull-stripped T1 "
                    "for full anatomical context."
                )
            jid = st.session_state.get('job_id')
            if jid and lod == 'preview':
                if st.button("🔄 Upgrade to full-resolution mesh"):
                    try:
                        rr = requests.get(f"{api_url}/jobs/{jid}/result", timeout=30)
                        if rr.status_code == 200:
                            st.session_state['segmentation_result'] = rr.json()
                            st.rerun()
                    except Exception as exc:
                        st.warning(f"Could not fetch full mesh: {exc}")
            info = geo.get('tumor_mesh_info', {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vertices",  f"{info.get('vertex_count',0):,}")
            c2.metric("Faces",     f"{info.get('face_count',0):,}")
            c3.metric("Attn Mean", f"{info.get('attention_mean',0):.3f}")
            c4.metric("Attn Max",  f"{info.get('attention_max',0):.3f}")
        else:
            st.warning(
                "⚠️ 3D mesh unavailable for this case (tumour too small or marching cubes failed). "
                "The 2D overlay in the Upload tab remains valid."
            )

# ===========================================================================
# TAB 5 — CLINICAL EVALUATION
# ===========================================================================
with tab_eval:
    render_evaluation_tab(
        api_url       = api_url,
        result        = st.session_state.get('segmentation_result'),
        t1_filename   = st.session_state.get('_t1_filename', ''),
        modality_hash = st.session_state.get('_modality_hash', ''),
    )

# ===========================================================================
# TAB 6 — INSTRUCTIONS
# ===========================================================================
with tab_instructions:
    st.markdown('<div class="section-tag">Documentation</div>', unsafe_allow_html=True)
    st.subheader("How to Use")
    st.markdown("""
### Workflow

1. **Upload & Segmentation** — upload all four NIfTI modalities, click **Run Segmentation**.
   A live progress bar tracks the pipeline. The Vanilla Gradient overlay and 3D preview
   appear first; Integrated Gradients follows ~4 min later in the background.

2. **Slice Navigator** — real-time axial viewer beneath the XAI overlay. Drag the slider
   and the image updates as you move. If you uploaded a non-skull-stripped T1 the badge
   shows **skull bg**. Switch between VG and IG overlays with the buttons.

3. **3D Model** — interactive tumour + oedema surface with XAI attention colourmap.
   Preview LOD (step_size=4, ~0.6 s) appears first; click **Upgrade** for the full mesh.

4. **Clinical Evaluation** — log in with your study credentials, confirm the case reference
   (auto-derived from the T1 filename), then complete:
   - Clinical concordance + XAI reading assessment
   - System Usability Scale (SUS, 10 items)
   - NASA Task Load Index (NASA-TLX, 6 dimensions)
   Your response is saved to a local SQLite database — no data leaves the laptop.

5. **Surgical Planning** — volume-based complexity and margin estimates (research heuristic only).

6. **Detailed Analysis** — segmentation metrics and intensity statistics.

### Model Information
| Parameter | Value |
|---|---|
| Architecture | 3D U-Net, features=(16,32,64,128,256) |
| Parameters | 22.6M |
| Training | 484 BraTS/Decathlon cases, 3-fold CV |
| Validation Dice | 0.746 (95% CI 0.725–0.766) |
| XAI (live) | Vanilla Gradient (~2.3 s) |
| XAI (background) | Integrated Gradients, 20 steps (~4 min) |
| Hardware | CPU-only — no GPU required |

⚠️ Research use only — not clinically validated.  
⚠️ All predictions must be reviewed by a qualified neurosurgeon.
""")

# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("""
<div class="xai-footer">
  <strong>Neurosurgery XAI</strong> &nbsp;·&nbsp; Phase 2 Research Prototype &nbsp;·&nbsp;
  CPU-optimised &nbsp;·&nbsp; For research use only
</div>
""", unsafe_allow_html=True)
