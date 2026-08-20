import sys
from pathlib import Path
import uuid

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from schemas import PatientInput, TriageAssessment
from triage_graph_engine import create_workflow
from ui_formatter import format_full_assessment_summary

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏥 ED Triage Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* Sidebar — force all text to light */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #94a3b8 !important;
}

/* Header */
.triage-header {
    background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(59,130,246,0.15) 100%);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.triage-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem 0;
}
.triage-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.95rem;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    color: #f1f5f9 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #f1f5f9 !important;
}
/* Inline code badges (backtick text) */
[data-testid="stChatMessage"] code {
    background: rgba(139,92,246,0.35) !important;
    color: #ffffff !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-size: 0.88em !important;
}

/* Sample vignette buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2));
    border: 1px solid rgba(139,92,246,0.5);
    color: #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s ease;
    white-space: normal;
    height: auto;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(139,92,246,0.4), rgba(59,130,246,0.4));
    border-color: rgba(139,92,246,0.9);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(139,92,246,0.3);
}

/* Chat input — dark background so white text is readable */
[data-testid="stChatInput"] textarea {
    background: #1e1b4b !important;
    border: 1px solid rgba(139,92,246,0.6) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    caret-color: #a78bfa !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
}
/* Entire bottom input container */
[data-testid="stBottom"] {
    background: transparent !important;
}

/* Spinner text */
.stSpinner > div {
    color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)

# ── Pre-defined Sample Vignettes ─────────────────────────────────────────────
SAMPLE_VIGNETTES = {
    "sample_red_flag": {
        "title": "🚨 Sample 1: Red Flag Chest Pain (Level 1)",
        "vignette": "69-year-old female presents with crushing chest pain radiating to left arm, shortness of breath, and diaphoresis. Patient in severe distress. Vital signs concerning.",
        "chief_complaint": "Crushing Chest Pain Radiating to Left Arm",
    },
    "sample_level3": {
        "title": "🟡 Sample 2: Abdominal Pain (Level 3)",
        "vignette": "45-year-old male presenting with right lower quadrant abdominal pain for 12 hours, fever 38.5°C, and nausea. Likely requires labs, CT scan, and IV fluids.",
        "chief_complaint": "Right Lower Quadrant Abdominal Pain",
    },
    "sample_level5": {
        "title": "🔵 Sample 3: Mild Skin Rash (Level 5)",
        "vignette": "22-year-old male with mild localized rash on left forearm for 3 days. No fever, no itching, normal vitals. Requests prescription refill for ointment.",
        "chief_complaint": "Mild Localized Skin Rash",
    },
}

# ── Session State Init ───────────────────────────────────────────────────────
if "triage_app" not in st.session_state:
    with st.spinner("⚙️ Initialising triage engine…"):
        st.session_state.triage_app = create_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": str, "content": str}

if "pending_vignette" not in st.session_state:
    st.session_state.pending_vignette = None


# ── Helper: run triage engine ────────────────────────────────────────────────
def run_triage(vignette_text: str, chief_complaint: str = None) -> str:
    """Invoke the LangGraph engine and return a formatted markdown card."""
    patient_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"
    if not chief_complaint:
        chief_complaint = (vignette_text[:60] + "…") if len(vignette_text) > 60 else vignette_text

    patient = PatientInput(
        patient_id=patient_id,
        chief_complaint=chief_complaint,
        vignette_text=vignette_text,
    )
    result_state = st.session_state.triage_app.invoke({"patient": patient})
    full_assessment: TriageAssessment = result_state.get("full_assessment")

    if full_assessment:
        return format_full_assessment_summary(full_assessment)
    return "⚠️ **Error**: Unable to generate triage assessment."


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Triage Chatbot")
    st.markdown("---")
    st.markdown("### ⚡ Quick-Test Samples")
    st.caption("Click a button to run a pre-loaded patient vignette.")

    for key, sample in SAMPLE_VIGNETTES.items():
        if st.button(sample["title"], key=f"btn_{key}"):
            st.session_state.pending_vignette = sample

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
1. ⚡ **Red-Flag Interceptor** — Instant Level 1 detection  
2. 📚 **ChromaDB ESI RAG** — Retrieves clinical guidelines  
3. 🧠 **Groq LLaMA-3.3-70B** — Structured ESI classification
""")
    st.markdown("---")
    st.caption("Based on ESI v4 Clinical Guidelines")


# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="triage-header">
  <h1>🏥 ED Triage Decision Support</h1>
  <p>Enter patient symptoms below or select a quick-test sample from the sidebar.</p>
</div>
""", unsafe_allow_html=True)

# ── Replay chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Process pending sample vignette (from sidebar button) ────────────────────
if st.session_state.pending_vignette:
    sample = st.session_state.pending_vignette
    st.session_state.pending_vignette = None   # consume it

    user_text = f"**Selected Sample**: {sample['title']}\n\n> *\"{sample['vignette']}\"*"
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("⏱️ Running triage assessment…"):
            result_md = run_triage(sample["vignette"], sample["chief_complaint"])
        st.markdown(result_md)

    st.session_state.messages.append({"role": "assistant", "content": result_md})
    st.rerun()


# ── Chat Input (custom vignettes) ─────────────────────────────────────────────
if prompt := st.chat_input("Describe patient symptoms or paste a clinical vignette…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⏱️ Running triage assessment…"):
            result_md = run_triage(prompt)
        st.markdown(result_md)

    st.session_state.messages.append({"role": "assistant", "content": result_md})
