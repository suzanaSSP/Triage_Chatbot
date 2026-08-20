import sys
import asyncio
from pathlib import Path
import uuid

sys.path.append(str(Path(__file__).resolve().parent))

import chainlit as cl
from schemas import PatientInput, TriageAssessment
from triage_graph_engine import create_workflow
from ui_formatter import format_full_assessment_summary

# Pre-defined Clinical Sample Vignettes for One-Click Quick Testing
SAMPLE_VIGNETTES = {
    "sample_red_flag": {
        "title": "🚨 Sample 1: Red Flag Chest Pain (Level 1)",
        "vignette": "69-year-old female presents with crushing chest pain radiating to left arm, shortness of breath, and diaphoresis. Patient in severe distress. Vital signs concerning.",
        "chief_complaint": "Crushing Chest Pain Radiating to Left Arm"
    },
    "sample_level3": {
        "title": "🟡 Sample 2: Abdominal Pain (Level 3)",
        "vignette": "45-year-old male presenting with right lower quadrant abdominal pain for 12 hours, fever 38.5°C, and nausea. Likely requires labs, CT scan, and IV fluids.",
        "chief_complaint": "Right Lower Quadrant Abdominal Pain"
    },
    "sample_level5": {
        "title": "🔵 Sample 3: Mild Skin Rash (Level 5)",
        "vignette": "22-year-old male with mild localized rash on left forearm for 3 days. No fever, no itching, normal vitals. Requests prescription refill for ointment.",
        "chief_complaint": "Mild Localized Skin Rash"
    }
}


@cl.on_chat_start
async def on_chat_start():
    """Send welcome card with Triage Chatbot header and Quick-Test Actions."""
    # Load workflow using run_in_executor (compatible with Python 3.9 + Chainlit 2.x)
    loop = asyncio.get_event_loop()
    app = await loop.run_in_executor(None, create_workflow)
    cl.user_session.set("triage_app", app)

    welcome_message = """
# 🏥 Triage Chatbot: Please input your symptoms

Welcome to the **Medical Triage Chatbot**. This decision-support tool evaluates emergency patient vignettes using:
1. ⚡ **Pure-Python Red-Flag Interceptor** — Instant Level 1 Emergency detection
2. 📚 **ChromaDB ESI Handbook RAG** — Retrieves ESI clinical guidelines
3. 🧠 **Groq LLaMA-3.3-70B** — Structured resource estimation & ESI level assignment

---
### 📋 How to Begin:
- **Option A**: Click one of the **Quick-Test Sample Buttons** below.
- **Option B**: Type or paste your patient symptoms into the chat box below!
"""

    actions = [
        cl.Action(
            name="run_sample",
            value="sample_red_flag",
            label="🚨 Sample 1: Red Flag Chest Pain (Level 1)",
            description="Test pure-python red flag interceptor"
        ),
        cl.Action(
            name="run_sample",
            value="sample_level3",
            label="🟡 Sample 2: Abdominal Pain (Level 3)",
            description="Test multi-resource ESI Level 3 classification"
        ),
        cl.Action(
            name="run_sample",
            value="sample_level5",
            label="🔵 Sample 3: Mild Skin Rash (Level 5)",
            description="Test zero-resource ESI Level 5 classification"
        )
    ]

    await cl.Message(content=welcome_message, actions=actions).send()


async def process_triage_vignette(vignette_text: str, chief_complaint: str = None, patient_id: str = None):
    """Execute LangGraph engine asynchronously and display formatted results."""
    triage_app = cl.user_session.get("triage_app")
    if not triage_app:
        loop = asyncio.get_event_loop()
        triage_app = await loop.run_in_executor(None, create_workflow)
        cl.user_session.set("triage_app", triage_app)

    if not patient_id:
        patient_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"

    if not chief_complaint:
        chief_complaint = vignette_text[:60] + "..." if len(vignette_text) > 60 else vignette_text

    patient = PatientInput(
        patient_id=patient_id,
        chief_complaint=chief_complaint,
        vignette_text=vignette_text
    )

    loading_msg = await cl.Message(content=f"⏱️ *Processing triage assessment for **{patient_id}**...*").send()

    state_input = {"patient": patient}
    loop = asyncio.get_event_loop()
    result_state = await loop.run_in_executor(None, triage_app.invoke, state_input)

    full_assessment: TriageAssessment = result_state.get("full_assessment")

    if full_assessment:
        formatted_card = format_full_assessment_summary(full_assessment)
        await loading_msg.remove()
        await cl.Message(content=formatted_card).send()
    else:
        await loading_msg.remove()
        await cl.Message(content="⚠️ **Error**: Unable to generate triage assessment.").send()


@cl.action_callback("run_sample")
async def on_action(action: cl.Action):
    """Handle clicks on the pre-loaded sample vignette action buttons."""
    sample_key = action.value
    sample_data = SAMPLE_VIGNETTES.get(sample_key)

    if sample_data:
        await cl.Message(
            content=f"**Selected Sample**: {sample_data['title']}\n\n> *\"{sample_data['vignette']}\"*"
        ).send()
        await process_triage_vignette(
            vignette_text=sample_data["vignette"],
            chief_complaint=sample_data["chief_complaint"]
        )


@cl.on_message
async def on_message(message: cl.Message):
    """Handle custom patient vignette inputs entered by the user."""
    vignette_text = message.content.strip()
    if not vignette_text:
        return
    await process_triage_vignette(vignette_text=vignette_text)
