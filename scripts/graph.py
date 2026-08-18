from typing import Optional
from typing_extensions import TypedDict
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from langgraph.graph.message import add_messages
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
# pyrefly: ignore [missing-import]
from red_flags import is_red_flag
from schemas import ESILevel, TriageAssessment, PatientInput, ResourceEstimate
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from IPython.display import Image, display

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "db" / "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatGroq(model="llama-3.3-70b-versatile")

class AgentState(TypedDict):
    patient: PatientInput   
    full_assessment: Optional[TriageAssessment]
    esi_context: str        
    
def decide_red_flag(state: AgentState) -> str:
    """Determine if it's an immediate emergency (conditional edge)"""
    flagged, _ = is_red_flag(state['patient'].vignette_text)
    if flagged:
        return "red flag"
    else:
        return "continue"

@tool
def immediate_care_node(state: AgentState)-> AgentState:
    """Assign immediate care for emergency patients"""
    _, text = is_red_flag(state['patient'].vignette_text)
    patient_info = state['patient']
    esi_level = TriageAssessment(
            patient_id=patient_info.patient_id,  
            triage_level=ESILevel(1),
            urgency_label='Immediate',
            estimated_resources=ResourceEstimate(),
            clinical_reasoning="Triggered red flag safety rule.",
            red_flags_detected=[text] if isinstance(text, str) else text,
            recommended_action="Immediate Resuscitation / Call 911",
            target_wait_time="0 minutes"
        )
    state['full_assessment'] = esi_level
    return state

@tool
def retriever_node(state: AgentState)->AgentState:
    """Retrieves relevant ESI Handbook content to avoid hallucination in LLM decision making"""
    patient = state["patient"]
    
    # Use chief_complaint or full vignette_text for Chroma search query
    query_text = f"{patient.chief_complaint} " + " ".join(patient.vignette_text) 
    docs = retriever.invoke(query_text)
    retrieved_esi_guidelines = "\n\n".join([doc.page_content for doc in docs])
    state['esi_context'] = retrieved_esi_guidelines

    return state

structured_model = model.with_structured_output(TriageAssessment)

@tool
def final_classifier_agent_node(state: AgentState)-> AgentState:
    """Classify patient to an ESI Level based on content from retriever"""

    system_prompt = f"""
    You are an expert Emergency Department Triage Nurse.
    Use the following ESI Guidelines retrieved from the ESI Handbook to evaluate the patient:
    --- RETRIEVED ESI GUIDELINES ---
    {state['esi_context']}
    --------------------------------
    Patient Complaint & Vitals: {state['patient_symptoms']}
    Your Task:
    1. Estimate the number of resources needed (0, 1, or 2+).
    2. Assign the appropriate ESI Triage Level (2, 3, 4, or 5).
    3. Provide your clinical reasoning.

    Output should look like this:
    ESI Triage Level: 
    Resources: 
    Clinical Reasoning: 
    """

    result: TriageAssessment = structured_model.invoke(system_prompt)
    state["full_assessment"] = result

    return state

def create_workflow():
    """Create full workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("classifier", final_classifier_agent_node)
    workflow.add_node("immediate care", immediate_care_node)

    workflow.add_conditional_edges(
        START,
        decide_red_flag,
        {
            "red flag": "immediate care",
            "continue": "retriever"
        }
    )

    workflow.add_edge("retriever", "classifier")
    workflow.add_edge("classifier", END)
    workflow.add_edge("immediate care", END)

    app = workflow.compile()
    png_data = app.get_graph().draw_mermaid_png()
    with open("graph_workflow.png", "wb") as f:
        f.write(png_data)
    return app


if __name__ == "__main__":
    # test_patient = PatientInput(
    #     'PAT000005',
    #     57,
    #     'M',
    #     'Cardiac Arrest',
    #     "57yo M presenting with Cardiac arrest. Patient appears critically ill. Immediate intervention required. Airway assessed, vitals unstable."
    # )
    # test_state = {
    #     'patient': test_patient,
    # }
    create_workflow()