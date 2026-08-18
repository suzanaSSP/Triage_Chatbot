from typing import Optional
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import ToolNode
from red_flags import is_red_flag
from schemas import ESILevel, TriageAssessment, PatientInput, ResourceEstimate
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "db" / "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings
)
# Convert to retriever (k=3 or k=4 chunks is usually optimal)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatGroq(model="llama-3.3-70b-versatile")

load_dotenv()

class AgentState(TypedDict):
    patient: PatientInput   # Storing your Pydantic model in the state
    esi_level: Optional[TriageAssessment]
    esi_context: str        # Note: String context (no add_messages reducer needed here)
    

    
def decide_red_flag(state: AgentState) -> str:
    """Determine if it's an immediate emergency (conditional edge)"""
    patient_info = state['patient']

    flagged, text = is_red_flag(patient_info.vignette_text)

    if flagged:
        esi_level = TriageAssessment(
            patient_id=patient_info.patient_id,  # dot notation
            triage_level=ESILevel(1),
            urgency_label='Immediate',
            estimated_resources=ResourceEstimate(),
            clinical_reasoning="Triggered red flag safety rule.",
            red_flags_detected=[text] if isinstance(text, str) else text,
            recommended_action="Immediate Resuscitation / Call 911",
            target_wait_time="0 minutes"
        )
        state['esi_level'] = esi_level
        return "red flag"
    else:
        return "continue"

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

@tool
def final_classifier_agent_node(state: AgentState)-> AgentState:
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

    response = model.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.0,   # deterministic for retrieval
        max_completion_tokens=100,
        stream=False
    )
    answer = response.choices[0].message.content.strip()


