# Medical Triage Chatbot — Technical Architecture & Stack Plan

## 1. Executive Summary & Technology Strategy

The Medical Triage Chatbot leverages a streamlined, state-of-the-art machine learning (ML) and Large Language Model (LLM) technical stack optimized for single-developer efficiency, high performance, and strict clinical safety. The architecture combines **LangGraph Agentic State Graphs**, **ChromaDB Retrieval-Augmented Generation (RAG)** grounded in Emergency Severity Index (ESI) protocols, **Deterministic Pure-Python Red-Flag Interceptors**, **Groq Ultra-Fast API Inference**, and **LangSmith & Ragas Observability**.

---

## 2. High-Level Architecture Diagram

```
                 +---------------------------------------+
                 |            Chainlit UI                |
                 |      (Interactive Chat Interface)     |
                 +------------------+--------------------+
                                    |
                                    v
                 +------------------+--------------------+
                 |    LangGraph Agentic Workflow Engine   |
                 +------------------+--------------------+
                                    |
                    [LangGraph Red-Flag Check Node]
                    (Pure Python Deterministic Regex)
                                    |
              +---------------------+---------------------+
              | (Red Flag Detected)                       | (No Red Flags)
              v                                           v
  +-----------------------+                   +-----------------------+
  |  Emergency Output     |                   |  Clinical RAG         |
  |  Node (Level 1 Alert) |                   |  Retriever Node       |
  |  *Bypasses RAG & LLM* |                   |  (ChromaDB Vector DB) |
  +-----------------------+                   +-----------+-----------+
                                                          |
                                                          v
                                              +-----------------------+
                                              |  Groq LLM Engine      |
                                              |  (LLaMA-3.3-70B API)  |
                                              +-----------+-----------+
                                                          |
                                                          v
                                              +-----------------------+
                                              | LangSmith & Ragas     |
                                              | Evaluation Suite      |
                                              +-----------------------+
```

---

## 3. Technology Stack & Tooling Matrix

| Component / Layer | Technology / Library | Selection Rationale & Purpose |
| :--- | :--- | :--- |
| **Agent Orchestration** | **LangGraph / LangChain** | Stateful multi-step LLM loops, branching triage decisions, and red-flag conditional routing. |
| **LLM Inference Engine** | **Groq API** (`LLaMA-3.3-70B-Versatile`) | Ultra-fast cloud inference ($< 500\text{ms}$ latency) eliminating local GPU / server overhead. |
| **Vector DB & RAG** | **ChromaDB** + **PubMedBERT** | Embedded, lightweight, Python-native vector store indexing the ESI Handbook v4 for grounded decisions. |
| **Red-Flag & Safety Interceptor** | **Pure Python Regex / Keyword Node in LangGraph** + **Pydantic v2** | Zero-LLM, instant deterministic check for emergency keywords. Immediately routes to Level 1 Alert without LLM latency or cost. |
| **User Interface (UI)** | **Chainlit** | Modern, native Python conversational interface with streaming responses, UI action buttons, and markdown support. |
| **Observability & Tracing** | **LangSmith** | Complete execution logging, prompt versioning, latency tracking, and LangGraph trajectory debugging. |
| **RAG & Model Evaluation** | **Ragas** | RAG evaluation framework measuring Faithfulness, Answer Relevance, Context Precision, and Triage Accuracy. |

---

## 4. Core Technical Components & Modules

### 4.1 LangGraph Stateful Triage Workflow
The graph orchestrates 4 primary nodes:
1. **Red-Flag Intercept Node (Pure Python / No LLM)**: 
   - Scans incoming patient text against deterministic medical regex/keyword rules (e.g., *"crushing chest pain radiating to arm"*, *"unresponsive"*, *"anaphylaxis"*, *"FAST stroke signs"*).
   - **If emergency match found**: Immediately routes to `EmergencyAlertNode`, setting `TriageLevel = 1` and skipping RAG and LLM execution entirely.
2. **Clinical RAG Retriever Node**:
   - Executes hybrid vector search against **ChromaDB** using ESI guidelines to retrieve exact diagnostic resource definitions using rank_bm25.
3. **LLM Triage & Resource Estimator Node (Groq)**:
   - Calls Groq API with retrieved ESI context to evaluate vital signs, count required resources (0, 1, or $\ge 2$), and determine Levels 2, 3, 4, or 5.
4. **Output Formatter Node**:
   - Uses **Pydantic v2** to ensure structured JSON output matching the clinical `TriageAssessment` model.

### 4.2 Data Models & Schema (Pydantic v2)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class VitalSigns(BaseModel):
    heart_rate: Optional[int] = Field(None, description="Beats per minute (bpm)")
    blood_pressure_systolic: Optional[int] = Field(None, description="mmHg")
    blood_pressure_diastolic: Optional[int] = Field(None, description="mmHg")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    oxygen_saturation: Optional[float] = Field(None, description="SpO2 percentage")
    temperature_celsius: Optional[float] = Field(None, description="Body temp in C")

class ClinicalResourceEstimate(BaseModel):
    labs: bool = False
    ecg: bool = False
    imaging_xray_ct: bool = False
    iv_fluids_meds: bool = False
    specialty_consult: bool = False
    estimated_total_count: int = 0

class TriageAssessment(BaseModel):
    patient_id: str
    triage_level: int = Field(..., ge=1, le=5, description="ESI Triage Level 1 to 5")
    urgency_label: str  # Immediate, Emergent, Urgent, Less Urgent, Non-Urgent
    primary_symptoms: List[str]
    vitals: VitalSigns
    estimated_resources: ClinicalResourceEstimate
    clinical_reasoning: str
    red_flags_detected: List[str]
    recommended_action: str
    target_wait_time: str
```

### 4.3 RAG Knowledge Base Architecture (ChromaDB)
- **Knowledge Corpus**: ESI Implementation Handbook v4, CDC Triage Protocols. Ingest data using **LangChain**.
- **Vector DB**: Local persistent **ChromaDB** store (`./chroma_db`).
- **Embedding Function**: HuggingFace `PubMedBERT` embeddings.

---

## 5. Evaluation & MLOps Framework (LangSmith + Ragas)

- **Execution Tracing (LangSmith)**:
  - Every conversation step in LangGraph is automatically logged to **LangSmith**.
  - Tracks step-by-step state transitions, Groq token consumption, and node latency.
- **RAG Evaluation Suite (Ragas)**:
  - Benchmarked against 50+ clinical vignettes (no PII handling implemented; this system is designed for synthetic/session data only, not real PHI).
  - **Ragas Metrics**:
    - `faithfulness`: Ensures reasoning strictly adheres to retrieved ESI guidelines.
    - `answer_relevancy`: Verifies patient questions are answered accurately.
    - `context_precision`: Evaluates ChromaDB retrieval accuracy.
    - `under_triage_score`: Custom metric enforcing zero misclassification of Level 1/2 emergencies.
