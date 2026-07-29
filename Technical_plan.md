# Medical Triage Chatbot — Technical Architecture & Stack Plan

## 1. Executive Summary & Technology Strategy

The Medical Triage Chatbot leverages the latest industry-standard, high-demand machine learning (ML) and Large Language Model (LLM) tools. The technical architecture combines **Agentic State Graphs**, **Retrieval-Augmented Generation (RAG)** grounded in Emergency Severity Index (ESI) protocols, **Deterministic Safety Guardrails**, and robust **MLOps Observability**.

---

## 2. High-Level Architecture Diagram

```
                 +---------------------------------------+
                 |       Patient Interactive UI          |
                 | (React + Tailwind / Streamlit / Chat) |
                 +------------------+--------------------+
                                    |
                                    v
                 +------------------+--------------------+
                 |    FastAPI Async Gateway Server       |
                 +------------------+--------------------+
                                    |
                                    v
                 +------------------+--------------------+
                 |   NeMo Guardrails / Pydantic Safety   |  <--- Red-Flag Deterministic Rules
                 | (PII Masking & Llama-Guard 3 Safety)  |       (Level 1 Emergency Intercept)
                 +------------------+--------------------+
                                    |
                                    v
                 +------------------+--------------------+
                 |   LangGraph Agentic Workflow Engine    |
                 +--------+---------------------+--------+
                          |                     |
                          v                     v
            +-------------+-------+     +-------+-------------+
            | RAG Clinical Vector |     | LLM Inference Engine|
            | Qdrant / ChromaDB   |     | vLLM / Groq / API   |
            | PubMedBERT Embeds   |     | (LLaMA-3.3 / MedLLM)|
            +---------------------+     +---------------------+
                                    |
                                    v
                 +------------------+--------------------+
                 |   MLOps & Evaluation Pipeline         |
                 | (LangSmith / MLflow / DeepEval)       |
                 +---------------------------------------+
```

---

## 3. Technology Stack & Trending Tooling Matrix

| Component / Layer | Technology / Library | Industry Relevance & Justification |
| :--- | :--- | :--- |
| **Agent Orchestration** | **LangGraph / LangChain** | Industry standard for stateful multi-step LLM loops, branching triage decisions, and agent memory management. |
| **Prompt Engineering & Tuning** | **DSPy** | Algorithmic prompt optimization & structured output enforcement; eliminates manual prompt tweaking. |
| **LLM Inference Engines** | **Groq API** (Cloud) & **vLLM / Ollama** (Local) | Groq for sub-500ms inference; vLLM for high-throughput GPU serving of local open-weights medical LLMs (e.g., LLaMA-3.3-70B, BioMistral). |
| **Vector DB & RAG** | **Qdrant** (or **ChromaDB**) + **PubMedBERT** / **bge-large-en-v1.5** | High-performance vector search indexing the ESI Handbook v4 and emergency triage clinical guidelines for grounded, non-hallucinated decisions. |
| **Safety & Guardrails** | **NeMo Guardrails**, **Llama Guard 3**, **Pydantic v2** | Enterprise-grade safety framework for input/output verification, jailbreak prevention, PII scrubbing, and mandatory Level 1 red-flag overrides. |
| **Backend Framework** | **FastAPI** + **Uvicorn** + **AsyncIO** | Modern high-speed Python REST API framework with native Pydantic data validation and WebSocket streaming support. |
| **MLOps & Observability** | **LangSmith** & **MLflow** | Real-time prompt tracing, token cost monitoring, latency tracking, and LLM call trajectory logging. |
| **Evaluation Framework** | **DeepEval** / **Ragas** | Continuous LLM benchmarking targeting clinical accuracy, hallucination detection, and **zero under-triage verification**. |
| **Frontend UI** | **Streamlit** / **Chainlit** (Rapid) or **Vite + React + TailwindCSS** | Interactive chat interface featuring color-coded triage badges, real-time vital sign intake forms, and nurse summary view. |

---

## 4. Core Technical Components & Modules

### 4.1 LangGraph Stateful Triage Workflow
The conversational graph consists of 5 main state nodes:
1. **Intake & Extraction Node**: Parses free-form patient descriptions into structured JSON (`Symptom`, `Severity`, `Onset`, `Vitals`).
2. **Red-Flag Intercept Node**: Evaluates symptoms against deterministic clinical rules (e.g., severe dyspnea, chest pressure, stroke symptoms). If triggered $\rightarrow$ Immediate Level 1 output.
3. **Clinical RAG Retriever Node**: Queries vector store for relevant ESI guidelines matching extracted symptoms.
4. **Resource Estimator Node**: Predicts required hospital resources (0, 1, or $\ge 2$) for Level 3 vs Level 4 vs Level 5 differentiation.
5. **Final Triage & Explanation Generator Node**: Synthesizes final triage level (1-5), safety advice, and physician SBAR summary.

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

### 4.3 RAG Knowledge Base Architecture
- **Source Documents**: Emergency Severity Index (ESI) Implementation Handbook v4, CDC Triage Protocols, PubMed Emergency Medicine guidelines.
- **Embedding Pipeline**: Chunking with semantic splitters $\rightarrow$ embedding with `PubMedBERT` or `bge-large-en-v1.5` $\rightarrow$ stored in `Qdrant`.
- **Hybrid Retrieval**: Dense Vector Similarity + BM25 Sparse Keyword search for exact medical terminologies (e.g., "diaphoresis", "hemoptysis").

### 4.4 Guardrails & Safety Pipeline
1. **Input Shield**: Llama Guard 3 screens for malformed queries, prompt injections, or off-topic requests.
2. **PII Anonymizer**: Replaces names, SSNs, phone numbers with tokens before API dispatch.
3. **Clinical Red-Flag Guard**: Hard-coded Python checks for high-acuity keywords that force Level 1/Level 2 routing regardless of LLM response.
4. **Output Verification**: Pydantic validation ensures outputs strictly adhere to the `TriageAssessment` JSON structure.

---

## 5. Evaluation & MLOps Framework

- **Dataset for Testing**: Synthetic clinical vignettes derived from public triage benchmarks (MIMIC-IV emergency department dataset subsets formatted for ESI classification).
- **Automated Metric Suite (via DeepEval / Ragas)**:
  - **Triage Accuracy**: Correct 1-5 level match against ground truth physician annotations.
  - **Under-Triage Penalty Score**: Severe penalty for misclassifying Level 1/2 as Level 3/4/5.
  - **Over-Triage Rate**: Tracks resource over-estimation without endangering patient safety.
  - **Hallucination Metric**: Verifies that medical advice remains within context retrieved by RAG.
