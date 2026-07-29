# Medical Triage Chatbot — Project Implementation Plan & Timeline

## 1. Project Overview & Execution Strategy

This implementation plan outlines the step-by-step roadmap for engineering the **5-Level Medical Triage Chatbot**. The timeline is structured into five distinct phases, moving from project setup and safety guardrails to agentic graph orchestration, UI development, and rigorous clinical evaluation.

---

## 2. Master Phase Timeline

```
+-----------------------------------------------------------------------------------+
| Phase 1: Environment Setup & Core Data Schemas                     (Week 1)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 2: RAG Pipeline & Safety Guardrails Integration             (Week 2)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3: LangGraph Agentic Engine & Clinical Logic                 (Week 3)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 4: FastAPI Backend & Interactive UI Development              (Week 4)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 5: MLOps, Evaluation Benchmarks & Safety Verification        (Week 5)       |
+-----------------------------------------------------------------------------------+
```

---

## 3. Phase Breakdown & Key Deliverables

### Phase 1: Environment Setup, Clinical Schemas & Foundation
**Goal**: Establish repository structure, modern Python environment, and Pydantic data schemas for 5-level triage.

- [ ] **Task 1.1: Project Setup & Package Management**
  - Configure Python $3.11+$ virtual environment using `uv` / `poetry`.
  - Install core dependencies (`langgraph`, `langchain`, `fastapi`, `pydantic`, `qdrant-client`, `groq`, `vllm`, `deepeval`).
- [ ] **Task 1.2: Clinical Schema Definition (`schemas.py`)**
  - Implement Pydantic models for `PatientInput`, `VitalSigns`, `ResourceEstimate`, and `TriageAssessment`.
  - Define enum constants for ESI Levels 1 through 5.
- [ ] **Task 1.3: Mock Clinical Test Dataset Creation**
  - Create a benchmark dataset of 50 patient vignettes covering all 5 ESI levels with ground-truth physician labels.

**Deliverables**: Clean repo setup, validated Pydantic schemas, baseline test dataset (`tests/fixtures/vignettes.json`).

---

### Phase 2: RAG Pipeline & Deterministic Guardrails
**Goal**: Build knowledge retrieval grounded in the ESI Handbook and implement safety guardrails for Level 1 red flags.

- [ ] **Task 2.1: ESI Clinical Knowledge Indexing**
  - Ingest ESI Implementation Handbook v4 and emergency clinical guidelines.
  - Chunk text using semantic splitters; generate embeddings via `PubMedBERT` / `bge-large-en-v1.5`.
  - Store vectors in `Qdrant` / `ChromaDB` with hybrid BM25 + dense retrieval.
- [ ] **Task 2.2: Red-Flag Deterministic Interceptor (`guardrails.py`)**
  - Build keyword and semantic regex rules for immediate Level 1 overrides (e.g., severe chest pain, FAST stroke signs, anaphylaxis).
  - Integrate PII anonymization maskers.
- [ ] **Task 2.3: Integration Testing for Guardrails**
  - Verify that 100% of Level 1 test cases bypass standard LLM processing and immediately trigger Red Alert protocols.

**Deliverables**: Functional `Qdrant` vector store, `RedFlagGuardrail` module, zero under-triage safety test suite.

---

### Phase 3: LangGraph Agentic Triage Engine
**Goal**: Orchestrate the conversational state graph to collect symptoms, assess vitals, retrieve ESI guidelines, estimate resources, and assign triage levels.

- [ ] **Task 3.1: LangGraph State Graph Design (`graph.py`)**
  - Define state graph nodes: `IntakeNode`, `RedFlagNode`, `RetrieverNode`, `ResourceEstimatorNode`, `FinalClassifierNode`.
  - Implement dynamic edge routing based on missing patient info or red-flag presence.
- [ ] **Task 3.2: LLM Provider Integration & Prompt Optimization**
  - Connect Groq LLaMA-3.3-70B API as primary LLM engine with fallback to OpenAI / vLLM local instance.
  - Optimize structured JSON output prompting using **DSPy**.
- [ ] **Task 3.3: Resource Counting Logic**
  - Implement reasoning steps for estimating diagnostic/therapeutic hospital resources to accurately distinguish Level 3 ($\ge 2$ resources), Level 4 ($1$ simple resource), and Level 5 ($0$ resources).

**Deliverables**: Fully operational `TriageGraphEngine` capable of multi-turn conversational intake and structured 5-level output generation.

---

### Phase 4: FastAPI Backend & Interactive UI
**Goal**: Wrap the triage engine in a high-throughput async REST API and create an intuitive UI for patients and triage nurses.

- [ ] **Task 4.1: Async REST & WebSocket API (`main.py`)**
  - Build `/api/v1/chat` endpoint with streaming response support.
  - Build `/api/v1/triage/assess` endpoint for batch vignette assessment.
  - Integrate CORS middleware and error handlers.
- [ ] **Task 4.2: Frontend Application Development**
  - Develop interactive Web UI using **Streamlit / Chainlit** or **React + Vite + TailwindCSS**.
  - Components:
    - Conversational Chat Feed with typing indicators.
    - Live Risk Urgency Badge (Color-coded: Red, Orange, Yellow, Green, Blue).
    - Vital Signs Quick Intake Panel.
    - Nurse Summary Handoff Drawer (SBAR format).

**Deliverables**: Running FastAPI server with OpenAPI docs (`/docs`), responsive frontend application.

---

### Phase 5: MLOps, Evaluation Benchmarks & Safety Verification
**Goal**: Implement comprehensive observability, benchmark model accuracy, verify safety guarantees, and finalize documentation.

- [ ] **Task 5.1: MLOps Tracing & Logging**
  - Integrate **LangSmith** and **MLflow** for full turn-by-turn trace logging, latency tracking, and token usage metrics.
- [ ] **Task 5.2: Automated Evaluation Suite (`eval_suite.py`)**
  - Run **DeepEval / Ragas** test pipeline against the 50+ clinical vignette dataset.
  - Calculate Metrics: Precision/Recall per ESI level, Overall Accuracy ($\ge 90\%$), Under-Triage Penalty Score ($0.0$).
- [ ] **Task 5.3: Final Documentation & Clinical Disclaimer Audit**
  - Conduct edge-case testing (jailbreak attempts, nonsensical input, missing vitals).
  - Finalize README and system deployment guide.

**Deliverables**: Evaluation benchmark report, LangSmith dashboard integration, production-ready triage chatbot project repository.

---

## 4. Verification & Milestone Criteria

| Milestone | Target Completion | Verification Method & Target Metric |
| :--- | :--- | :--- |
| **M1: Schemas & Data** | End of Week 1 | 100% Pydantic validation pass rate on test vignettes. |
| **M2: Safety Guardrails** | End of Week 2 | Zero under-triage ($0\%$ false negatives) on Level 1 red-flag tests. |
| **M3: Triage Engine** | End of Week 3 | Successful end-to-end multi-turn conversation execution. |
| **M4: Full Stack API & UI** | End of Week 4 | Web UI responsive with latency $< 2.0\text{s}$ per chat turn. |
| **M5: MLOps & Eval** | End of Week 5 | $\ge 90\%$ overall triage accuracy on benchmark dataset; full LangSmith traces. |
