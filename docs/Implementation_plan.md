# Medical Triage Chatbot — Project Implementation Plan & Timeline

## 1. Project Overview & Execution Strategy

This implementation plan outlines the step-by-step roadmap for building the **5-Level Medical Triage Chatbot**. The timeline is structured into five distinct phases, moving from project setup to ChromaDB RAG indexing, LangGraph agent orchestration with pure-Python red-flag routing, Chainlit UI integration, and Ragas/LangSmith evaluation.

---

## 2. Master Phase Timeline

```
+-----------------------------------------------------------------------------------+
| Phase 1: Environment Setup & Data Schemas                          (Week 1)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 2: ChromaDB RAG & Pure-Python Red-Flag Interceptor           (Week 2)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3: LangGraph Agent Engine & Groq LLM Integration              (Week 3)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 4: Chainlit Interactive Chat Application                     (Week 4)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 5: LangSmith Tracing & Ragas Evaluation Benchmark            (Week 5)       |
+-----------------------------------------------------------------------------------+
```

---

## 3. Phase Breakdown & Key Deliverables

### Phase 1: Environment Setup, Clinical Schemas & Foundation
**Goal**: Establish clean repository structure, virtual environment, and Pydantic data schemas for 5-level ESI triage.

- [✅] **Task 1.1: Project Setup & Package Management**
  - Configure Python $3.11+$ virtual environment using `uv` / `venv`.
  - Install core dependencies (`langgraph`, `langchain`, `pydantic`, `chromadb`, `groq`, `chainlit`, `ragas`, `langsmith`).
- [✅] **Task 1.2: Clinical Schema Definition (`schemas.py`)**
  - Implement Pydantic models for `PatientInput`, `VitalSigns`, `ResourceEstimate`, and `TriageAssessment`.
  - Define enum constants for ESI Levels 1 through 5.
- [✅] **Task 1.3: Benchmark Vignette Dataset Creation**
  - Create a test dataset of 50 patient vignettes covering all 5 ESI levels with ground-truth physician labels (`tests/fixtures/vignettes.json`).

**Deliverables**: Clean project environment, validated Pydantic schemas, baseline test dataset.

---

### Phase 2: ChromaDB RAG & Pure-Python Red-Flag Interceptor
**Goal**: Build knowledge retrieval grounded in the ESI Handbook and implement a pure-Python red-flag keyword node in LangGraph.

- [ ] **Task 2.1: ESI Clinical Knowledge Indexing (ChromaDB)**
  - Ingest ESI Implementation Handbook v4 and emergency clinical guidelines.
  - Generate embeddings using `bge-large-en-v1.5` / `PubMedBERT` and index into persistent **ChromaDB** (`./chroma_db`).
- [✅] **Task 2.2: Pure-Python Red-Flag Deterministic Node (`red_flags.py`)**
  - Implement regex & keyword rules for high-acuity symptoms (e.g., *"crushing chest pain radiating to arm"*, *"unresponsive"*, *"anaphylaxis"*, *"stroke signs"*).
  - Ensure this node bypasses RAG and LLM calls when triggered, directly returning a Level 1 Emergency Alert.
- [✅] **Task 2.3: Red-Flag Unit Test Suite**
  - Unit test 100% of emergency vignettes to guarantee zero under-triage and instant execution without LLM latency.

**Deliverables**: Persistent ChromaDB vector store, deterministic `RedFlagNode`, unit test suite.

---

### Phase 3: LangGraph Agent Engine & Groq LLM Integration
**Goal**: Orchestrate the conversational state graph to collect symptoms, assess vitals, retrieve ESI guidelines, estimate resources, and assign triage levels via Groq API.

- [ ] **Task 3.1: LangGraph State Graph Design (`graph.py`)**
  - Define graph state and nodes: `RedFlagNode`, `IntakeNode`, `RetrieverNode`, `ResourceEstimatorNode`, `FinalClassifierNode`.
  - Configure conditional routing: If `RedFlagNode` fires $\rightarrow$ bypass to `EmergencyAlertNode`; else $\rightarrow$ proceed through RAG & Groq LLM.
- [ ] **Task 3.2: Groq LLM API Integration (`LLaMA-3.3-70B`)**
  - Configure Groq API client with structured Pydantic output parsing.
- [ ] **Task 3.3: Resource Counting & Decision Reasoning**
  - Implement prompt logic to estimate hospital resources (0, 1, or $\ge 2$) for Level 3 vs Level 4 vs Level 5 classification.

**Deliverables**: Operational `TriageGraphEngine` performing stateful triage logic.

---

### Phase 4: Chainlit Interactive Chat Application
**Goal**: Build a modern, responsive chat interface using Chainlit that connects directly to the LangGraph engine.

- [ ] **Task 4.1: Chainlit Application (`app.py`)**
  - Implement `@cl.on_chat_start` and `@cl.on_message` handlers connecting to `TriageGraphEngine`.
  - Stream Groq LLM output tokens in real-time to the Chainlit UI.
- [ ] **Task 4.2: UI Enhancements & Nurse Handoff Summary**
  - Add interactive element cards for color-coded triage badges (Red = Level 1, Orange = Level 2, Yellow = Level 3, Green = Level 4, Blue = Level 5).
  - Render clinical SBAR summaries for nurse review.

**Deliverables**: Single-command runnable local chat app (`chainlit run app.py`).

---

### Phase 5: LangSmith Tracing & Ragas Evaluation Benchmark
**Goal**: Integrate full execution tracing with LangSmith and run automated RAG/accuracy benchmarks using Ragas.

- [ ] **Task 5.1: LangSmith Tracing Setup**
  - Configure `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to log execution graphs, node latencies, and token usage to **LangSmith**.
- [ ] **Task 5.2: Ragas Benchmark Pipeline (`eval_ragas.py`)**
  - Run Ragas evaluation over the 50 vignette test dataset.
  - Calculate `faithfulness`, `answer_relevancy`, `context_precision`, and `triage_accuracy`.
- [ ] **Task 5.3: Project Documentation & Final Audit**
  - Finalize README.md with clear setup instructions and execution guide.

**Deliverables**: Complete Ragas evaluation report, active LangSmith dashboard workspace, production-ready codebase.

---

## 4. Verification & Milestone Criteria

| Milestone | Target Completion | Verification Method & Target Metric |
| :--- | :--- | :--- |
| **M1: Schemas & Data** | End of Week 1 | 100% Pydantic validation pass rate on test vignettes. |
| **M2: ChromaDB & Red Flag** | End of Week 2 | $0\%$ false negatives on Level 1 red-flag regex tests; instant bypass verified. |
| **M3: LangGraph & Groq** | End of Week 3 | Successful end-to-end multi-turn conversation and ESI level assignment. |
| **M4: Chainlit Chat App** | End of Week 4 | Local Chainlit UI streaming responses with $< 1.5\text{s}$ turn latency. |
| **M5: LangSmith & Ragas** | End of Week 5 | $\ge 90\%$ triage accuracy on benchmark vignettes; full LangSmith traces active. |
