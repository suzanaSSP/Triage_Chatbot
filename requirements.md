# Medical Triage Chatbot — Product Requirements Document (PRD)

## 1. Overview & Vision
The **Medical Triage Chatbot** is an AI-powered conversational clinical assistant designed to interact with patients, gather symptom information and vital signs, and accurately categorize the patient's condition into one of five emergency triage levels (Emergency Severity Index - ESI framework). 

The system aims to assist healthcare intake teams, reduce emergency department bottlenecking, provide instant patient triage guidance, and flag life-threatening conditions immediately with zero tolerance for under-triaging critical emergencies.

---

## 2. Emergency Triage Classification System (5 Levels)

The chatbot classifies incoming patient reports into five distinct emergency acuity levels:

| Triage Level | Classification Name | Clinical Definition & Criteria | Target Action & Response Time |
| :--- | :--- | :--- | :--- |
| **Level 1** | **Immediate** | Requires immediate life-saving intervention (e.g., cardiac arrest, anaphylaxis, severe respiratory distress, unresponsiveness, major trauma). | Immediate evaluation by physician & resuscitation team. Immediate RED ALERT trigger. |
| **Level 2** | **Emergent** | High-risk situation, acute mental status change, severe pain/distress, or unstable vital signs. | Evaluated by medical staff within 10 minutes. Urgent notification dispatched. |
| **Level 3** | **Urgent** | Vital signs are stable, but the condition requires **2 or more hospital resources** (e.g., lab tests + IV fluids + X-ray). | Evaluated within 30 minutes. Placed in standard queue. |
| **Level 4** | **Less Urgent** | Vital signs are stable, requiring **only 1 simple hospital resource** (e.g., simple suture/stitches, a single X-ray, tetanus shot). | Safe wait time up to 1 hour. |
| **Level 5** | **Non-Urgent** | Vital signs are stable, requiring **no hospital resources** (e.g., prescription refill, minor rash, suture removal). | Safe wait time over 1 hour / outpatient redirection. |

---

## 3. Core Functional Requirements

### 3.1 Patient Intake & Dynamic Clinical Interviewing
- **Adaptive Questioning**: Dynamically ask follow-up questions regarding symptom onset, severity (1-10 scale), location, duration, and past medical history.
- **Vital Signs Extraction**: Prompt for and process vital signs when available (Heart Rate, Blood Pressure, Respiratory Rate, Oxygen Saturation \(SpO_2\), Body Temperature, GCS / Mental Alertness).
- **Multi-lingual & Conversational Understanding**: Support natural human phrasing, slang, and non-expert symptom descriptions (e.g., "my chest feels like an elephant is sitting on it" $\rightarrow$ suspected acute coronary syndrome).

### 3.2 Red-Flag & Safety Guardrails
- **Immediate Escalation Override**: If at any point during intake a Level 1 "Red Flag" symptom is detected (e.g., chest pain with radiation, stroke signs like FAST, severe dyspnea), immediately interrupt interview, trigger Level 1 alert, and instruct patient to call 911 / seek emergency services.
- **Under-Triage Prevention**: Implement strict safety rules prioritizing patient safety over hospital resource conservation.
- **Medical Disclaimer**: Display persistent, clear medical disclaimers declaring that the system is an AI triage assistant, not a replacement for professional clinical judgment.

### 3.3 Clinical Reasoning & Resource Estimation
- **Resource Counting Logic**: For Level 3 vs Level 4 vs Level 5 differentiation, accurately estimate required clinical resources (Labs, ECG, Imaging, IV Medication, Specialty Consults).
- **Explainable Triage Recommendation**: Provide a structured explanation detailing *why* a specific triage level was assigned, referencing detected risk factors and estimated resources.

### 3.4 Nurse/Physician Dashboard & Handoff
- **Summary Generation**: Generate a concise clinical summary (SBAR format: Situation, Background, Assessment, Recommendation) for attending nurses/physicians.
- **Real-Time Level Badge**: Display color-coded urgency badges (Red = Level 1, Orange = Level 2, Yellow = Level 3, Green = Level 4, Blue = Level 5).

---

## 4. Non-Functional Requirements (NFRs)

- **Performance & Latency**:
  - Initial Red-Flag / Level 1 classification check latency $< 1.0 \text{ second}$.
  - Full conversational turn-around time $< 2.0 \text{ seconds}$.
- **Reliability & Availability**:
  - $99.9\%$ system uptime with automated LLM fallback (e.g., switching from primary LLM to backup provider/local model if API fails).
- **Safety & Metrics**:
  - **Zero Under-Triage Rate** for Level 1 & Level 2 cases during validation.
  - Overall triage accuracy $\ge 90\%$ aligned with expert emergency physician benchmarks.
- **Privacy & Compliance**:
  - HIPAA-aligned data processing: PII/PHI anonymization before sending prompts to external APIs.
  - Zero persistent storage of unencrypted patient identification data.
- **Auditability**:
  - Complete logging of all triage inputs, LLM reasoning chains, guardrail triggers, and final triage levels for clinical governance.

---

## 5. Target User Personas

1. **Patient / Care Seeker**: Interacts via chat UI to evaluate symptom urgency before arriving at or while sitting in an emergency waiting room.
2. **Triage Nurse / Intake Specialist**: Receives automated patient intake summaries, reviews triage score recommendations, and confirms final priority.
3. **Medical Director / System Admin**: Reviews MLOps dashboard for triage accuracy metrics, safety guardrail logs, and prompt performance.
