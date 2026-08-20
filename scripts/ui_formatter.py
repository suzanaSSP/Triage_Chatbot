"""
UI Formatter for Medical Triage Chatbot Streamlit Interface.
Formats TriageAssessment Pydantic objects into native rich Markdown cards.
"""

from typing import Dict
from schemas import TriageAssessment

# ESI Badge Styling Configuration for Markdown Callouts
ESI_BADGE_CONFIG: Dict[int, Dict[str, str]] = {
    1: {
        "title": "LEVEL 1 - RESUSCITATION",
        "icon": "🔴",
        "badge_tag": "`🔴 ESI LEVEL 1: RESUSCITATION`",
        "default_wait": "0 minutes (Immediate)"
    },
    2: {
        "title": "LEVEL 2 - EMERGENT",
        "icon": "🟠",
        "badge_tag": "`🟠 ESI LEVEL 2: EMERGENT`",
        "default_wait": "< 10 minutes"
    },
    3: {
        "title": "LEVEL 3 - URGENT",
        "icon": "🟡",
        "badge_tag": "`🟡 ESI LEVEL 3: URGENT`",
        "default_wait": "< 30 minutes"
    },
    4: {
        "title": "LEVEL 4 - LESS URGENT",
        "icon": "🟢",
        "badge_tag": "`🟢 ESI LEVEL 4: LESS URGENT`",
        "default_wait": "< 60 minutes"
    },
    5: {
        "title": "LEVEL 5 - NON-URGENT",
        "icon": "🔵",
        "badge_tag": "`🔵 ESI LEVEL 5: NON-URGENT`",
        "default_wait": "< 120 minutes"
    }
}


def format_triage_badge(assessment: TriageAssessment) -> str:
    """Generate Markdown callout badge card for the ESI Level."""
    level_num = int(assessment.triage_level)
    config = ESI_BADGE_CONFIG.get(level_num, ESI_BADGE_CONFIG[5])
    wait_time = assessment.target_wait_time or config["default_wait"]

    markdown = f"""
# {config['icon']} ESI {config['title']}

> **Urgency Category:** `{assessment.urgency_label}` | **Patient ID:** `{assessment.patient_id}` | **Target Wait Time:** `{wait_time}`
"""
    return markdown.strip()


def format_red_flags_alert(red_flags: list) -> str:
    """Format red flags warning callout block if any emergency keywords were detected."""
    if not red_flags:
        return ""

    flags_str = ", ".join(f"`{flag}`" for flag in red_flags)
    return f"""
> 🚨 **RED FLAG EMERGENCY ALERT DETECTED**
> 
> The pure-Python red flag interceptor identified critical emergency indicators: {flags_str}
> 
> *Bypassed routine RAG/LLM assessment. Immediate emergency protocol activated.*
""".strip()


def format_resource_table(assessment: TriageAssessment) -> str:
    """Format Resource Estimate breakdown matrix in markdown table."""
    res = assessment.estimated_resources
    
    def check_mark(val: bool) -> str:
        return "✅ **Yes**" if val else "❌ No"

    markdown = fr"""
### 🧪 Predicted ED Resource Estimator

| Clinical Resource | Predicted Requirement |
| :--- | :--- |
| **Laboratory Tests** (Blood / Urine / Labs) | {check_mark(res.labs)} |
| **ECG / EKG** (Electrocardiogram) | {check_mark(res.ecg)} |
| **Imaging** (X-Ray, CT Scan, Ultrasound, MRI) | {check_mark(res.imaging_xray_ct)} |
| **IV Fluids / IV & IM Medications** | {check_mark(res.iv_fluids_meds)} |
| **Specialty Consult** (Cardiology, Surgery, etc.) | {check_mark(res.specialty_consult)} |
| **Total Predicted Resources Needed** | **`{res.estimated_total_count}` Resource(s)** |

*ESI Guideline Rule: Level 3 ($\ge 2$ resources), Level 4 ($1$ resource), Level 5 ($0$ resources).*
""".strip()
    return markdown


def format_full_assessment_summary(assessment: TriageAssessment) -> str:
    """Assemble complete clinical markdown output card."""
    badge_md = format_triage_badge(assessment)
    red_flag_md = format_red_flags_alert(assessment.red_flags_detected)
    resource_md = format_resource_table(assessment)

    components = [badge_md]
    
    if red_flag_md:
        components.append(red_flag_md)

    handoff_report = f"""
### 🩺 Clinical Summary & Nurse Handoff Report

- **Primary Symptoms / Complaint**: {assessment.primary_symptoms}
- **Recommended Action**: **{assessment.recommended_action}**
- **Target Wait Time**: {assessment.target_wait_time}
"""
    components.append(handoff_report.strip())
    components.append("---")
    components.append(resource_md)
    components.append("---")

    reasoning_section = f"""
### 🧠 Clinical Reasoning & Decision Rationale
{assessment.clinical_reasoning}
"""
    components.append(reasoning_section.strip())

    return "\n\n".join(components)
