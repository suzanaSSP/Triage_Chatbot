# pyrefly: ignore [missing-import]
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ESILevel(int, Enum):
    """Emergency Severity Index (ESI) Triage Levels 1 through 5."""
    LEVEL_1_RESUSCITATION = 1
    LEVEL_2_EMERGENT = 2
    LEVEL_3_URGENT = 3
    LEVEL_4_LESS_URGENT = 4
    LEVEL_5_NON_URGENT = 5

class VitalSigns(BaseModel):
    heart_rate: Optional[int] = Field(None, description="Beats per minute (bpm)")
    blood_pressure_systolic: Optional[int] = Field(None, description="mmHg")
    blood_pressure_diastolic: Optional[int] = Field(None, description="mmHg")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    oxygen_saturation: Optional[float] = Field(None, description="SpO2 percentage")
    temperature_celsius: Optional[float] = Field(None, description="Body temp in °C")

class ResourceEstimate(BaseModel):
    labs: bool = Field(default=False, description="Blood work / lab tests")
    ecg: bool = Field(default=False, description="Electrocardiogram")
    imaging_xray_ct: bool = Field(default=False, description="X-Ray, CT scan, or MRI")
    iv_fluids_meds: bool = Field(default=False, description="IV fluids or IV/IM medications")
    specialty_consult: bool = Field(default=False, description="Specialist consultation")
    estimated_total_count: int = Field(default=0, description="Total predicted distinct clinical resources")

class PatientInput(BaseModel):
    patient_id: Optional[str] = Field(None, description="Unique identifier for the patient/session")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Biological sex / gender")
    chief_complaint: str = Field(..., description="Primary reason for visit / main symptoms reported")
    vignette_text: List[str] = Field(default_factory=list, description="List of individual reported symptoms")
  
class TriageAssessment(BaseModel):
    patient_id: str
    triage_level: ESILevel = Field(..., ge=1, le=5, description="ESI Triage Level 1 to 5")
    urgency_label: str  # Immediate, Emergent, Urgent, Less Urgent, Non-Urgent
    primary_symptoms: List[str]
    estimated_resources: ResourceEstimate
    clinical_reasoning: str
    red_flags_detected: List[str]
    recommended_action: str
    target_wait_time: str
