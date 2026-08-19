from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Cleaned keyword list (duplicates removed, structured)
KEYWORDS_ESI_1 = [
    "unresponsiveness", "unresponsive", "active seizure", "seizure", 
    "occluded airway", "ineffective gas exchange", "impaired gas exchange", 
    "ineffective airway clearance", "ineffective respiratory pattern", 
    "ineffective tissue perfusion", "anaphylaxis", 
    "hypotension with signs of hypoperfusion", "hypotension", 
     "hypoglycemia", "bradycardia", "severe tachycardia", 
    "flaccid infant", "cardiac pulmonary arrest", "cardiac arrest", 
    "appears imminent", "immediate intervention", "respiratory distress",
    "hemorrhage", "trauma of head", "trauma of neck", "trauma of abdomen", "trauma of chest"
]

NEGATION_TERMS = ["no ", "denies ", "without ", "ruled out "]

def is_red_flag(patient_sentence: str) -> Tuple[bool, List[str]]:
    """
    Evaluates patient text against deterministic ESI-1 emergency keywords.
    Returns (is_red_flag: bool, matched_keywords: List[str]).
    """
    normalized = patient_sentence.lower()
    matched = []

    for kw in KEYWORDS_ESI_1:
        if kw in normalized:
            # Simple negation check: ignore if preceded by negation terms
            kw_idx = normalized.find(kw)
            prefix = normalized[max(0, kw_idx - 15):kw_idx]
            if not any(neg in prefix for neg in NEGATION_TERMS):
                matched.append(kw)

    return len(matched) > 0, matched


def test():
    data_path = Path(__file__).parent / "ed_triage_vignettes_500_enriched.csv"
    df = pd.read_csv(data_path)
    
    df_esi_1 = df[df['esi_level'] == 1]
    incorrect = []

    for text in df_esi_1["vignette_text"].to_list():
        flagged, _ = is_red_flag(text)
        if not flagged:
            incorrect.append(text)

    if not incorrect:
        print(f"✅ Success! All {len(df_esi_1)} ESI Level 1 vignettes correctly identified.")
    else:
        print(f"⚠️ {len(incorrect)} vignettes misclassified: {incorrect}")


if __name__ == "__main__":
    test()
