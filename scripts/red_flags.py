import pandas as pd
from pathlib import Path
DATA_PATH = Path(__file__).parent / "ed_triage_vignettes_500_enriched.csv"

keywords_esi_1 = [
    "unresponsiveness", "active seizure", "occluded airway", "ineffective gas exchange", 
 "ineffective airway clearance", "ineffective respiratory pattern",
 "impaired gas exchange", "ineffective tissue perfusion", "unresponsive", "anaphylaxis", 
 "hypotension with signs of hypoperfusion", "chest pain", "hypoglycemia", "bradycardia",
 "severe tachycardia", "flaccid infant", "cardiac pulmonary arrest", "appears imminent",
 "trauma of head", "trauma of neck", "trauma of abdomen", "trauma of chest", "immediate intervention",
 "hypotension", "cardiac arrest", "respiratory distress", "hemorrhage", "seizure"
]

def check_red_flags(patient_sentence: str) -> tuple[bool, list[str]]:
    normalized = patient_sentence.lower()
    matched = [kw for kw in keywords_esi_1 if kw in normalized]
    return len(matched) > 0, matched

def test():
    answers = []
    incorrect = []
    df = pd.read_csv(DATA_PATH)
    df_esi_1 = df[df['esi_level'] == 1]

    for text in df_esi_1["vignette_text"].to_list():
        answer = 1 if check_red_flags(text) else 0
        if answer == 0:
            incorrect.append(text)

        answers.append(text)

    if len(incorrect) == 0:
        print("Everything is working")
    else:
        print(f"There are {len(incorrect)} vignettes incorrectly classified out of {len(df_esi_1)}")
        print(f"These are the ones incorrectly classified: {incorrect}")




if __name__ == "__main__":
    test()
    
