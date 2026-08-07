import asyncio
import json
import os
import pandas as pd
from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError
from tqdm.asyncio import tqdm_asyncio

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found in .env file!")

# Initialize Groq client
client = AsyncGroq(api_key=GROQ_API_KEY)

# System / User Prompt Template
PROMPT_TEMPLATE = """You are an expert emergency physician creating a concise, realistic clinical patient vignette for triage benchmark evaluation.

Patient Data:
- Demographics: {age}-year-old {sex_full}
- Chief Complaint: {chief_complaint}
- Clinical Notes: {clinical_notes}
- Vital Signs:
  - Blood Pressure: {bp_str}
  - Heart Rate: {hr_str}
  - Respiratory Rate: {rr_str}
  - Temperature: {temp_str}
  - Oxygen Saturation (SpO2): {spo2_str}
  - Pain Score: {pain_str}

Instructions:
1. Synthesize this data into a clear, realistic 1 to 2 sentence clinical patient vignette describing how the patient presents at triage.
2. Do NOT mention ESI triage levels, numerical scores, or specific resource counts.
3. Keep the focus strictly on presenting symptoms, clinical appearance, and vital sign findings.
4. Output ONLY the 1-2 sentence vignette text.
"""

async def generate_single_vignette(client: AsyncGroq, semaphore: asyncio.Semaphore, row: pd.Series) -> str:
    """Generate 1-2 sentence vignette for a single patient record with exponential backoff retry."""
    sex_map = {"M": "male", "F": "female"}
    sex_full = sex_map.get(str(row["sex"]).strip().upper(), "patient")
    
    # Format vitals safely
    bp_str = f"{row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f} mmHg" if pd.notna(row['systolic_bp']) and pd.notna(row['diastolic_bp']) else "Not measured"
    hr_str = f"{row['heart_rate']:.0f} bpm" if pd.notna(row['heart_rate']) else "Not measured"
    rr_str = f"{row['respiratory_rate']:.0f} bpm" if pd.notna(row['respiratory_rate']) else "Not measured"
    temp_str = f"{row['temperature']:.1f}°C" if pd.notna(row['temperature']) else "Not measured"
    spo2_str = f"{row['spo2']:.1f}%" if pd.notna(row['spo2']) else "Not measured"
    pain_str = f"{row['pain_score']:.0f}/10" if pd.notna(row['pain_score']) else "Not reported"
    
    notes_str = str(row["clinical_notes"]) if pd.notna(row["clinical_notes"]) else "None reported"

    prompt = PROMPT_TEMPLATE.format(
        age=row["age"],
        sex_full=sex_full,
        chief_complaint=row["chief_complaint"],
        clinical_notes=notes_str,
        bp_str=bp_str,
        hr_str=hr_str,
        rr_str=rr_str,
        temp_str=temp_str,
        spo2_str=spo2_str,
        pain_str=pain_str,
    )

    async with semaphore:
        backoff = 2.0
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150,
                )
                return response.choices[0].message.content.strip()
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff *= 2.0
            except Exception as e:
                print(f"Error for patient {row.get('patient_id', '')}: {e}")
                await asyncio.sleep(1.0)
        
        # Fallback to direct clinical notes if API fails
        return f"{row['age']}-year-old {sex_full} presents with {row['chief_complaint']}. {notes_str}"

async def main():
    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(10)

    csv_file = "ed_triage_vignettes_500.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found. Please run data_cleaning.py first.")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}. Generating 1-2 sentence vignettes via Groq API...")

    tasks = [generate_single_vignette(client, semaphore, row) for _, row in df.iterrows()]
    vignettes = await tqdm_asyncio.gather(*tasks, desc="Generating Vignettes")

    df["vignette_text"] = vignettes

    # 1. Save enriched CSV
    output_csv = "ed_triage_vignettes_500_enriched.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved enriched CSV with vignettes to: {output_csv}")

    # 2. Save JSON fixture for testing/eval
    os.makedirs("tests/fixtures", exist_ok=True)
    json_records = []
    for _, row in df.iterrows():
        json_records.append({
            "patient_id": str(row["patient_id"]),
            "esi_level": int(row["esi_level"]),
            "vignette_text": row["vignette_text"],
            "demographics": {
                "age": int(row["age"]),
                "sex": str(row["sex"])
            },
            "chief_complaint": str(row["chief_complaint"]),
            "vitals": {
                "systolic_bp": float(row["systolic_bp"]) if pd.notna(row["systolic_bp"]) else None,
                "diastolic_bp": float(row["diastolic_bp"]) if pd.notna(row["diastolic_bp"]) else None,
                "heart_rate": float(row["heart_rate"]) if pd.notna(row["heart_rate"]) else None,
                "respiratory_rate": float(row["respiratory_rate"]) if pd.notna(row["respiratory_rate"]) else None,
                "temperature": float(row["temperature"]) if pd.notna(row["temperature"]) else None,
                "spo2": float(row["spo2"]) if pd.notna(row["spo2"]) else None,
                "pain_score": float(row["pain_score"]) if pd.notna(row["pain_score"]) else None,
            }
        })

    output_json_500 = "tests/fixtures/vignettes_500.json"
    with open(output_json_500, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)
    print(f"✅ Saved 500 JSON vignettes to: {output_json_500}")

    # 3. Save 50-row baseline fixture (Task 1.3 requirement)
    output_json_50 = "tests/fixtures/vignettes.json"
    # Take 10 stratified samples from each ESI level for the 50-sample benchmark
    df_50 = df.groupby("esi_level", group_keys=False).apply(lambda x: x.sample(n=10, random_state=42))
    json_records_50 = [r for r in json_records if r["patient_id"] in set(df_50["patient_id"])]
    with open(output_json_50, "w", encoding="utf-8") as f:
        json.dump(json_records_50, f, indent=2)
    print(f"✅ Saved 50 baseline JSON vignettes to: {output_json_50}")

if __name__ == "__main__":
    asyncio.run(main())
