#%%
import pandas as pd
# Columns to drop for vignette prompt generation
COLS_TO_DROP = [
    # Administrative
    "encounter_id", "site_id", "country", "arrival_timestamp",
    # Labs (downstream diagnostic data unavailable at initial intake)
    "wbc", "hemoglobin", "platelet_count", "sodium", "potassium", 
    "creatinine", "glucose", "troponin", "bnp", "lactate", "inr"
]
df = pd.read_csv("fedmml_ed_triage_dataset.csv")
# Filtered dataset ready for vignette generation
df_vignettes = df.drop(columns=COLS_TO_DROP)
# %%
df_vignettes
# %%
SAMPLES_PER_LEVEL = 100
df_sampled = (
    df_vignettes
    .groupby("esi_level", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), SAMPLES_PER_LEVEL), random_state=42))
    .reset_index(drop=True)
)
print(df_sampled["esi_level"].value_counts().sort_index())
# Saves lightweight dataset (~150 KB vs 22 MB)
df_sampled.to_csv("ed_triage_vignettes_500.csv", index=False)
# %%
