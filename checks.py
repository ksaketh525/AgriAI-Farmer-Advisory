# checks.py — quick sanity
from utils import load_and_merge_data
df = load_and_merge_data()
print("Columns:", list(df.columns)); print("Rows:", len(df))
for col in ["average_rain_fall_mm_per_year","avg_temp","pesticides_tonnes","hg/ha_yield"]:
    if col in df.columns:
        print(f"{col}: min={df[col].min():.2f}, p50={df[col].median():.2f}, max={df[col].max():.2f}")
