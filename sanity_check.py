"""
Sanity Check for Final Stable Pipeline
"""

from pathlib import Path
import pandas as pd

from src.pipeline import build_preprocessor, ensure_full_schema

print("\n📌 Running FINAL preprocessing sanity check...")

ROOT = Path(".")
DATA_PATH = ROOT / "data" / "upscaled_data.csv"

df = pd.read_csv(DATA_PATH)

# Take a bigger sample to avoid empty rows
df_sample = ensure_full_schema(df.head(300))

pre = build_preprocessor()

pre.fit(df_sample)
X = pre.transform(df_sample)

print("\n🎉 SANITY CHECK PASSED!")
print("Transformed shape:", X.shape)
print("Everything working perfectly.\n")
