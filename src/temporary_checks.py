import pandas as pd
from pathlib import Path

OUT_PATH = Path("data/processed/m5_subset.csv")

df = pd.read_csv(OUT_PATH, parse_dates=["date"])
counts = df.groupby(["store_id", "item_id"]).size()

print("series:", len(counts))
print("series with fewer than 730 days:")
print(counts[counts < 730])