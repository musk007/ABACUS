import numpy as np, pandas as pd
from pathlib import Path
from utils.labels import ATTR_NAMES

df = pd.read_csv("generated/attr_manifest.csv")
df["area"] = df["w"] * df["h"]
for name in ATTR_NAMES:
    means = df.groupby(name)["area"].agg(["mean", "count"])
    print(f"\n{name}\n{means}")