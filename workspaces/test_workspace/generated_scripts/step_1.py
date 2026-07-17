import os
import json
import pandas as pd
import numpy as np

# Paths
DATA_PATH = os.path.join("data", "active", "copy_test_data.csv")
ARTIFACTS_DIR = "artifacts"
PROFILE_PATH = os.path.join(ARTIFACTS_DIR, "data_profile.json")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Columns to profile
cols = ["TOTAL_REVENUE_USD", "ITEM_COUNT"]

profile = {}

for col in cols:
    series = df[col]
    desc = series.describe()
    q1 = desc["25%"]
    q3 = desc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    skew = series.skew()
    profile[col] = {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]),
        "min": float(desc["min"]),
        "25%": float(q1),
        "50%": float(desc["50%"]),
        "75%": float(q3),
        "max": float(desc["max"]),
        "iqr": float(iqr),
        "lower_outlier_threshold": float(lower_bound),
        "upper_outlier_threshold": float(upper_bound),
        "skewness": float(skew)
    }

# Write profiling results to JSON
with open(PROFILE_PATH, "w") as f:
    json.dump(profile, f, indent=4)