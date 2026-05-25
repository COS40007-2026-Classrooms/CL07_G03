import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

# make sure all the output folders exist before we try writing to them
os.makedirs("artifacts/data", exist_ok=True)
os.makedirs("artifacts/preprocessing", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# load the raw Tetouan City Power Consumption dataset from data/raw/.
# I never modify the original CSV directly - all transformations get saved
# separately so the raw file always stays intact as the source of truth
df = pd.read_csv("data/raw/Tetuan City power consumption.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# parse the DateTime column using mixed format since the CSV has
# inconsistent date formatting across some rows
df["DateTime"] = pd.to_datetime(df["DateTime"], format="mixed")

# basic data cleaning.
# I used forward-fill for missing values rather than dropping them because
# this is time-series data - carrying the last known value forward preserves
# the temporal pattern better than leaving gaps or using a global mean.
# duplicate rows get dropped as well just to be safe
print("Cleaning data...")
df.ffill(inplace=True)
df.drop_duplicates(inplace=True)
print(f"After cleaning: {len(df)} rows")

# extract time-based features from the DateTime column.
# from our Sprint 1 EDA, hour of day and month both showed strong
# correlations with consumption - which makes sense given peak usage
# hours during the day and seasonal patterns across the year.
# I also added day_of_week and is_weekend since weekday vs weekend
# consumption patterns tend to be quite different
df["hour_of_day"] = df["DateTime"].dt.hour
df["day_of_week"] = df["DateTime"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"]       = df["DateTime"].dt.month
df["is_weekend"]  = (df["DateTime"].dt.dayofweek >= 5).astype(int)

# 9 input features total - 5 weather features from the original dataset
# plus the 4 time-based features derived above.
# the targets are the three zone-level power consumption columns
feature_cols = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend"
]

target_cols = [
    "Zone 1 Power Consumption",
    "Zone 2 Power Consumption",
    "Zone 3 Power Consumption"
]

x = df[feature_cols].values
y = df[target_cols].values

# chronological 70/15/15 split - this is really important for time-series data.
# I avoided a random split here because it would allow future observations
# to end up in the training set, which basically means the model gets to
# peek at the future during training and ends up with inflated metrics
# that don't reflect real performance. keeping it chronological avoids
# that data leakage problem entirely
print("Splitting data...")
n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

x_train = x[:train_end]
x_val   = x[train_end:val_end]
x_test  = x[val_end:]

y_train = y[:train_end]
y_val   = y[train_end:val_end]
y_test  = y[val_end:]

print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

# normalise features using StandardScaler (zero mean, unit variance).
# one thing to note - I only fit the scaler on the training set and then
# apply the same transformation to val and test. fitting on the full dataset
# would let test set statistics influence the scaling, which is another
# form of data leakage we want to avoid
print("Normalising features...")
scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
x_test  = scaler.transform(x_test)

# save everything to artifacts/data/ using lowercase filenames throughout.
# lowercase matters here because GitHub Actions runs on ubuntu-latest which
# is case-sensitive - mixing cases would cause file not found errors in CI
np.save("artifacts/data/x_train.npy", x_train)
np.save("artifacts/data/y_train.npy", y_train)
np.save("artifacts/data/x_val.npy",   x_val)
np.save("artifacts/data/y_val.npy",   y_val)
np.save("artifacts/data/x_test.npy",  x_test)
np.save("artifacts/data/y_test.npy",  y_test)

# save the fitted scaler so I can apply the exact same transformation
# to any new data that comes in later without having to refit from scratch
joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")

# save the feature list so downstream scripts know which features were used
# and in what order - this helps with reproducibility across retraining runs
with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

with open("artifacts/metadata/data_version.txt", "w") as f:
    f.write("v1.0")

with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Preprocessing complete.")
print(f"  x_train : {x_train.shape}")
print(f"  x_val   : {x_val.shape}")
print(f"  x_test  : {x_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_val   : {y_val.shape}")
print(f"  y_test  : {y_test.shape}")
print(f"  Features: {feature_cols}")
print(f"  Targets : {target_cols}")