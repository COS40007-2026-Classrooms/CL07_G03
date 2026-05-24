import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

# create output directories if they don't already exist
os.makedirs("artifacts/data", exist_ok=True)
os.makedirs("artifacts/preprocessing", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# load the raw Tetouan City Power Consumption dataset
# the CSV lives in data/raw/ and is never modified directly -
# all transformations are applied to a copy so the original stays intact
df = pd.read_csv("data/raw/Tetuan City power consumption.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# parse the DateTime column so we can extract time-based features later.
# the format in the CSV is day/month/year hour:minute so we set dayfirst=True
df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)

# data cleaning
print("Cleaning data...")
df.ffill(inplace=True)           # forward-fill any missing values to preserve temporal ordering
df.drop_duplicates(inplace=True) # remove exact duplicate rows
print(f"After cleaning: {len(df)} rows")

# extract time-based features from the DateTime column.
# these were identified in Sprint 1 EDA as having strong correlations
# with consumption patterns - hour of day and month in particular showed
# clear cyclical trends in the data that the model needs to be aware of
df["hour_of_day"]  = df["DateTime"].dt.hour
df["day_of_week"]  = df["DateTime"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month"]        = df["DateTime"].dt.month
df["is_weekend"]   = (df["DateTime"].dt.dayofweek >= 5).astype(int)

# define the feature set and target columns.
# using all five weather features plus the four time-based features derived above.
# the three zone consumption columns are the regression targets
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

# chronological 70/15/15 train/validation/test split.
# a random split is deliberately avoided here because this is time-series data -
# a random split would allow future observations to leak into the training set,
# which would give inflated performance metrics that don't reflect real-world behaviour.
# the chronological split ensures the model is always evaluated on genuinely unseen future data
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

# normalise features using StandardScaler.
# the scaler is fit only on the training set to prevent data leakage -
# fitting on the full dataset would let information from the validation and test
# sets influence the scaling, which would make the evaluation unfair
print("Normalising features...")
scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
x_test  = scaler.transform(x_test)

# save the processed numpy arrays to artifacts/data/
# lowercase filenames used throughout for consistency on case-sensitive Linux filesystems
np.save("artifacts/data/x_train.npy", x_train)
np.save("artifacts/data/y_train.npy", y_train)
np.save("artifacts/data/x_val.npy",   x_val)
np.save("artifacts/data/y_val.npy",   y_val)
np.save("artifacts/data/x_test.npy",  x_test)
np.save("artifacts/data/y_test.npy",  y_test)

# save the fitted scaler so the same transformation can be applied to new data
# during inference or retraining without needing to refit from scratch
joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")

# save the feature column list so downstream scripts know exactly which
# features were used and in what order - important for reproducibility
with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

# write data version and timestamp to metadata
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