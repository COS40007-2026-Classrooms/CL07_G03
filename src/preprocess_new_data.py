#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

#Loading Data
df = pd.read_csv("data/raw/Tetuan City power consumption.csv")
print(f"Loaded {len(df)} rows")

#Checking for missing values, then forward filling, as well as removing duplicates
print("Data Cleaning:")
df.ffill(inplace=True)               # forward-fill missing values
df.drop_duplicates(inplace=True)     # remove duplicates
print(f"After cleaning: {len(df)} rows")

# Seperating the target columns from the features
feature_cols = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows"
]

target_cols = [
    "Zone 1 Power Consumption",
    "Zone 2 Power Consumption",
    "Zone 3 Power Consumption"
]

#Creating x & y for input into models
x = df[feature_cols].values
y = df[target_cols].values

# Implementing the chronological 70/15/15 split as agreed upon by group members
print("Creating Data Split")
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

x_train = x[:train_end]
x_val   = x[train_end:val_end]
x_test  = x[val_end:]

y_train = y[:train_end]
y_val   = y[train_end:val_end]
y_test  = y[val_end:]

print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

#Normalising the data
print("Normalising the Data")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fits only onto train
x_val   = scaler.transform(x_val)
x_test  = scaler.transform(x_test)

# Saving the numpy arrays
print("Artifacts saved")
np.save("artifacts/data/x_train.npy", x_train)
np.save("artifacts/data/y_train.npy", y_train)
np.save("artifacts/data/x_val.npy",   x_val)
np.save("artifacts/data/y_val.npy",   y_val)
np.save("artifacts/data/x_test.npy",  x_test)
np.save("artifacts/data/y_test.npy",  y_test)

#Saving the scalar and feature columns
joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")

with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

# Saving the metadata
with open("artifacts/metadata/data_version.txt", "w") as f:
    f.write("v1.0")

with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("artifacts saved.")
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape}")
print(f"y_test shape:  {y_test.shape}")