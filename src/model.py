import os
import json
import numpy as np
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create required directories
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# Load training and testing data
X_train = np.load("artifacts/data/X_train.npy")
y_train = np.load("artifacts/data/y_train.npy")

X_test = np.load("artifacts/data/X_test.npy")
y_test = np.load("artifacts/data/y_test.npy")

print("Data loaded successfully.")

# Build model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("Starting model training...")

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Save trained model
model.save("artifacts/models/model.keras")

print("Model saved.")

# Save training history
history_dict = history.history

with open("artifacts/metrics/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

# Save model version
with open("artifacts/metadata/model_version.txt", "w") as f:
    f.write("v1.0")

# Save retrain timestamp
with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Training history and metadata saved.")
print("Model training completed successfully.")