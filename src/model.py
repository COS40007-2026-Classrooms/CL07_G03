import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend - no display available in CI environments
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow info/warning logs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# create output directories if they don't already exist
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# load preprocessed data from artifacts/data/.
# lowercase filenames are used throughout to stay consistent on
# case-sensitive Linux filesystems (GitHub Actions runs on ubuntu-latest)
x_train = np.load("artifacts/data/x_train.npy")
y_train = np.load("artifacts/data/y_train.npy")
x_val   = np.load("artifacts/data/x_val.npy")
y_val   = np.load("artifacts/data/y_val.npy")
x_test  = np.load("artifacts/data/x_test.npy")
y_test  = np.load("artifacts/data/y_test.npy")

print(f"x_train : {x_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"x_val   : {x_val.shape}")
print(f"x_test  : {x_test.shape}")

n_features = x_train.shape[1]  # number of input features (9 after feature engineering)
n_targets  = y_train.shape[1]  # number of output targets (3 zones)

# build a feedforward regression network.
# the architecture uses two hidden layers with batch normalisation and dropout:
#   - batch normalisation stabilises training by normalising layer inputs,
#     which reduces sensitivity to learning rate and speeds up convergence
#   - dropout (0.2) randomly zeros 20% of neurons each forward pass during
#     training, which acts as regularisation and reduces overfitting
#   - the output layer has n_targets units with no activation since this
#     is a regression task - we want raw continuous predictions, not probabilities
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(n_targets)  # no activation - regression output for all three zones
], name="tetouan_power_forecasting_model")

# adam optimiser with gradient clipping to prevent exploding gradients.
# clipnorm=1.0 scales down any gradient whose L2 norm exceeds 1.0 during
# backpropagation - this keeps training stable especially in early epochs
# when weights are far from their optimal values
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# callbacks used during training:
#   - EarlyStopping: stops training if val_loss hasn't improved for 15 epochs,
#     then restores the best weights seen during training. this prevents
#     wasted compute and avoids overfitting to the training set
#   - ReduceLROnPlateau: halves the learning rate if val_loss plateaus for
#     8 epochs - helps the model fine-tune once it gets close to a minimum
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
]

print("Training model...")

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print(f"Training finished at epoch {len(history.history['loss'])}")
print(f"Final train loss : {history.history['loss'][-1]:.2f}")
print(f"Final val loss   : {history.history['val_loss'][-1]:.2f}")

# save the final trained model
model.save("artifacts/models/model.keras")
print("Model saved to artifacts/models/model.keras")

# save full training history as JSON so it can be inspected later
# without needing to retrain - also used by evaluate.py for plotting
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open("artifacts/metrics/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# save model version and retrain timestamp to metadata
with open("artifacts/metadata/model_version.txt", "w") as f:
    f.write("v1.0")

with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Training history and metadata saved.")
print("Model training completed successfully.")