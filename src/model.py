import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # needed for headless CI - no display available on the runner
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # stop tensorflow printing walls of info logs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# load the preprocessed arrays produced by preprocess_new_data.py.
# these come from the previous DVC stage so they should always be
# up to date with whatever data was most recently pushed
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

n_features = x_train.shape[1]  # 9 input features after feature engineering
n_targets  = y_train.shape[1]  # 3 output targets - one per zone

# I built a feedforward neural network for multi-output regression.
# the architecture has two hidden layers, each followed by BatchNormalization
# and Dropout - both of which were covered in Week 6.
#
# BatchNormalization normalises the activations coming out of each layer
# during training. this stabilises learning and reduces how sensitive
# the model is to the initial learning rate - basically keeps things in a
# reasonable range so training doesn't go all over the place.
#
# Dropout randomly zeros out 20% of neurons each forward pass during training.
# I used this as a regularisation technique to stop the network from relying
# too heavily on any single neuron, which helps reduce overfitting on the
# training set.
#
# the output layer has no activation since this is a regression task -
# I want raw continuous predictions, not probabilities
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(n_targets)
], name="tetouan_power_forecasting_model")

# Adam optimiser with gradient clipping (clipnorm=1.0).
# gradient clipping was covered in Week 6 as one of the main strategies
# for dealing with exploding gradients during backpropagation. if any
# gradient's L2 norm goes above 1.0 it gets scaled down automatically.
# this keeps training stable especially in the early epochs when weights
# are still far from their optimal values and gradients can get really large.
# I chose Adam over plain SGD because it adapts the learning rate per
# parameter and generally converges much faster
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# two callbacks to help training run more efficiently.
#
# EarlyStopping watches validation loss and stops training if it hasn't
# improved for 15 epochs, then restores the weights from the best epoch.
# this saves time and stops the model from overfitting - I set patience
# to 15 to give it enough room to work through plateaus before giving up.
#
# ReduceLROnPlateau halves the learning rate if val_loss plateaus for
# 8 epochs. once the model gets close to a minimum, smaller updates
# help it fine-tune without overshooting
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

# save the trained model so the evaluate and monitor stages can load it
# without needing to retrain everything from scratch each time
model.save("artifacts/models/model.keras")
print("Model saved to artifacts/models/model.keras")

# save the full training history as JSON.
# evaluate.py uses this to generate loss curve plots without retraining.
# I also store it so we can compare training runs across retraining cycles
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open("artifacts/metrics/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# write version and timestamp to metadata for traceability.
# this way we always know exactly when this model was trained and
# can track it across the full retraining history
with open("artifacts/metadata/model_version.txt", "w") as f:
    f.write("v1.0")

with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Training history and metadata saved.")
print("Model training completed successfully.")