import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend - needed for CI with no display
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model

os.makedirs("artifacts/metrics", exist_ok=True)

# load the trained model and the held-out test set.
# the test set was never seen during training or validation so this gives
# a fair measure of how the model performs on genuinely unseen data
model  = load_model("artifacts/models/model.keras")
x_test = np.load("artifacts/data/x_test.npy")
y_test = np.load("artifacts/data/y_test.npy")

print(f"Evaluating on {len(x_test)} test samples...")

y_pred     = model.predict(x_test, verbose=0)
zone_names = ["Zone 1", "Zone 2", "Zone 3"]

# I calculated RMSE, MAE, and R² for each zone separately, then overall.
# using three metrics rather than just one gives a more complete picture -
# RMSE penalises large errors more heavily than MAE, while R² tells us
# how much of the variance in the target the model actually explains.
# reporting all three makes it easier to see if the model is struggling
# on any particular zone
metrics = {}
for i, zone in enumerate(zone_names):
    rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
    mae  = float(mean_absolute_error(y_test[:, i], y_pred[:, i]))
    r2   = float(r2_score(y_test[:, i], y_pred[:, i]))
    metrics[zone] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"  {zone} - RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

# overall metrics across all three zones combined
all_rmse = float(np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten())))
all_mae  = float(mean_absolute_error(y_test.flatten(), y_pred.flatten()))
all_r2   = float(r2_score(y_test, y_pred, multioutput="uniform_average"))
metrics["overall"] = {"RMSE": all_rmse, "MAE": all_mae, "R2": all_r2}
print(f"  Overall - RMSE: {all_rmse:.2f} | MAE: {all_mae:.2f} | R²: {all_r2:.4f}")

with open("artifacts/metrics/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: artifacts/metrics/evaluation_metrics.json")

# load training history to plot the loss curves
with open("artifacts/metrics/training_history.json") as f:
    history = json.load(f)

# plot 1 - loss curves over training.
# I did two panels side by side - left shows all epochs so you can see the
# full picture, right zooms in from epoch 5 onwards where the actual
# convergence behaviour happens. the early epochs tend to have really high
# loss which squashes the rest of the chart and makes it hard to read
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["loss"],     label="Train Loss", linewidth=2, color="steelblue")
axes[0].plot(history["val_loss"], label="Val Loss",   linewidth=2, color="tomato")
axes[0].set_title("Training & Validation Loss (All Epochs)", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["loss"][5:],     label="Train Loss", linewidth=2, color="steelblue")
axes[1].plot(history["val_loss"][5:], label="Val Loss",   linewidth=2, color="tomato")
axes[1].set_title("Loss (Epoch 5 Onwards)", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/metrics/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/loss_curves.png")

# plot 2 - actual vs predicted scatter for each zone.
# ideally all points should sit close to the diagonal (predicted = actual).
# anything far from the diagonal is where the model is over or under predicting.
# I plotted all three zones side by side so it's easy to compare performance
# across zones at a glance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colours = ["steelblue", "seagreen", "tomato"]

for i, (zone, colour) in enumerate(zip(zone_names, colours)):
    axes[i].scatter(y_test[:, i], y_pred[:, i],
                    alpha=0.3, s=8, color=colour, label="Predictions")
    min_val = min(y_test[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val],
                 "k--", linewidth=1.5, label="Perfect prediction")
    r2   = metrics[zone]["R2"]
    rmse = metrics[zone]["RMSE"]
    axes[i].set_title(f"{zone}\nR²={r2:.4f} | RMSE={rmse:.0f} kW", fontweight="bold")
    axes[i].set_xlabel("Actual (kW)")
    axes[i].set_ylabel("Predicted (kW)")
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Actual vs Predicted - Tetouan City Power Consumption",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("artifacts/metrics/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/actual_vs_predicted.png")

print("Evaluation complete.")