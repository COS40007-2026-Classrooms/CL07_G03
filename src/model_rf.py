import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
from datetime import datetime

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# load the same preprocessed data used by the neural network.
# using identical train/val/test splits ensures the comparison is fair -
# both models see exactly the same data in the same order
print("Loading preprocessed data...")
x_train = np.load("artifacts/data/x_train.npy")
y_train = np.load("artifacts/data/y_train.npy")
x_val   = np.load("artifacts/data/x_val.npy")
y_val   = np.load("artifacts/data/y_val.npy")
x_test  = np.load("artifacts/data/x_test.npy")
y_test  = np.load("artifacts/data/y_test.npy")

# combine train and val for RF - unlike neural networks, Random Forest
# doesn't use a validation set during training (no gradient descent, no
# early stopping needed). so I merged train+val to give it more data,
# which is standard practice for tree-based models
x_train_full = np.concatenate([x_train, x_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

print(f"Train (full): {x_train_full.shape}")
print(f"Test        : {x_test.shape}")

zone_names = ["Zone 1", "Zone 2", "Zone 3"]

# Random Forest with MultiOutputRegressor for the 3-zone prediction task.
# I chose Random Forest over a single Decision Tree because it averages
# across many trees (n_estimators=200), which reduces variance and prevents
# overfitting - a key limitation of individual decision trees.
#
# key hyperparameters:
# n_estimators=200   - number of trees in the forest. more trees = lower
#                      variance but slower training. 200 is a solid balance
#                      for a dataset this size without excessive compute time.
# max_depth=20       - limits tree depth to prevent individual trees from
#                      memorising the training set (overfitting)
# min_samples_leaf=5 - each leaf must have at least 5 samples, which
#                      smooths predictions on noisy data
# n_jobs=-1          - uses all available CPU cores for parallel training
# random_state=42    - ensures reproducible results across runs
print("Training Random Forest model...")
rf_base = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

# MultiOutputRegressor wraps the base estimator so it can handle
# predicting all 3 zones simultaneously - it trains one RF per zone
# internally and combines the predictions at inference time
model_rf = MultiOutputRegressor(rf_base)
model_rf.fit(x_train_full, y_train_full)
print("Training complete.")

# generate predictions on the held-out test set
y_pred_rf = model_rf.predict(x_test)

# evaluate per zone and overall using the same 3 metrics as the NN:
# RMSE, MAE, and R² - this ensures a fair apples-to-apples comparison
# in the report table
print("Evaluating Random Forest...")
rf_metrics = {}
for i, zone in enumerate(zone_names):
    rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred_rf[:, i])))
    mae  = float(mean_absolute_error(y_test[:, i], y_pred_rf[:, i]))
    r2   = float(r2_score(y_test[:, i], y_pred_rf[:, i]))
    rf_metrics[zone] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"  {zone} - RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

overall_rmse = float(np.sqrt(mean_squared_error(y_test.flatten(), y_pred_rf.flatten())))
overall_mae  = float(mean_absolute_error(y_test.flatten(), y_pred_rf.flatten()))
overall_r2   = float(r2_score(y_test, y_pred_rf, multioutput="uniform_average"))
rf_metrics["overall"] = {"RMSE": overall_rmse, "MAE": overall_mae, "R2": overall_r2}
print(f"  Overall - RMSE: {overall_rmse:.2f} | MAE: {overall_mae:.2f} | R²: {overall_r2:.4f}")

with open("artifacts/metrics/rf_evaluation_metrics.json", "w") as f:
    json.dump(rf_metrics, f, indent=2)
print("Saved: artifacts/metrics/rf_evaluation_metrics.json")

# save the trained model so it can be loaded later without retraining
joblib.dump(model_rf, "artifacts/models/model_rf.pkl")
print("Saved: artifacts/models/model_rf.pkl")

# feature importance - one of the key advantages of Random Forest over
# neural networks is interpretability. I can extract exactly how much
# each input feature contributed to the predictions, which is useful
# for understanding the data and for the report discussion
feature_names = [
    "Temperature", "Humidity", "Wind Speed",
    "general diffuse flows", "diffuse flows",
    "hour_of_day", "day_of_week", "month", "is_weekend"
]

# average feature importance across all 3 zone estimators
importances = np.mean([
    est.feature_importances_ for est in model_rf.estimators_
], axis=0)

# plot feature importances as a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importances)
colours = ["#028090" if imp > np.mean(importances) else "#CADCFC" for imp in importances[sorted_idx]]
ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color=colours)
ax.set_xlabel("Mean Feature Importance (across all zones)", fontsize=12)
ax.set_title("Random Forest - Feature Importances\nTetouan City Power Consumption", fontweight="bold", fontsize=13)
ax.axvline(np.mean(importances), color="tomato", linestyle="--", linewidth=1.5, label="Mean importance")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("artifacts/metrics/rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/rf_feature_importance.png")

# actual vs predicted scatter plots for the RF model - same format as
# the neural network plots so they can be compared side by side in the report
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colours = ["steelblue", "seagreen", "tomato"]

for i, (zone, colour) in enumerate(zip(zone_names, colours)):
    axes[i].scatter(y_test[:, i], y_pred_rf[:, i],
                    alpha=0.3, s=8, color=colour, label="Predictions")
    min_val = min(y_test[:, i].min(), y_pred_rf[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred_rf[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val],
                 "k--", linewidth=1.5, label="Perfect prediction")
    r2   = rf_metrics[zone]["R2"]
    rmse = rf_metrics[zone]["RMSE"]
    axes[i].set_title(f"{zone}\nR²={r2:.4f} | RMSE={rmse:.0f} kW", fontweight="bold")
    axes[i].set_xlabel("Actual (kW)")
    axes[i].set_ylabel("Predicted (kW)")
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Random Forest - Actual vs Predicted\nTetouan City Power Consumption",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("artifacts/metrics/rf_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/rf_actual_vs_predicted.png")

with open("artifacts/metadata/rf_trained.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("\nRandom Forest training and evaluation complete.")
print(f"  Overall RMSE : {overall_rmse:.2f} kW")
print(f"  Overall MAE  : {overall_mae:.2f} kW")
print(f"  Overall R²   : {overall_r2:.4f}")
