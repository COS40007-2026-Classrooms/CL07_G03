import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("artifacts/metrics", exist_ok=True)

# load evaluation metrics from both models.
# both files use the same structure so comparison is straightforward
print("Loading evaluation metrics...")
with open("artifacts/metrics/evaluation_metrics.json") as f:
    nn_metrics = json.load(f)

with open("artifacts/metrics/rf_evaluation_metrics.json") as f:
    rf_metrics = json.load(f)

zone_names   = ["Zone 1", "Zone 2", "Zone 3", "overall"]
metric_names = ["RMSE", "MAE", "R2"]

# print a clean comparison table to the console so results are easy
# to read and copy into the report
print("\n" + "=" * 70)
print(f"{'Model Comparison - Tetouan City Power Consumption':^70}")
print("=" * 70)
print(f"{'Zone':<12} {'Metric':<8} {'Feedforward NN':>16} {'Random Forest':>16} {'Winner':>10}")
print("-" * 70)

comparison = {}
for zone in zone_names:
    comparison[zone] = {}
    for metric in metric_names:
        nn_val = nn_metrics[zone][metric]
        rf_val = rf_metrics[zone][metric]

        # for RMSE and MAE lower is better, for R² higher is better
        if metric == "R2":
            winner = "RF" if rf_val > nn_val else "NN"
        else:
            winner = "RF" if rf_val < nn_val else "NN"

        comparison[zone][metric] = {
            "nn": nn_val,
            "rf": rf_val,
            "winner": winner
        }
        print(f"{zone:<12} {metric:<8} {nn_val:>16.2f} {rf_val:>16.2f} {winner:>10}")
    print("-" * 70)

print("=" * 70)

# save the full comparison as JSON for reference
with open("artifacts/metrics/model_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\nSaved: artifacts/metrics/model_comparison.json")

# plot 1 - side by side RMSE comparison bar chart per zone.
# RMSE is the most important metric for regression since it penalises
# large errors more heavily - I used this as the primary comparison metric
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
zone_labels = ["Zone 1", "Zone 2", "Zone 3"]
bar_width   = 0.35
x           = np.arange(3)

nn_rmse = [nn_metrics[z]["RMSE"] for z in zone_labels]
rf_rmse = [rf_metrics[z]["RMSE"] for z in zone_labels]
nn_mae  = [nn_metrics[z]["MAE"]  for z in zone_labels]
rf_mae  = [rf_metrics[z]["MAE"]  for z in zone_labels]
nn_r2   = [nn_metrics[z]["R2"]   for z in zone_labels]
rf_r2   = [rf_metrics[z]["R2"]   for z in zone_labels]

# RMSE
axes[0].bar(x - bar_width/2, nn_rmse, bar_width, label="Feedforward NN", color="steelblue", alpha=0.85)
axes[0].bar(x + bar_width/2, rf_rmse, bar_width, label="Random Forest",  color="seagreen",  alpha=0.85)
axes[0].set_title("RMSE per Zone (lower is better)", fontweight="bold")
axes[0].set_xlabel("Zone")
axes[0].set_ylabel("RMSE (kW)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(zone_labels)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# MAE
axes[1].bar(x - bar_width/2, nn_mae, bar_width, label="Feedforward NN", color="steelblue", alpha=0.85)
axes[1].bar(x + bar_width/2, rf_mae, bar_width, label="Random Forest",  color="seagreen",  alpha=0.85)
axes[1].set_title("MAE per Zone (lower is better)", fontweight="bold")
axes[1].set_xlabel("Zone")
axes[1].set_ylabel("MAE (kW)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(zone_labels)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# R²
axes[2].bar(x - bar_width/2, nn_r2, bar_width, label="Feedforward NN", color="steelblue", alpha=0.85)
axes[2].bar(x + bar_width/2, rf_r2, bar_width, label="Random Forest",  color="seagreen",  alpha=0.85)
axes[2].set_title("R² per Zone (higher is better)", fontweight="bold")
axes[2].set_xlabel("Zone")
axes[2].set_ylabel("R²")
axes[2].set_xticks(x)
axes[2].set_xticklabels(zone_labels)
axes[2].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="R²=0 baseline")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle("Feedforward NN vs Random Forest - Model Comparison\nTetouan City Power Consumption Dataset",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("artifacts/metrics/model_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/model_comparison_bars.png")

# plot 2 - overall metrics radar/summary chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

# build a clean table for the report
table_data = [
    ["Zone", "Model", "RMSE (kW)", "MAE (kW)", "R²"],
]
for zone in zone_names:
    table_data.append([zone, "Feedforward NN",
                       f"{nn_metrics[zone]['RMSE']:.0f}",
                       f"{nn_metrics[zone]['MAE']:.0f}",
                       f"{nn_metrics[zone]['R2']:.4f}"])
    table_data.append(["", "Random Forest",
                       f"{rf_metrics[zone]['RMSE']:.0f}",
                       f"{rf_metrics[zone]['MAE']:.0f}",
                       f"{rf_metrics[zone]['R2']:.4f}"])

table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# colour header row
for j in range(5):
    table[0, j].set_facecolor("#1E2761")
    table[0, j].set_text_props(color="white", fontweight="bold")

# alternate row colouring and highlight overall rows
for i in range(1, len(table_data)):
    for j in range(5):
        if table_data[i][0] in ["Overall", ""]:
            if table_data[i][1] in ["Feedforward NN", "Random Forest"] and \
               (i > 0 and table_data[i-1][0] == "Overall" or table_data[i][0] == "Overall"):
                table[i, j].set_facecolor("#E8F4F8")
        if i % 4 in [2, 3]:
            table[i, j].set_facecolor("#F8F9FA")

plt.title("Model Comparison Summary Table\nFeedforward NN vs Random Forest - Tetouan Dataset",
          fontweight="bold", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig("artifacts/metrics/model_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: artifacts/metrics/model_comparison_table.png")

print("\nModel comparison complete.")
print("Output files:")
print("  artifacts/metrics/model_comparison.json")
print("  artifacts/metrics/model_comparison_bars.png")
print("  artifacts/metrics/model_comparison_table.png")
