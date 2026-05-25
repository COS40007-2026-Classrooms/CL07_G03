import os
import json
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model

os.makedirs("monitoring/alerts", exist_ok=True)
os.makedirs("monitoring/logs", exist_ok=True)
os.makedirs("monitoring/reports", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)

# set up logging to write to both the console and a persistent file.
# this means every monitoring run leaves a record with timestamps so
# you can always trace back when drift was detected and what triggered it -
# which ties into the accountability principle from our Week 7 RAI content
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/drift_detection.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# drift thresholds from our Task 1 MLOps design.
# these come directly from what we covered in Seminar 9 on monitoring -
# PSI > 0.25 flags significant data drift in input features,
# KS p < 0.05 flags statistically significant concept drift,
# and RMSE going more than 10% above baseline means performance has degraded
PSI_THRESHOLD     = 0.25
KS_P_THRESHOLD    = 0.05
RMSE_DRIFT_FACTOR = 1.10


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI) - covered in Seminar 9 as one of the
    main techniques for detecting data drift in input features.

    It works by bucketing both distributions and comparing the proportion
    of values in each bucket. a high PSI score means the feature distribution
    has shifted significantly between when the model was trained and now,
    which can cause the model to perform worse on new data.

    PSI < 0.10  - no real change, all good
    PSI 0.10 to 0.25 - moderate shift, worth keeping an eye on
    PSI > 0.25  - significant drift, retraining recommended

    I floored empty buckets at 0.0001 to avoid log(0) errors
    """
    expected_counts, bin_edges = np.histogram(expected, bins=buckets)
    actual_counts, _           = np.histogram(actual, bins=bin_edges)

    expected_pct = np.where(expected_counts == 0, 0.0001,
                            expected_counts / len(expected))
    actual_pct   = np.where(actual_counts == 0, 0.0001,
                            actual_counts / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def run_ks_test(train_preds, test_preds):
    """
    Kolmogorov-Smirnov test for concept drift - also from Seminar 9.
    I used this to compare the distribution of predictions on training data
    vs test data. if the two distributions are statistically different
    (p < 0.05), it's a signal that the model's learned relationship may
    not be holding up on newer data - basically the data has changed in a
    way that the model wasn't trained to handle
    """
    statistic, p_value = stats.ks_2samp(train_preds, test_preds)
    return float(statistic), float(p_value)


logger.info("Loading model and data...")
model   = load_model("artifacts/models/model.keras")
x_train = np.load("artifacts/data/x_train.npy")
y_train = np.load("artifacts/data/y_train.npy")
x_test  = np.load("artifacts/data/x_test.npy")
y_test  = np.load("artifacts/data/y_test.npy")

# load the baseline RMSE from the evaluate stage so I can compare
# current performance against it. if the file doesn't exist yet
# I skip the RMSE check rather than crashing the whole monitoring run
baseline_rmse = None
eval_path     = "artifacts/metrics/evaluation_metrics.json"
if os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_metrics = json.load(f)
    baseline_rmse = eval_metrics.get("overall", {}).get("RMSE")
    logger.info(f"Baseline RMSE loaded: {baseline_rmse:.2f}")
else:
    logger.warning("No evaluation_metrics.json found - skipping RMSE drift check")

logger.info("Generating predictions...")
train_preds = model.predict(x_train, verbose=0)
test_preds  = model.predict(x_test,  verbose=0)

zone_names = ["Zone 1", "Zone 2", "Zone 3"]

# data drift detection using PSI on each input feature.
# I compare the training distribution (reference window) against the test
# distribution (monitoring window) to see if anything has shifted enough
# to be a problem - this is the approach recommended in Seminar 9
logger.info("Running PSI data drift detection...")
feature_names = [
    "Temperature", "Humidity", "Wind Speed",
    "general diffuse flows", "diffuse flows",
    "hour_of_day", "day_of_week", "month", "is_weekend"
]

psi_results         = {}
data_drift_detected = False

for i, feat in enumerate(feature_names):
    psi               = calculate_psi(x_train[:, i], x_test[:, i])
    psi_results[feat] = psi
    status            = "DRIFT" if psi > PSI_THRESHOLD else "OK"
    if psi > PSI_THRESHOLD:
        data_drift_detected = True
    logger.info(f"  PSI [{feat}]: {psi:.4f} [{status}]")

# concept drift detection using the KS test on prediction distributions.
# this checks whether the model is predicting differently on test data
# compared to training data - a sign the underlying relationship has shifted
logger.info("Running KS concept drift detection...")
ks_results             = {}
concept_drift_detected = False

for i, zone in enumerate(zone_names):
    ks_stat, ks_p    = run_ks_test(train_preds[:, i], test_preds[:, i])
    ks_results[zone] = {"statistic": ks_stat, "p_value": ks_p}
    status           = "DRIFT" if ks_p < KS_P_THRESHOLD else "OK"
    if ks_p < KS_P_THRESHOLD:
        concept_drift_detected = True
    logger.info(f"  KS [{zone}]: stat={ks_stat:.4f}, p={ks_p:.4f} [{status}]")

# performance monitoring - check current RMSE against the baseline.
# if it's gone up by more than 10% that's a meaningful enough drop to
# warrant retraining rather than just normal variance between runs
logger.info("Running performance monitoring...")
current_rmse = float(np.sqrt(mean_squared_error(
    y_test.flatten(), test_preds.flatten()
)))
logger.info(f"Current RMSE: {current_rmse:.2f}")

perf_degraded = False
rmse_ratio    = None
if baseline_rmse:
    rmse_ratio    = current_rmse / baseline_rmse
    perf_degraded = rmse_ratio > RMSE_DRIFT_FACTOR
    logger.info(f"RMSE ratio vs baseline: {rmse_ratio:.4f} "
                f"({'DEGRADED' if perf_degraded else 'OK'})")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

report = {
    "timestamp":              timestamp,
    "data_drift_detected":    data_drift_detected,
    "concept_drift_detected": concept_drift_detected,
    "performance_degraded":   perf_degraded,
    "retraining_recommended": data_drift_detected or concept_drift_detected or perf_degraded,
    "psi_threshold":          PSI_THRESHOLD,
    "ks_p_threshold":         KS_P_THRESHOLD,
    "rmse_drift_factor":      RMSE_DRIFT_FACTOR,
    "current_rmse":           current_rmse,
    "baseline_rmse":          baseline_rmse,
    "rmse_ratio":             rmse_ratio,
    "psi_results":            psi_results,
    "ks_results":             ks_results
}

with open("reports/drift_report.json", "w") as f:
    json.dump(report, f, indent=2)
logger.info("Saved: reports/drift_report.json")

with open("artifacts/metrics/monitoring_metrics.json", "w") as f:
    json.dump({
        "timestamp":              timestamp,
        "current_rmse":           current_rmse,
        "baseline_rmse":          baseline_rmse,
        "data_drift":             data_drift_detected,
        "concept_drift":          concept_drift_detected,
        "retraining_recommended": report["retraining_recommended"]
    }, f, indent=2)

# write an alert file if anything triggered a retraining recommendation.
# this gives a clear written record of what caused the alert and when,
# which ties into the accountability and transparency principles from Week 7
if report["retraining_recommended"]:
    alert_file = f"monitoring/alerts/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    reasons    = []
    if data_drift_detected:
        reasons.append("Data drift detected (PSI > 0.25 on one or more features)")
    if concept_drift_detected:
        reasons.append("Concept drift detected (KS test p < 0.05 on one or more zones)")
    if perf_degraded:
        reasons.append(f"Performance degraded (RMSE ratio {rmse_ratio:.4f} > {RMSE_DRIFT_FACTOR})")

    with open(alert_file, "w") as f:
        f.write(f"RETRAINING ALERT - {timestamp}\n\n")
        f.write("Reasons:\n")
        for reason in reasons:
            f.write(f"  - {reason}\n")
        f.write(f"\nCurrent RMSE : {current_rmse:.2f}\n")
        f.write(f"Baseline RMSE: {baseline_rmse}\n")
    logger.warning(f"Alert written to {alert_file}")

# generate the HTML monitoring dashboard.
# I wanted a visual summary that anyone on the team can open in a browser
# to quickly check the health of the model without having to read through logs
logger.info("Generating HTML monitoring dashboard...")

status_colour = "#e74c3c" if report["retraining_recommended"] else "#27ae60"
status_text   = "RETRAINING RECOMMENDED" if report["retraining_recommended"] else "ALL SYSTEMS NORMAL"

psi_rows = "".join(
    f"<tr><td>{feat}</td><td>{psi:.4f}</td>"
    f"<td style='color:{'#e74c3c' if psi > PSI_THRESHOLD else '#27ae60'};font-weight:bold'>"
    f"{'DRIFT' if psi > PSI_THRESHOLD else 'OK'}</td></tr>"
    for feat, psi in psi_results.items()
)

ks_rows = "".join(
    f"<tr><td>{zone}</td><td>{v['statistic']:.4f}</td><td>{v['p_value']:.4f}</td>"
    f"<td style='color:{'#e74c3c' if v['p_value'] < KS_P_THRESHOLD else '#27ae60'};font-weight:bold'>"
    f"{'DRIFT' if v['p_value'] < KS_P_THRESHOLD else 'OK'}</td></tr>"
    for zone, v in ks_results.items()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CL07_G03 - MLOps Monitoring Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; color: #333; }}
  h1   {{ color: #2c3e50; }}
  h2   {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 6px; }}
  .status-banner {{ background: {status_colour}; color: white; padding: 16px 24px;
                    border-radius: 6px; font-size: 1.3em; font-weight: bold;
                    margin-bottom: 30px; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 6px; overflow: hidden; margin-bottom: 30px; }}
  th    {{ background: #2c3e50; color: white; padding: 10px 14px; text-align: left; }}
  td    {{ padding: 9px 14px; border-bottom: 1px solid #ecf0f1; }}
  tr:last-child td {{ border-bottom: none; }}
  .metric-grid {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
  .metric-card {{ background: white; border-radius: 6px; padding: 20px 28px;
                  min-width: 180px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
  .metric-card .label {{ color: #7f8c8d; font-size: 0.9em; margin-top: 4px; }}
</style>
</head>
<body>
<h1>CL07_G03 - MLOps Monitoring Dashboard</h1>
<p>Tetouan City Power Consumption Dataset &nbsp;|&nbsp; Last updated: {timestamp}</p>

<div class="status-banner">{status_text}</div>

<h2>Performance Metrics</h2>
<div class="metric-grid">
  <div class="metric-card">
    <div class="value">{current_rmse:.0f}</div>
    <div class="label">Current RMSE (kW)</div>
  </div>
  <div class="metric-card">
    <div class="value">{f"{baseline_rmse:.0f}" if baseline_rmse else "N/A"}</div>
    <div class="label">Baseline RMSE (kW)</div>
  </div>
  <div class="metric-card">
    <div class="value">{"%.4f" % rmse_ratio if rmse_ratio else "N/A"}</div>
    <div class="label">RMSE Ratio vs Baseline</div>
  </div>
</div>

<h2>Data Drift - PSI Results (threshold: {PSI_THRESHOLD})</h2>
<table>
  <tr><th>Feature</th><th>PSI Score</th><th>Status</th></tr>
  {psi_rows}
</table>

<h2>Concept Drift - KS Test Results (threshold: p &lt; {KS_P_THRESHOLD})</h2>
<table>
  <tr><th>Zone</th><th>KS Statistic</th><th>p-value</th><th>Status</th></tr>
  {ks_rows}
</table>

<h2>Summary</h2>
<table>
  <tr><th>Check</th><th>Result</th></tr>
  <tr><td>Data Drift (PSI)</td>
      <td style="color:{'#e74c3c' if data_drift_detected else '#27ae60'};font-weight:bold">
      {'DETECTED' if data_drift_detected else 'NOT DETECTED'}</td></tr>
  <tr><td>Concept Drift (KS Test)</td>
      <td style="color:{'#e74c3c' if concept_drift_detected else '#27ae60'};font-weight:bold">
      {'DETECTED' if concept_drift_detected else 'NOT DETECTED'}</td></tr>
  <tr><td>Performance Degradation</td>
      <td style="color:{'#e74c3c' if perf_degraded else '#27ae60'};font-weight:bold">
      {'DETECTED' if perf_degraded else 'NOT DETECTED'}</td></tr>
  <tr><td>Retraining Recommended</td>
      <td style="color:{'#e74c3c' if report['retraining_recommended'] else '#27ae60'};font-weight:bold">
      {'YES' if report['retraining_recommended'] else 'NO'}</td></tr>
</table>
</body>
</html>"""

with open("reports/monitoring_dashboard.html", "w") as f:
    f.write(html)
logger.info("Saved: reports/monitoring_dashboard.html")

logger.info("Monitoring run complete.")
logger.info(f"  Data drift detected   : {data_drift_detected}")
logger.info(f"  Concept drift detected: {concept_drift_detected}")
logger.info(f"  Performance degraded  : {perf_degraded}")
logger.info(f"  Retraining recommended: {report['retraining_recommended']}")