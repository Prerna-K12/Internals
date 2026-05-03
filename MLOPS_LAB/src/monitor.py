import json, os, pandas as pd, numpy as np

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE, "logs", "predictions.jsonl")
DATA     = os.path.join(BASE, "data", "training_data.csv")
RESULTS  = os.path.join(BASE, "results")

# load training stats
train_df = pd.read_csv(DATA)
train_means = {
    "model_params_millions": train_df["model_params_millions"].mean(),
    "queue_depth":           train_df["queue_depth"].mean()
}

# load live predictions
records = []
with open(LOG_FILE) as f:
    for line in f:
        records.append(json.loads(line))

inputs = [r["input"] for r in records]
live_df = pd.DataFrame(inputs)
predictions = [r["prediction"] for r in records]

thresholds = {"model_params_millions": 500, "queue_depth": 5}
alerts = []
drift_detected = False

for feature, threshold in thresholds.items():
    live_mean  = live_df[feature].mean()
    train_mean = train_means[feature]
    shift      = abs(live_mean - train_mean)
    status     = "ALERT" if shift > threshold else "OK"
    if status == "ALERT":
        drift_detected = True
    alerts.append({
        "feature":    feature,
        "train_mean": round(train_mean, 4),
        "live_mean":  round(live_mean,  4),
        "shift":      round(shift,      4),
        "threshold":  threshold,
        "status":     status
    })
    print(f"[{status}] {feature}: train={train_mean:.2f}, "
          f"live={live_mean:.2f}, shift={shift:.2f}")

output = {
    "total_predictions": len(records),
    "mean_prediction":   round(float(np.mean(predictions)), 4),
    "drift_detected":    drift_detected,
    "alerts":            alerts
}
with open(os.path.join(RESULTS, "step3_monitoring.json"), "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved → results/step3_monitoring.json")