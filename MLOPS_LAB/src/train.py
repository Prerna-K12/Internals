
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json, os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, "data", "training_data.csv")
MODEL_DIR   = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop("job_completion_min", axis=1)
y = df["job_completion_min"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── helper ──────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": round(mae,4), "rmse": round(rmse,4),
            "r2": round(r2,4), "mape": round(mape,4)}

# ── MLflow setup ────────────────────────────────────────────────────────────
mlflow.set_experiment("gpuforge-job-completion")

models_cfg = [
    ("Ridge", Ridge(alpha=1.0), {"alpha": 1.0}),
    ("GradientBoosting",
     GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                               max_depth=3, random_state=42),
     {"n_estimators": 100, "learning_rate": 0.1,
      "max_depth": 3, "random_state": 42}),
]

results = []
for name, model, params in models_cfg:
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("experiment_type", "baseline_comparison")
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        metrics = calc_metrics(y_test, preds)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=name)

        # also save locally
        import pickle
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        results.append({"name": name, **metrics})
        print(f"{name}: {metrics}")

# ── pick best by RMSE ────────────────────────────────────────────────────────
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "gpuforge-job-completion",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open(os.path.join(RESULTS_DIR, "step1_tracking.json"), "w") as f:
    json.dump(output, f, indent=2)

print("\nBest model:", best["name"])
print("Saved → results/step1_tracking.json")