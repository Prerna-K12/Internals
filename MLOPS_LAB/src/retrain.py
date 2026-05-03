import pandas as pd, numpy as np, pickle, json, os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OLD = os.path.join(BASE, "data", "training_data.csv")
DATA_NEW = os.path.join(BASE, "data", "new_data.csv")
RESULTS  = os.path.join(BASE, "results")
MODEL_DIR= os.path.join(BASE, "models")

# ── load step1 to know champion model type ──────────────────────────────────
with open(os.path.join(RESULTS, "step1_tracking.json")) as f:
    step1 = json.load(f)
CHAMPION_NAME = step1["best_model"]
CHAMPION_RMSE = step1["best_metric_value"]

# ── combine data ─────────────────────────────────────────────────────────────
old_df = pd.read_csv(DATA_OLD)
new_df = pd.read_csv(DATA_NEW)
combined = pd.concat([old_df, new_df], ignore_index=True)

X = combined.drop("job_completion_min", axis=1)
y = combined["job_completion_min"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── build same model type ────────────────────────────────────────────────────
if CHAMPION_NAME == "Ridge":
    new_model = Ridge(alpha=1.0)
else:
    new_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )

new_model.fit(X_train, y_train)
preds = new_model.predict(X_test)
retrained_rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)

improvement = round(CHAMPION_RMSE - retrained_rmse, 4)
MIN_THRESHOLD = 0.5

if improvement >= MIN_THRESHOLD:
    action = "promoted"
    with open(os.path.join(MODEL_DIR, f"{CHAMPION_NAME}.pkl"), "wb") as f:
        pickle.dump(new_model, f)
    print("✅ New model PROMOTED.")
else:
    action = "kept_champion"
    print("⚠️  Champion KEPT — improvement insufficient.")

output = {
    "original_data_rows":    len(old_df),
    "new_data_rows":         len(new_df),
    "combined_data_rows":    len(combined),
    "champion_rmse":         CHAMPION_RMSE,
    "retrained_rmse":        retrained_rmse,
    "improvement":           improvement,
    "min_improvement_threshold": MIN_THRESHOLD,
    "action":                action,
    "comparison_metric":     "rmse"
}
with open(os.path.join(RESULTS, "step4_retraining.json"), "w") as f:
    json.dump(output, f, indent=2)
print("Saved → results/step4_retrain.json")