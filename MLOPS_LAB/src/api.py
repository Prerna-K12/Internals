from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle, json, os, uvicorn, datetime
import pandas as pd

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
RESULTS   = os.path.join(BASE, "results")
LOGS_DIR  = os.path.join(BASE, "logs")
LOG_FILE  = os.path.join(LOGS_DIR, "predictions.jsonl")

# create logs folder if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# load best model
with open(os.path.join(RESULTS, "step1_tracking.json")) as f:
    step1 = json.load(f)
BEST_MODEL_NAME = step1["best_model"]

with open(os.path.join(MODEL_DIR, f"{BEST_MODEL_NAME}.pkl"), "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class JobFeatures(BaseModel):
    gpu_memory_gb:         float = Field(..., ge=8,  le=80)
    batch_size:            float = Field(..., ge=8,  le=256)
    model_params_millions: float = Field(..., ge=10, le=7000)
    queue_depth:           float = Field(..., ge=1,  le=20)

@app.get("/status")
def status():
    return {"status": "running", "model": BEST_MODEL_NAME, "version": "1.0"}

@app.post("/estimate")
def estimate(job: JobFeatures):
    X = pd.DataFrame([job.dict()])
    pred = float(model.predict(X)[0])

    # ── log every prediction ──────────────────────────────────────────
    log_entry = {
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "input":      job.dict(),
        "prediction": round(pred, 4),
        "endpoint":   "/estimate"
    }
    with open(LOG_FILE, "a") as lf:
        lf.write(json.dumps(log_entry) + "\n")
    # ──────────────────────────────────────────────────────────────────

    return {"prediction": round(pred, 4)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8500, reload=False)