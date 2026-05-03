import requests, json, os

BASE_URL = "http://localhost:8500"
RESULTS  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

health = requests.get(f"{BASE_URL}/status").json()
test_input = {
    "gpu_memory_gb": 40,
    "batch_size": 64,
    "model_params_millions": 1500,
    "queue_depth": 8
}
pred = requests.post(f"{BASE_URL}/estimate", json=test_input).json()

output = {
    "health_endpoint": "/status",
    "predict_endpoint": "/estimate",
    "port": 8500,
    "health_response": health,
    "test_input": test_input,
    "prediction": pred["prediction"]
}
with open(os.path.join(RESULTS, "step2_serving.json"), "w") as f:
    json.dump(output, f, indent=2)
print("Saved → results/step2_serving.json")
print(output)