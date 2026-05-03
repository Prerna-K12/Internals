import requests, random, time

URL = "http://localhost:8500/estimate"

# 40 normal requests (training data ranges)
normal = [
    {"gpu_memory_gb": random.randint(8,80),
     "batch_size":    random.choice([8,16,32,64,128,256]),
     "model_params_millions": random.randint(10, 800),
     "queue_depth":   random.randint(1, 10)}
    for _ in range(40)
]

# 10 drifted requests (new_data ranges — large models, deep queues)
drifted = [
    {"gpu_memory_gb": random.randint(56, 80),
     "batch_size":    random.choice([128, 256]),
     "model_params_millions": random.randint(3500, 7000),
     "queue_depth":   random.randint(14, 20)}
    for _ in range(10)
]

for req in normal + drifted:
    r = requests.post(URL, json=req)
    print(r.json())
    time.sleep(0.05)

print("Done — 50 requests sent.")