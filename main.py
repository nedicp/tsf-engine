import requests
import numpy as np

data = np.load("input_data.npy", allow_pickle=True)
y_true = np.load("y_true.npy", allow_pickle=True)

print(f"Loaded data shape: {data.shape}")
print(f"Loaded ground truth shape: {y_true.shape}")

try:
    response = requests.get("http://localhost:5000/healthz")
    if response.json()["status"] == "ok":
        print("Health check passed")
    else:
        print("Health check failed: ", response.json())
        exit(1)
except Exception as e:
    print("Health check failed: ", e)
    exit(1)

print("Testing N-BEATS-CNN model...")
response1 = requests.post("http://localhost:5000/predict/nbeats-cnn", json={"data": data.tolist()})
print(response1.json())
print("--------------------------------")

print("Testing CNN-N-BEATS model...")
response2 = requests.post("http://localhost:5000/predict/cnn-nbeats", json={"data": data.tolist()})
print(response2.json())
print("--------------------------------")

print("Testing N-BEATS model...")
response3 = requests.post("http://localhost:5000/predict/nbeats", json={"data": data.tolist()})
print(response3.json())

if not (response1.json().get("success", False) and
        response2.json().get("success", False) and
        response3.json().get("success", False)):
    print("One or more API calls failed!")
    exit(1)

mape1 = np.mean(np.abs((y_true - np.array(response1.json()["forecast"])) / (y_true + 1e-8))) * 100
mape2 = np.mean(np.abs((y_true - np.array(response2.json()["forecast"])) / (y_true + 1e-8))) * 100
mape3 = np.mean(np.abs((y_true - np.array(response3.json()["forecast"])) / (y_true + 1e-8))) * 100

print(f"MAPE N-BEATS-CNN: {mape1:.4f}%")
print(f"MAPE CNN-N-BEATS: {mape2:.4f}%")
print(f"MAPE N-BEATS: {mape3:.4f}%")

print("All tests completed successfully!")
