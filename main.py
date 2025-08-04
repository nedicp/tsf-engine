import requests
import numpy as np
import matplotlib.pyplot as plt

data = np.load("input_data.npy", allow_pickle=True)
y_true = np.load("y_true.npy", allow_pickle=True)

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

response1 = requests.post("http://localhost:5000/predict/nbeats-cnn", json={"data": data.tolist()})
print(response1.json())
print("--------------------------------")
response2 = requests.post("http://localhost:5000/predict/cnn-nbeats", json={"data": data.tolist()})
print(response2.json())
print("--------------------------------")
response3 = requests.post("http://localhost:5000/predict/nbeats", json={"data": data.tolist()})
print(response3.json())

## plot results
plt.plot(y_true, label="True")
plt.plot(response1.json()["forecast"], label="N-BEATS-CNN")
plt.plot(response2.json()["forecast"], label="CNN-N-BEATS")
plt.plot(response3.json()["forecast"], label="N-BEATS")
plt.legend()
plt.show()

## print mape
print(np.mean(np.abs(y_true - np.array(response1.json()["forecast"]))/y_true)*100)
print(np.mean(np.abs(y_true - np.array(response2.json()["forecast"]))/y_true)*100)
print(np.mean(np.abs(y_true - np.array(response3.json()["forecast"]))/y_true)*100)
