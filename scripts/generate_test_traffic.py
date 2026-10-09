import json
import time
import urllib.request

patient = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}

for i in range(30):
    request = urllib.request.Request(
        "http://127.0.0.1:8080/predict",
        data=json.dumps(patient).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)

    print(f"Request {i + 1}: {result['prediction']}")
    time.sleep(1)