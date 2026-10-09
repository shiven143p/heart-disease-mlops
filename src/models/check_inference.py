
"""Smoke test inference using the saved trained model."""

from src.models.predict import (
    load_model,
    predict_heart_disease,
)


SAMPLE_PATIENT = {
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


def main():
    model = load_model()

    result = predict_heart_disease(
        model,
        SAMPLE_PATIENT,
    )

    print("Model loaded successfully.")
    print("\nSample patient prediction:")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
