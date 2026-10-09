
"""Print runtime and dependency versions."""

import platform
import sys

import joblib
import mlflow
import numpy
import pandas
import sklearn


def main():
    print("Environment information")
    print("-" * 40)

    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("MLflow:", mlflow.__version__)
    print("joblib:", joblib.__version__)


if __name__ == "__main__":
    main()
