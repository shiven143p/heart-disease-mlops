
"""Prometheus metrics for the Heart Disease Prediction API."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests processed by the API",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "heart_disease_http_request_duration_seconds",
    "Time spent processing HTTP requests",
    ["method", "endpoint"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

PREDICTION_COUNT = Counter(
    "heart_disease_predictions_total",
    "Number of predictions by predicted class",
    ["prediction"],
)

PREDICTION_ERRORS = Counter(
    "heart_disease_prediction_errors_total",
    "Number of prediction processing errors",
)
