"""
Prometheus metrics definitions for AI model monitoring.

This module defines custom metrics for tracking model performance,
drift detection, and data quality. All metrics follow Prometheus
naming conventions and include comprehensive labels for filtering.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict

# Prediction metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name', 'model_version', 'environment']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name', 'model_version'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Performance metrics
model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy (0-1)',
    ['model_name', 'model_version', 'metric_window']
)

model_precision = Gauge(
    'ml_model_precision',
    'Current model precision (0-1)',
    ['model_name', 'model_version', 'metric_window']
)

model_recall = Gauge(
    'ml_model_recall',
    'Current model recall (0-1)',
    ['model_name', 'model_version', 'metric_window']
)

model_f1_score = Gauge(
    'ml_model_f1_score',
    'Current model F1 score (0-1)',
    ['model_name', 'model_version', 'metric_window']
)

# Drift detection metrics
feature_drift_score = Gauge(
    'ml_feature_drift_score',
    'Population Stability Index (PSI) for feature drift',
    ['model_name', 'feature_name', 'drift_method']
)

prediction_drift_score = Gauge(
    'ml_prediction_drift_score',
    'Drift score for model predictions',
    ['model_name', 'model_version', 'drift_method']
)

drift_alerts_total = Counter(
    'ml_drift_alerts_total',
    'Total number of drift alerts triggered',
    ['model_name', 'feature_name', 'severity']
)

# Data quality metrics
missing_features_total = Counter(
    'ml_missing_features_total',
    'Count of missing feature values',
    ['model_name', 'feature_name']
)

invalid_features_total = Counter(
    'ml_invalid_features_total',
    'Count of invalid feature values (out of expected range)',
    ['model_name', 'feature_name', 'validation_rule']
)

data_quality_score = Gauge(
    'ml_data_quality_score',
    'Overall data quality score (0-1)',
    ['model_name', 'quality_dimension']
)

# System health metrics
monitoring_service_health = Gauge(
    'ml_monitoring_service_health',
    'Health status of monitoring service (1=healthy, 0=unhealthy)',
    ['service_name']
)

drift_check_duration = Summary(
    'ml_drift_check_duration_seconds',
    'Time spent calculating drift metrics',
    ['model_name']
)

reference_data_age_days = Gauge(
    'ml_reference_data_age_days',
    'Age of reference data in days',
    ['model_name']
)


def record_prediction(
    model_name: str,
    model_version: str,
    environment: str,
    latency: float
) -> None:
    """Record a model prediction with latency tracking."""
    prediction_counter.labels(
        model_name=model_name,
        model_version=model_version,
        environment=environment
    ).inc()
    
    prediction_latency.labels(
        model_name=model_name,
        model_version=model_version
    ).observe(latency)


def record_performance_metrics(
    model_name: str,
    model_version: str,
    metric_window: str,
    metrics: Dict[str, float]
) -> None:
    """Record model performance metrics."""
    if 'accuracy' in metrics:
        model_accuracy.labels(
            model_name=model_name,
            model_version=model_version,
            metric_window=metric_window
        ).set(metrics['accuracy'])
    
    if 'precision' in metrics:
        model_precision.labels(
            model_name=model_name,
            model_version=model_version,
            metric_window=metric_window
        ).set(metrics['precision'])
    
    if 'recall' in metrics:
        model_recall.labels(
            model_name=model_name,
            model_version=model_version,
            metric_window=metric_window
        ).set(metrics['recall'])
    
    if 'f1_score' in metrics:
        model_f1_score.labels(
            model_name=model_name,
            model_version=model_version,
            metric_window=metric_window
        ).set(metrics['f1_score'])


def record_drift_metrics(
    model_name: str,
    feature_name: str,
    drift_method: str,
    drift_score: float,
    severity: str
) -> None:
    """Record feature drift metrics."""
    feature_drift_score.labels(
        model_name=model_name,
        feature_name=feature_name,
        drift_method=drift_method
    ).set(drift_score)
    
    if severity in ['warning', 'alert']:
        drift_alerts_total.labels(
            model_name=model_name,
            feature_name=feature_name,
            severity=severity
        ).inc()


def record_data_quality(
    model_name: str,
    feature_name: str,
    missing_count: int,
    invalid_count: int,
    quality_score: float
) -> None:
    """Record data quality metrics."""
    if missing_count > 0:
        missing_features_total.labels(
            model_name=model_name,
            feature_name=feature_name
        ).inc(missing_count)
    
    if invalid_count > 0:
        invalid_features_total.labels(
            model_name=model_name,
            feature_name=feature_name,
            validation_rule='range_check'
        ).inc(invalid_count)
    
    data_quality_score.labels(
        model_name=model_name,
        quality_dimension='completeness'
    ).set(quality_score)
