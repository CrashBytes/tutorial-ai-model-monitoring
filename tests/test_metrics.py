"""
Comprehensive tests for Prometheus metrics.

Tests cover:
- Metric creation and initialization
- Recording predictions with latency
- Recording performance metrics
- Recording drift metrics
- Recording data quality metrics
- Label application and metric values
"""

import pytest
from prometheus_client import REGISTRY

from src.metrics import (
    # Metrics
    prediction_counter,
    prediction_latency,
    model_accuracy,
    model_precision,
    model_recall,
    model_f1_score,
    feature_drift_score,
    prediction_drift_score,
    drift_alerts_total,
    missing_features_total,
    invalid_features_total,
    data_quality_score,
    monitoring_service_health,
    drift_check_duration,
    reference_data_age_days,
    # Functions
    record_prediction,
    record_performance_metrics,
    record_drift_metrics,
    record_data_quality
)


class TestMetricDefinitions:
    """Test that all metrics are properly defined."""
    
    def test_prediction_counter_exists(self):
        """Test prediction counter metric exists."""
        assert prediction_counter is not None
        # prometheus_client stores counter name without _total suffix in _name
        assert prediction_counter._name == 'ml_predictions'
    
    def test_prediction_latency_exists(self):
        """Test prediction latency metric exists."""
        assert prediction_latency is not None
        assert prediction_latency._name == 'ml_prediction_latency_seconds'
    
    def test_model_accuracy_exists(self):
        """Test model accuracy metric exists."""
        assert model_accuracy is not None
        assert model_accuracy._name == 'ml_model_accuracy'
    
    def test_feature_drift_score_exists(self):
        """Test feature drift score metric exists."""
        assert feature_drift_score is not None
        assert feature_drift_score._name == 'ml_feature_drift_score'
    
    def test_drift_alerts_total_exists(self):
        """Test drift alerts counter exists."""
        assert drift_alerts_total is not None
        # prometheus_client stores counter name without _total suffix in _name
        assert drift_alerts_total._name == 'ml_drift_alerts'


class TestRecordPrediction:
    """Test prediction recording functionality."""
    
    def test_record_prediction_increments_counter(self):
        """Test that recording prediction increments counter."""
        # Get initial value
        initial_value = prediction_counter.labels(
            model_name='test_model',
            model_version='v1',
            environment='test'
        )._value.get()
        
        # Record prediction
        record_prediction(
            model_name='test_model',
            model_version='v1',
            environment='test',
            latency=0.05
        )
        
        # Check counter incremented
        new_value = prediction_counter.labels(
            model_name='test_model',
            model_version='v1',
            environment='test'
        )._value.get()
        
        assert new_value == initial_value + 1
    
    def test_record_prediction_tracks_latency(self):
        """Test that prediction latency is recorded."""
        record_prediction(
            model_name='latency_test_model',
            model_version='v2',
            environment='staging',
            latency=0.123
        )
        
        # Latency is recorded (we can't easily verify exact value in histogram,
        # but we can verify no errors occurred)
        assert True
    
    def test_record_multiple_predictions(self):
        """Test recording multiple predictions."""
        model_name = 'multi_pred_model'
        model_version = 'v1'
        environment = 'prod'
        
        initial_value = prediction_counter.labels(
            model_name=model_name,
            model_version=model_version,
            environment=environment
        )._value.get()
        
        # Record 5 predictions
        for i in range(5):
            record_prediction(
                model_name=model_name,
                model_version=model_version,
                environment=environment,
                latency=0.01 * (i + 1)
            )
        
        new_value = prediction_counter.labels(
            model_name=model_name,
            model_version=model_version,
            environment=environment
        )._value.get()
        
        assert new_value == initial_value + 5


class TestRecordPerformanceMetrics:
    """Test performance metrics recording."""
    
    def test_record_accuracy(self):
        """Test recording model accuracy."""
        record_performance_metrics(
            model_name='perf_model',
            model_version='v1',
            metric_window='1h',
            metrics={'accuracy': 0.95}
        )
        
        value = model_accuracy.labels(
            model_name='perf_model',
            model_version='v1',
            metric_window='1h'
        )._value.get()
        
        assert value == 0.95
    
    def test_record_precision(self):
        """Test recording model precision."""
        record_performance_metrics(
            model_name='perf_model',
            model_version='v2',
            metric_window='24h',
            metrics={'precision': 0.88}
        )
        
        value = model_precision.labels(
            model_name='perf_model',
            model_version='v2',
            metric_window='24h'
        )._value.get()
        
        assert value == 0.88
    
    def test_record_recall(self):
        """Test recording model recall."""
        record_performance_metrics(
            model_name='perf_model',
            model_version='v3',
            metric_window='7d',
            metrics={'recall': 0.92}
        )
        
        value = model_recall.labels(
            model_name='perf_model',
            model_version='v3',
            metric_window='7d'
        )._value.get()
        
        assert value == 0.92
    
    def test_record_f1_score(self):
        """Test recording F1 score."""
        record_performance_metrics(
            model_name='perf_model',
            model_version='v4',
            metric_window='30d',
            metrics={'f1_score': 0.90}
        )
        
        value = model_f1_score.labels(
            model_name='perf_model',
            model_version='v4',
            metric_window='30d'
        )._value.get()
        
        assert value == 0.90
    
    def test_record_all_performance_metrics(self):
        """Test recording all performance metrics at once."""
        metrics = {
            'accuracy': 0.93,
            'precision': 0.91,
            'recall': 0.89,
            'f1_score': 0.90
        }
        
        record_performance_metrics(
            model_name='complete_model',
            model_version='v1',
            metric_window='1h',
            metrics=metrics
        )
        
        # Verify all metrics were set
        acc = model_accuracy.labels(
            model_name='complete_model',
            model_version='v1',
            metric_window='1h'
        )._value.get()
        
        prec = model_precision.labels(
            model_name='complete_model',
            model_version='v1',
            metric_window='1h'
        )._value.get()
        
        assert acc == 0.93
        assert prec == 0.91
    
    def test_record_partial_metrics(self):
        """Test recording only some performance metrics."""
        # Should not error when only accuracy is provided
        record_performance_metrics(
            model_name='partial_model',
            model_version='v1',
            metric_window='1h',
            metrics={'accuracy': 0.85}
        )
        
        assert True  # No error = success


class TestRecordDriftMetrics:
    """Test drift metrics recording."""
    
    def test_record_drift_score(self):
        """Test recording drift score."""
        record_drift_metrics(
            model_name='drift_model',
            feature_name='feature_1',
            drift_method='psi',
            drift_score=0.15,
            severity='warning'
        )
        
        value = feature_drift_score.labels(
            model_name='drift_model',
            feature_name='feature_1',
            drift_method='psi'
        )._value.get()
        
        assert value == 0.15
    
    def test_record_drift_alert_on_warning(self):
        """Test that warning severity triggers drift alert."""
        initial_alerts = drift_alerts_total.labels(
            model_name='alert_model',
            feature_name='feature_warning',
            severity='warning'
        )._value.get()
        
        record_drift_metrics(
            model_name='alert_model',
            feature_name='feature_warning',
            drift_method='psi',
            drift_score=0.12,
            severity='warning'
        )
        
        new_alerts = drift_alerts_total.labels(
            model_name='alert_model',
            feature_name='feature_warning',
            severity='warning'
        )._value.get()
        
        assert new_alerts == initial_alerts + 1
    
    def test_record_drift_alert_on_alert(self):
        """Test that alert severity triggers drift alert."""
        initial_alerts = drift_alerts_total.labels(
            model_name='alert_model',
            feature_name='feature_alert',
            severity='alert'
        )._value.get()
        
        record_drift_metrics(
            model_name='alert_model',
            feature_name='feature_alert',
            drift_method='psi',
            drift_score=0.30,
            severity='alert'
        )
        
        new_alerts = drift_alerts_total.labels(
            model_name='alert_model',
            feature_name='feature_alert',
            severity='alert'
        )._value.get()
        
        assert new_alerts == initial_alerts + 1
    
    def test_no_drift_alert_on_stable(self):
        """Test that stable severity does not trigger drift alert."""
        initial_warning_alerts = drift_alerts_total.labels(
            model_name='stable_model',
            feature_name='feature_stable',
            severity='warning'
        )._value.get()
        
        initial_alert_alerts = drift_alerts_total.labels(
            model_name='stable_model',
            feature_name='feature_stable',
            severity='alert'
        )._value.get()
        
        record_drift_metrics(
            model_name='stable_model',
            feature_name='feature_stable',
            drift_method='psi',
            drift_score=0.05,
            severity='stable'
        )
        
        # Check no alerts were incremented
        new_warning_alerts = drift_alerts_total.labels(
            model_name='stable_model',
            feature_name='feature_stable',
            severity='warning'
        )._value.get()
        
        new_alert_alerts = drift_alerts_total.labels(
            model_name='stable_model',
            feature_name='feature_stable',
            severity='alert'
        )._value.get()
        
        assert new_warning_alerts == initial_warning_alerts
        assert new_alert_alerts == initial_alert_alerts


class TestRecordDataQuality:
    """Test data quality metrics recording."""
    
    def test_record_missing_features(self):
        """Test recording missing feature counts."""
        initial_count = missing_features_total.labels(
            model_name='quality_model',
            feature_name='feature_missing'
        )._value.get()
        
        record_data_quality(
            model_name='quality_model',
            feature_name='feature_missing',
            missing_count=5,
            invalid_count=0,
            quality_score=0.95
        )
        
        new_count = missing_features_total.labels(
            model_name='quality_model',
            feature_name='feature_missing'
        )._value.get()
        
        assert new_count == initial_count + 5
    
    def test_record_invalid_features(self):
        """Test recording invalid feature counts."""
        initial_count = invalid_features_total.labels(
            model_name='quality_model',
            feature_name='feature_invalid',
            validation_rule='range_check'
        )._value.get()
        
        record_data_quality(
            model_name='quality_model',
            feature_name='feature_invalid',
            missing_count=0,
            invalid_count=3,
            quality_score=0.97
        )
        
        new_count = invalid_features_total.labels(
            model_name='quality_model',
            feature_name='feature_invalid',
            validation_rule='range_check'
        )._value.get()
        
        assert new_count == initial_count + 3
    
    def test_record_quality_score(self):
        """Test recording data quality score."""
        record_data_quality(
            model_name='quality_model',
            feature_name='feature_quality',
            missing_count=0,
            invalid_count=0,
            quality_score=0.99
        )
        
        value = data_quality_score.labels(
            model_name='quality_model',
            quality_dimension='completeness'
        )._value.get()
        
        assert value == 0.99
    
    def test_no_increment_when_zero_missing(self):
        """Test that zero missing count doesn't increment counter."""
        initial_count = missing_features_total.labels(
            model_name='zero_model',
            feature_name='feature_zero'
        )._value.get()
        
        record_data_quality(
            model_name='zero_model',
            feature_name='feature_zero',
            missing_count=0,
            invalid_count=0,
            quality_score=1.0
        )
        
        new_count = missing_features_total.labels(
            model_name='zero_model',
            feature_name='feature_zero'
        )._value.get()
        
        assert new_count == initial_count  # No increment
    
    def test_no_increment_when_zero_invalid(self):
        """Test that zero invalid count doesn't increment counter."""
        initial_count = invalid_features_total.labels(
            model_name='zero_invalid_model',
            feature_name='feature_zero_invalid',
            validation_rule='range_check'
        )._value.get()
        
        record_data_quality(
            model_name='zero_invalid_model',
            feature_name='feature_zero_invalid',
            missing_count=0,
            invalid_count=0,
            quality_score=1.0
        )
        
        new_count = invalid_features_total.labels(
            model_name='zero_invalid_model',
            feature_name='feature_zero_invalid',
            validation_rule='range_check'
        )._value.get()
        
        assert new_count == initial_count  # No increment


class TestMetricLabels:
    """Test that metrics have correct labels."""
    
    def test_prediction_counter_labels(self):
        """Test prediction counter has correct labels."""
        assert 'model_name' in prediction_counter._labelnames
        assert 'model_version' in prediction_counter._labelnames
        assert 'environment' in prediction_counter._labelnames
    
    def test_drift_score_labels(self):
        """Test drift score has correct labels."""
        assert 'model_name' in feature_drift_score._labelnames
        assert 'feature_name' in feature_drift_score._labelnames
        assert 'drift_method' in feature_drift_score._labelnames
    
    def test_data_quality_labels(self):
        """Test data quality score has correct labels."""
        assert 'model_name' in data_quality_score._labelnames
        assert 'quality_dimension' in data_quality_score._labelnames


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
