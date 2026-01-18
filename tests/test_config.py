"""
Comprehensive tests for configuration management.

Tests cover:
- Default configuration values
- Environment variable loading
- Configuration validation
- Type checking for all settings
"""

import pytest
from unittest.mock import patch
import os

from src.config import Settings, settings


class TestSettingsDefaults:
    """Test default configuration values."""
    
    def test_app_name_default(self):
        """Test default application name."""
        config = Settings()
        assert config.app_name == "ai-model-monitoring"
    
    def test_app_version_default(self):
        """Test default application version."""
        config = Settings()
        assert config.app_version == "1.0.0"
    
    def test_debug_default(self):
        """Test debug mode defaults to False."""
        config = Settings()
        assert config.debug is False
    
    def test_host_default(self):
        """Test default host binding."""
        config = Settings()
        assert config.host == "0.0.0.0"
    
    def test_port_default(self):
        """Test default server port."""
        config = Settings()
        assert config.port == 8000
    
    def test_workers_default(self):
        """Test default worker count."""
        config = Settings()
        assert config.workers == 4
    
    def test_metrics_port_default(self):
        """Test default metrics port."""
        config = Settings()
        assert config.metrics_port == 8001
    
    def test_drift_check_interval_default(self):
        """Test default drift check interval."""
        config = Settings()
        assert config.drift_check_interval == 3600  # 1 hour in seconds
    
    def test_min_samples_for_drift_default(self):
        """Test default minimum samples for drift detection."""
        config = Settings()
        assert config.min_samples_for_drift == 100


class TestDriftThresholds:
    """Test drift detection threshold defaults."""
    
    def test_psi_warning_threshold(self):
        """Test PSI warning threshold default."""
        config = Settings()
        assert config.psi_warning_threshold == 0.1
    
    def test_psi_alert_threshold(self):
        """Test PSI alert threshold default."""
        config = Settings()
        assert config.psi_alert_threshold == 0.25
    
    def test_ks_test_alpha(self):
        """Test KS test alpha default."""
        config = Settings()
        assert config.ks_test_alpha == 0.05


class TestDataRetention:
    """Test data retention configuration."""
    
    def test_reference_data_window_days(self):
        """Test reference data retention window."""
        config = Settings()
        assert config.reference_data_window_days == 90
    
    def test_metrics_retention_days(self):
        """Test metrics retention period."""
        config = Settings()
        assert config.metrics_retention_days == 365


class TestAlertConfiguration:
    """Test alert configuration defaults."""
    
    def test_slack_webhook_url_optional(self):
        """Test Slack webhook URL is optional."""
        config = Settings()
        assert config.slack_webhook_url is None
    
    def test_pagerduty_integration_key_optional(self):
        """Test PagerDuty integration key is optional."""
        config = Settings()
        assert config.pagerduty_integration_key is None
    
    def test_alert_email_optional(self):
        """Test alert email is optional."""
        config = Settings()
        assert config.alert_email is None


class TestEnvironmentVariableLoading:
    """Test configuration loading from environment variables."""
    
    @patch.dict(os.environ, {"DEBUG": "true"})
    def test_debug_from_env(self):
        """Test loading debug flag from environment."""
        config = Settings()
        assert config.debug is True
    
    @patch.dict(os.environ, {"PORT": "9000"})
    def test_port_from_env(self):
        """Test loading port from environment."""
        config = Settings()
        assert config.port == 9000
    
    @patch.dict(os.environ, {"WORKERS": "8"})
    def test_workers_from_env(self):
        """Test loading worker count from environment."""
        config = Settings()
        assert config.workers == 8
    
    @patch.dict(os.environ, {"PSI_WARNING_THRESHOLD": "0.15"})
    def test_psi_warning_from_env(self):
        """Test loading PSI warning threshold from environment."""
        config = Settings()
        assert config.psi_warning_threshold == 0.15
    
    @patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"})
    def test_slack_webhook_from_env(self):
        """Test loading Slack webhook URL from environment."""
        config = Settings()
        assert config.slack_webhook_url == "https://hooks.slack.com/test"


class TestGlobalSettingsInstance:
    """Test the global settings instance."""
    
    def test_settings_instance_exists(self):
        """Test that global settings instance is created."""
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_settings_has_correct_defaults(self):
        """Test global settings has correct default values."""
        assert settings.app_name == "ai-model-monitoring"
        assert settings.port == 8000


class TestConfigurationTypes:
    """Test configuration value types."""
    
    def test_all_string_fields(self):
        """Test all string configuration fields."""
        config = Settings()
        assert isinstance(config.app_name, str)
        assert isinstance(config.app_version, str)
        assert isinstance(config.host, str)
    
    def test_all_integer_fields(self):
        """Test all integer configuration fields."""
        config = Settings()
        assert isinstance(config.port, int)
        assert isinstance(config.workers, int)
        assert isinstance(config.metrics_port, int)
        assert isinstance(config.drift_check_interval, int)
        assert isinstance(config.min_samples_for_drift, int)
        assert isinstance(config.reference_data_window_days, int)
        assert isinstance(config.metrics_retention_days, int)
    
    def test_all_float_fields(self):
        """Test all float configuration fields."""
        config = Settings()
        assert isinstance(config.psi_warning_threshold, float)
        assert isinstance(config.psi_alert_threshold, float)
        assert isinstance(config.ks_test_alpha, float)
    
    def test_all_boolean_fields(self):
        """Test all boolean configuration fields."""
        config = Settings()
        assert isinstance(config.debug, bool)


class TestThresholdValidation:
    """Test drift threshold value ranges."""
    
    def test_psi_thresholds_ordered(self):
        """Test PSI thresholds are properly ordered."""
        config = Settings()
        assert config.psi_warning_threshold < config.psi_alert_threshold
    
    def test_ks_alpha_in_valid_range(self):
        """Test KS test alpha is in valid probability range."""
        config = Settings()
        assert 0 < config.ks_test_alpha < 1
    
    def test_psi_warning_positive(self):
        """Test PSI warning threshold is positive."""
        config = Settings()
        assert config.psi_warning_threshold > 0
    
    def test_psi_alert_positive(self):
        """Test PSI alert threshold is positive."""
        config = Settings()
        assert config.psi_alert_threshold > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
