"""
Configuration management for AI Model Monitoring Service.

Supports environment variable configuration with sensible defaults
for development and production deployments.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Application settings
    app_name: str = "ai-model-monitoring"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Monitoring settings
    metrics_port: int = 8001
    drift_check_interval: int = 3600  # seconds
    min_samples_for_drift: int = 100
    
    # Drift detection thresholds
    psi_warning_threshold: float = 0.1
    psi_alert_threshold: float = 0.25
    ks_test_alpha: float = 0.05
    
    # Data retention
    reference_data_window_days: int = 90
    metrics_retention_days: int = 365
    
    # Alert configuration
    slack_webhook_url: Optional[str] = None
    pagerduty_integration_key: Optional[str] = None
    alert_email: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
