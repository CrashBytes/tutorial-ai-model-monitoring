# Enterprise AI Model Monitoring and Observability

[![Tests](https://github.com/CrashBytes/tutorial-ai-model-monitoring/workflows/Tests/badge.svg)](https://github.com/CrashBytes/tutorial-ai-model-monitoring/actions)
[![codecov](https://codecov.io/gh/CrashBytes/tutorial-ai-model-monitoring/branch/main/graph/badge.svg)](https://codecov.io/gh/CrashBytes/tutorial-ai-model-monitoring)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)

Production-grade AI model monitoring system with drift detection, performance tracking, and automated alerting for Kubernetes environments.

**Full Tutorial:** [CrashBytes - Enterprise AI Model Monitoring](https://crashbytes.com/articles/tutorial-enterprise-ai-model-monitoring-observability-production-2025)

## Features

- **Statistical Drift Detection**: PSI, KS test, Jensen-Shannon divergence
- **Real-time Performance Tracking**: Accuracy, precision, recall, F1 score
- **Prometheus Integration**: Production-ready metrics exposition
- **Custom Grafana Dashboards**: Comprehensive model observability
- **Automated Alerting**: Drift, performance degradation, data quality
- **Kubernetes Native**: Deployment, autoscaling, health checks
- **Production Ready**: Battle-tested across Fortune 500 AI platforms

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kubernetes cluster with kubectl access
- Helm 3+

### Local Development

```bash
# Clone repository
git clone https://github.com/crashbytes/crashbytes-tutorial-ai-model-monitoring.git
cd crashbytes-tutorial-ai-model-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python -m src.main

# Access service
# API: http://localhost:8000
# Metrics: http://localhost:8000/metrics
# Docs: http://localhost:8000/docs
```

### Kubernetes Deployment

```bash
# Install Prometheus & Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Deploy monitoring service
kubectl create namespace ml-monitoring
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -n ml-monitoring
kubectl logs -f -l app=ai-model-monitoring -n ml-monitoring
```

## Usage Examples

### Log Predictions

```python
import requests

# Log a prediction for drift monitoring
response = requests.post(
    "http://localhost:8000/api/v1/predictions",
    json={
        "model_name": "fraud_detection_v2",
        "model_version": "2.1.0",
        "environment": "production",
        "prediction_id": "pred-12345",
        "features": [
            {"name": "transaction_amount", "value": 127.50, "data_type": "continuous"},
            {"name": "merchant_category", "value": "grocery", "data_type": "categorical"}
        ],
        "prediction": "legitimate",
        "prediction_probability": 0.92,
        "latency_ms": 45.2
    }
)
```

### Set Reference Data

```python
import requests
import numpy as np

# Set reference distribution for drift detection
reference_data = {
    "model_name": "fraud_detection_v2",
    "feature_name": "transaction_amount",
    "data_type": "continuous",
    "values": np.random.normal(100, 25, 10000).tolist()
}

response = requests.post(
    "http://localhost:8000/api/v1/reference",
    json=reference_data
)
```

### Check Drift

```python
# Check drift for model features
response = requests.post(
    "http://localhost:8000/api/v1/drift/check",
    json={
        "model_name": "fraud_detection_v2",
        "min_samples": 100
    }
)

print(response.json()["drift_results"])
```

### Run Example Scripts

```bash
# Generate sample predictions
python examples/sample_predictions.py

# Load reference data
python examples/load_reference_data.py
```

## Architecture

```
┌─────────────────────────────────────┐
│     ML Model Services               │
│  (Inference APIs, Batch Jobs)       │
└─────────────┬───────────────────────┘
              │ Prediction Logs
              ↓
┌─────────────────────────────────────┐
│  Model Monitoring Service (FastAPI) │
│  ┌──────────┬──────────┬──────────┐ │
│  │ Metrics  │  Drift   │  Perf    │ │
│  └──────────┴──────────┴──────────┘ │
└─────────────┬───────────┬───────────┘
              │           │
    ┌─────────↓──┐  ┌─────↓──────┐
    │ Prometheus │  │  Alertmgr  │
    └─────────┬──┘  └────────────┘
              │
    ┌─────────↓──┐
    │  Grafana   │
    └────────────┘
```

## Configuration

### Environment Variables

```bash
# Application
DEBUG=false
WORKERS=4

# Drift Detection
PSI_WARNING_THRESHOLD=0.1
PSI_ALERT_THRESHOLD=0.25
DRIFT_CHECK_INTERVAL=3600

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
PAGERDUTY_INTEGRATION_KEY=your-key
```

### Drift Detection Thresholds

- **PSI < 0.1**: Stable (no drift)
- **PSI 0.1-0.25**: Moderate drift (warning)
- **PSI > 0.25**: Significant drift (alert)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/test_monitoring.py -v

# Run specific test
pytest tests/test_drift_detection.py::TestDriftDetector::test_psi_no_drift -v
```

## Monitoring & Alerts

### Grafana Dashboards

1. Port-forward Grafana: `kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80`
2. Access at http://localhost:3000 (admin/admin123)
3. Import `dashboards/model-monitoring.json`
4. View:
   - Model performance metrics
   - Drift detection trends
   - Prediction volume & latency
   - Alert history

### Alert Rules

Pre-configured alerts for:
- Significant model drift (PSI > 0.25)
- Moderate drift (PSI > 0.1)
- Model accuracy degradation
- High prediction latency

## Production Considerations

### Scaling

- **Horizontal scaling**: 3-20 pods based on traffic
- **Prediction sampling**: Monitor 10-20% for cost optimization
- **Metric aggregation**: Reduce Prometheus cardinality
- **Resource optimization**: Adjust based on prediction volume

### Security

- Encrypt prediction logs with PII
- RBAC for Grafana dashboards
- Secret management for webhooks
- TLS for service communication
- Network policies for pod isolation

### Cost Optimization

- Implement prediction sampling
- Use spot instances for non-critical workloads
- Tiered storage (hot/cold) for historical data
- Metric downsampling for long-term retention
- Batch drift calculations

## Troubleshooting

### Common Issues

**Issue**: Insufficient samples for drift calculation  
**Solution**: Increase `min_samples_for_drift` or accumulate more predictions

**Issue**: High memory usage  
**Solution**: Reduce `MAX_BUFFER_SIZE` or implement external storage (Redis)

**Issue**: Prometheus scrape failures  
**Solution**: Verify ServiceMonitor configuration and network policies

**Issue**: Drift alerts not triggering  
**Solution**: Check reference data is set and thresholds are appropriate

## Resources

- **Tutorial Blog Post**: [CrashBytes Tutorial](https://crashbytes.com/articles/tutorial-enterprise-ai-model-monitoring-observability-production-2025)
- **Documentation**: [Architecture](docs/architecture.md) | [Deployment](docs/deployment.md) | [Troubleshooting](docs/troubleshooting.md)
- **Related Tutorials**:
  - [Enterprise AI Model Lifecycle Management](https://crashbytes.com/articles/enterprise-ai-model-lifecycle-management-vp-guide-production-scale-governance-2025)
  - [Production LLM Guardrails](https://crashbytes.com/articles/tutorial-production-llm-guardrails-python-fastapi-2025)
  - [MLOps Pipeline Kubernetes](https://crashbytes.com/articles/tutorial-mlops-pipeline-kubernetes-production-deployment-2025)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Support

- **Issues**: [GitHub Issues](https://github.com/crashbytes/crashbytes-tutorial-ai-model-monitoring/issues)
- **Discussions**: [GitHub Discussions](https://github.com/crashbytes/crashbytes-tutorial-ai-model-monitoring/discussions)
- **Contact**: [LinkedIn - Michael Eakins](https://linkedin.com/in/michael-eakins)
- **Blog**: [CrashBytes](https://crashbytes.com)

## Acknowledgments

Built with production lessons learned from deploying AI monitoring systems across Fortune 500 enterprises in regulated industries including financial services, healthcare, and manufacturing.

**Key Technologies:**
- FastAPI for high-performance async API
- Prometheus for metrics collection
- Grafana for visualization
- Kubernetes for orchestration
- Python scientific stack (NumPy, SciPy, pandas)

---

**If this tutorial helped you, please star the repository!**

Made by the [CrashBytes](https://crashbytes.com) team
