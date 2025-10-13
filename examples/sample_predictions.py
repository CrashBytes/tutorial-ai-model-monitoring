"""
Example script for logging predictions to the monitoring service.

Demonstrates how to integrate the monitoring service with your
ML model inference pipeline.
"""

import requests
import random
import time
from datetime import datetime
from typing import List, Dict, Any


class MonitoringClient:
    """Client for logging predictions to monitoring service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        features: List[Dict[str, Any]],
        prediction: Any,
        prediction_probability: float = None,
        latency_ms: float = None,
        ground_truth: Any = None
    ) -> Dict:
        """
        Log a prediction to the monitoring service.
        
        Args:
            model_name: Model identifier
            model_version: Model version
            features: List of feature dictionaries with name, value, data_type
            prediction: Model prediction output
            prediction_probability: Confidence score (0-1)
            latency_ms: Prediction latency in milliseconds
            ground_truth: Actual outcome if available
            
        Returns:
            Response from monitoring service
        """
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "environment": "production",
            "prediction_id": f"pred-{int(time.time() * 1000)}",
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "latency_ms": latency_ms or random.uniform(10, 100),
            "ground_truth": ground_truth
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/predictions",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def generate_sample_prediction():
    """Generate a sample fraud detection prediction."""
    
    # Sample feature values
    features = [
        {
            "name": "transaction_amount",
            "value": random.uniform(10, 500),
            "data_type": "continuous"
        },
        {
            "name": "transaction_hour",
            "value": random.randint(0, 23),
            "data_type": "continuous"
        },
        {
            "name": "merchant_category",
            "value": random.choice(["grocery", "restaurant", "gas", "retail"]),
            "data_type": "categorical"
        },
        {
            "name": "days_since_last_transaction",
            "value": random.randint(0, 30),
            "data_type": "continuous"
        },
        {
            "name": "transaction_count_24h",
            "value": random.randint(1, 10),
            "data_type": "continuous"
        }
    ]
    
    # Simulate prediction
    is_fraud = random.random() < 0.05  # 5% fraud rate
    confidence = random.uniform(0.85, 0.99) if not is_fraud else random.uniform(0.60, 0.95)
    
    return {
        "features": features,
        "prediction": "fraud" if is_fraud else "legitimate",
        "prediction_probability": confidence,
        "latency_ms": random.uniform(15, 75)
    }


def main():
    """Main function to demonstrate prediction logging."""
    
    # Initialize monitoring client
    client = MonitoringClient(base_url="http://localhost:8000")
    
    print("Starting prediction logging simulation...")
    print("Logging 100 predictions to monitoring service\n")
    
    # Log 100 sample predictions
    for i in range(100):
        prediction_data = generate_sample_prediction()
        
        try:
            response = client.log_prediction(
                model_name="fraud_detection_v2",
                model_version="2.1.0",
                **prediction_data
            )
            
            if (i + 1) % 10 == 0:
                print(f"Logged {i + 1} predictions - Status: {response['status']}")
                
        except Exception as e:
            print(f"Error logging prediction: {e}")
        
        # Small delay between predictions
        time.sleep(0.1)
    
    print("\n✓ Successfully logged 100 predictions")
    print("View metrics at: http://localhost:8000/metrics")


if __name__ == "__main__":
    main()
