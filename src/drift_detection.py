"""
Statistical drift detection algorithms for ML model monitoring.

Implements multiple drift detection methods:
- Population Stability Index (PSI) for categorical features
- Kolmogorov-Smirnov (KS) test for continuous features
- Jensen-Shannon divergence for distribution comparison

All methods return standardized drift scores and statistical significance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Comprehensive drift detection for ML features and predictions.
    
    Maintains reference distributions and compares production data
    to detect statistical drift using multiple methods.
    """
    
    def __init__(
        self,
        psi_warning_threshold: float = 0.1,
        psi_alert_threshold: float = 0.25,
        ks_alpha: float = 0.05
    ):
        """
        Initialize drift detector with configurable thresholds.
        
        Args:
            psi_warning_threshold: PSI value triggering warning (default: 0.1)
            psi_alert_threshold: PSI value triggering alert (default: 0.25)
            ks_alpha: Significance level for KS test (default: 0.05)
        """
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_alert_threshold = psi_alert_threshold
        self.ks_alpha = ks_alpha
        
        # Reference distributions storage
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.reference_bins: Dict[str, np.ndarray] = {}
        self.feature_types: Dict[str, str] = {}
    
    def set_reference_distribution(
        self,
        feature_name: str,
        reference_data: np.ndarray,
        feature_type: str = 'continuous',
        n_bins: int = 10
    ) -> None:
        """
        Store reference distribution for a feature.
        
        Args:
            feature_name: Name of the feature
            reference_data: Reference data array
            feature_type: 'continuous' or 'categorical'
            n_bins: Number of bins for discretization (continuous features)
        """
        self.feature_types[feature_name] = feature_type
        
        if feature_type == 'continuous':
            # Create quantile-based bins for continuous features
            self.reference_bins[feature_name] = np.quantile(
                reference_data,
                np.linspace(0, 1, n_bins + 1)
            )
            # Calculate reference distribution in bins
            ref_binned, _ = np.histogram(
                reference_data,
                bins=self.reference_bins[feature_name]
            )
            self.reference_distributions[feature_name] = ref_binned / len(reference_data)
        else:
            # For categorical features, store value counts
            unique, counts = np.unique(reference_data, return_counts=True)
            self.reference_distributions[feature_name] = dict(zip(unique, counts / len(reference_data)))
        
        logger.info(f"Reference distribution set for feature '{feature_name}' (type: {feature_type})")
    
    def calculate_psi(
        self,
        feature_name: str,
        production_data: np.ndarray
    ) -> Tuple[float, str]:
        """
        Calculate Population Stability Index (PSI) for drift detection.
        
        PSI Formula:
        PSI = Σ (prod_pct - ref_pct) * ln(prod_pct / ref_pct)
        
        PSI Interpretation:
        < 0.1: No significant change (stable)
        0.1-0.25: Moderate drift (warning)
        > 0.25: Significant drift (alert)
        
        Args:
            feature_name: Name of the feature to check
            production_data: Production data array
            
        Returns:
            Tuple of (psi_score, severity_level)
        """
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference distribution for feature '{feature_name}'")
        
        feature_type = self.feature_types[feature_name]
        ref_dist = self.reference_distributions[feature_name]
        
        if feature_type == 'continuous':
            # Bin production data using reference bins
            prod_binned, _ = np.histogram(
                production_data,
                bins=self.reference_bins[feature_name]
            )
            prod_dist = prod_binned / len(production_data)
            
            # Calculate PSI with epsilon for numerical stability
            epsilon = 1e-10
            psi = np.sum(
                (prod_dist - ref_dist) * 
                np.log((prod_dist + epsilon) / (ref_dist + epsilon))
            )
        else:
            # For categorical features
            unique, counts = np.unique(production_data, return_counts=True)
            prod_dist = dict(zip(unique, counts / len(production_data)))
            
            # Calculate PSI for all categories in reference
            epsilon = 1e-10
            psi = 0.0
            for category in ref_dist.keys():
                prod_pct = prod_dist.get(category, epsilon)
                ref_pct = ref_dist[category]
                psi += (prod_pct - ref_pct) * np.log((prod_pct + epsilon) / (ref_pct + epsilon))
        
        # Determine severity
        if psi < self.psi_warning_threshold:
            severity = 'stable'
        elif psi < self.psi_alert_threshold:
            severity = 'warning'
        else:
            severity = 'alert'
        
        logger.info(f"PSI for '{feature_name}': {psi:.4f} (severity: {severity})")
        return psi, severity
    
    def calculate_ks_statistic(
        self,
        feature_name: str,
        production_data: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Calculate Kolmogorov-Smirnov test statistic for continuous features.
        
        KS test compares cumulative distributions and detects any type of
        distribution shift (location, scale, or shape).
        
        Args:
            feature_name: Name of the feature to check
            production_data: Production data array
            
        Returns:
            Tuple of (ks_statistic, p_value, is_significant)
        """
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference distribution for feature '{feature_name}'")
        
        if self.feature_types[feature_name] != 'continuous':
            logger.warning(f"KS test is designed for continuous features, but '{feature_name}' is categorical")
        
        # Reconstruct reference samples from stored distribution
        ref_samples = self._reconstruct_samples_from_bins(feature_name)
        
        # Perform two-sample KS test
        ks_statistic, p_value = stats.ks_2samp(ref_samples, production_data)
        is_significant = p_value < self.ks_alpha
        
        logger.info(
            f"KS test for '{feature_name}': statistic={ks_statistic:.4f}, "
            f"p-value={p_value:.4f}, significant={is_significant}"
        )
        
        return ks_statistic, p_value, is_significant
    
    def calculate_jensen_shannon_divergence(
        self,
        feature_name: str,
        production_data: np.ndarray
    ) -> Tuple[float, str]:
        """
        Calculate Jensen-Shannon divergence between reference and production distributions.
        
        JS divergence is a symmetric measure of distribution similarity.
        Returns value between 0 (identical) and 1 (completely different).
        
        Args:
            feature_name: Name of the feature to check
            production_data: Production data array
            
        Returns:
            Tuple of (js_divergence, severity_level)
        """
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference distribution for feature '{feature_name}'")
        
        feature_type = self.feature_types[feature_name]
        ref_dist = self.reference_distributions[feature_name]
        
        if feature_type == 'continuous':
            # Bin production data
            prod_binned, _ = np.histogram(
                production_data,
                bins=self.reference_bins[feature_name]
            )
            prod_dist = prod_binned / len(production_data)
            
            # Calculate JS divergence
            js_div = jensenshannon(ref_dist, prod_dist)
        else:
            # For categorical features
            unique, counts = np.unique(production_data, return_counts=True)
            prod_dist_dict = dict(zip(unique, counts / len(production_data)))
            
            # Align distributions
            all_categories = set(ref_dist.keys()) | set(prod_dist_dict.keys())
            ref_vec = np.array([ref_dist.get(cat, 0) for cat in all_categories])
            prod_vec = np.array([prod_dist_dict.get(cat, 0) for cat in all_categories])
            
            js_div = jensenshannon(ref_vec, prod_vec)
        
        # Determine severity
        if js_div < 0.05:
            severity = 'stable'
        elif js_div < 0.15:
            severity = 'warning'
        else:
            severity = 'alert'
        
        logger.info(f"JS divergence for '{feature_name}': {js_div:.4f} (severity: {severity})")
        return js_div, severity
    
    def detect_drift_all_methods(
        self,
        feature_name: str,
        production_data: np.ndarray
    ) -> Dict:
        """
        Run all applicable drift detection methods and return comprehensive results.
        
        Args:
            feature_name: Name of the feature to check
            production_data: Production data array
            
        Returns:
            Dictionary with drift scores, p-values, and severity assessments
        """
        results = {
            'feature_name': feature_name,
            'feature_type': self.feature_types.get(feature_name, 'unknown'),
            'sample_size': len(production_data)
        }
        
        try:
            # PSI calculation
            psi_score, psi_severity = self.calculate_psi(feature_name, production_data)
            results['psi'] = {
                'score': psi_score,
                'severity': psi_severity
            }
            
            # KS test (for continuous features)
            if self.feature_types[feature_name] == 'continuous':
                ks_stat, ks_pvalue, ks_significant = self.calculate_ks_statistic(
                    feature_name,
                    production_data
                )
                results['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'is_significant': ks_significant
                }
            
            # Jensen-Shannon divergence
            js_div, js_severity = self.calculate_jensen_shannon_divergence(
                feature_name,
                production_data
            )
            results['jensen_shannon'] = {
                'divergence': js_div,
                'severity': js_severity
            }
            
            # Overall assessment
            severities = [psi_severity, js_severity]
            if 'alert' in severities:
                results['overall_severity'] = 'alert'
            elif 'warning' in severities:
                results['overall_severity'] = 'warning'
            else:
                results['overall_severity'] = 'stable'
            
        except Exception as e:
            logger.error(f"Error detecting drift for '{feature_name}': {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def batch_detect_drift(
        self,
        feature_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Detect drift for multiple features simultaneously.
        
        Args:
            feature_data: Dictionary mapping feature names to production data arrays
            
        Returns:
            Dictionary with drift results for each feature
        """
        results = {}
        
        for feature_name, production_data in feature_data.items():
            if feature_name in self.reference_distributions:
                results[feature_name] = self.detect_drift_all_methods(
                    feature_name,
                    production_data
                )
            else:
                logger.warning(f"No reference distribution for feature '{feature_name}', skipping")
        
        return results
    
    def _reconstruct_samples_from_bins(
        self,
        feature_name: str,
        n_samples: int = 10000
    ) -> np.ndarray:
        """Reconstruct samples from binned reference distribution."""
        ref_dist = self.reference_distributions[feature_name]
        bins = self.reference_bins[feature_name]
        
        # Sample from bins according to reference distribution
        bin_indices = np.random.choice(
            len(ref_dist),
            size=n_samples,
            p=ref_dist
        )
        
        # Generate random values within each bin
        samples = np.array([
            np.random.uniform(bins[i], bins[i+1])
            for i in bin_indices
        ])
        
        return samples
