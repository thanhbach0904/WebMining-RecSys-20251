"""
WebMining-RecSys-2025: Ensemble Scoring Utilities.
Implements Z-Score standardization to harmonize multi-model outputs.
"""

import numpy as np


class ScoreNormalizer:
    """
    Standardizes prediction scores using statistical moments (mean/variance).
    Maintains model-specific distribution parameters for consistent scaling.
    """
    
    def __init__(self):
        # Keeps internal state of calibration status
        self.fitted = False
        # Dictionary storing μ and σ per model architecture
        self.model_stats = {}
    
    def fit(self, model_name, scores):
        """
        Learns the distribution parameters from a provided score set.
        
        Args:
            model_name: Unique key for the model (e.g., 'svd', 'autoencoder').
            scores: Numerical collection of raw prediction outputs.
        """
        raw_data = np.asarray(scores).ravel()
        
        # Calculate statistical moments with a small epsilon to prevent NaN
        self.model_stats[model_name] = {
            'mean': raw_data.mean(),
            'std': raw_data.std() + 1e-10
        }
        
        self.fitted = True
    
    def transform(self, model_name, scores):
        """
        Transforms raw scores into a normalized probability space [0, 1].
        """
        input_array = np.asarray(scores).ravel()
        
        # Fallback to Min-Max if model_name hasn't been calibrated yet
        if model_name not in self.model_stats:
            lower, upper = input_array.min(), input_array.max()
            return (input_array - lower) / (upper - lower + 1e-10)
        
        # Fetch pre-computed mu and sigma
        params = self.model_stats[model_name]
        
        # Execution of Z-score logic followed by Logistic Sigmoid activation
        z_score_val = (input_array - params['mean']) / params['std']
        
        # Output is mapped to [0, 1] range
        return 1.0 / (1.0 + np.exp(-z_score_val))
    
    def fit_transform(self, model_name, scores):
        """Helper to fit and scale in a single execution block."""
        self.fit(model_name, scores)
        return self.transform(model_name, scores)


class PerUserNormalizer:
    """
    User-centric normalization engine.
    Calculates relative preference weights within a localized candidate set.
    """
    
    def __init__(self):
        """Stateless initialization."""
        pass
    
    def normalize(self, scores):
        """
        Applies local standardization for a specific user's recommendation list.
        """
        data_points = np.asarray(scores).ravel()
        
        if data_points.size == 0:
            return data_points
        
        # Local distribution estimation
        mu_local = data_points.mean()
        sigma_local = data_points.std() + 1e-10
        
        # Mapping to normalized space via Sigmoid function
        z = (data_points - mu_local) / sigma_local
        
        return 1.0 / (1.0 + np.exp(-z))