"""
Utilities for normalizing scores in ensemble learning.
Facilitates alignment of score distributions from diverse models prior to combination.
"""

import numpy as np
from scipy import stats #type: ignore


class ScoreNormalizer:
    """
    Standardizes scores from various models to a common scale.
    Method: z-score standardization.
    """
    
    def __init__(self):
        self.fitted = False
        # Per-model statistics (learned from validation set)
        self.model_stats = {}
    
    def fit(self, model_name, scores):
        """
        Calibrate the normalizer using a set of scores from a specific model.
        
        Args:
            model_name (str): Unique identifier for the model (e.g., 'svd', 'ae').
            scores (np.array): Array of raw scores produced by the model.
        """
        scores = np.array(scores).flatten()
        
        self.model_stats[model_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores) + 1e-8  # Avoid division by zero
        }
        
        self.fitted = True
    
    def transform(self, model_name, scores):
        """
        Apply normalization to the provided scores.
        
        Args:
            model_name (str): Identifier for the model.
            scores (np.array/list): Raw scores to be normalized.
        
        Returns:
            np.array: Normalized scores scaled to the [0, 1] range.
        """
        scores = np.array(scores).flatten()
        
        if model_name not in self.model_stats:
            # If not fitted, return raw scores scaled to [0, 1]
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        stats = self.model_stats[model_name]
        
        # Z-score then sigmoid to [0, 1]
        z_scores = (scores - stats['mean']) / stats['std']
        # Use sigmoid to map to [0, 1]
        normalized = 1 / (1 + np.exp(-z_scores))
        
        return normalized
    
    def fit_transform(self, model_name, scores):
        """Fit the normalizer and transform the scores in a single operation."""
        self.fit(model_name, scores)
        return self.transform(model_name, scores)


class PerUserNormalizer:
    """
    Normalization for individual user scores.
    Adjusts scores relative to the distribution of scores within each user's candidate set.
    """
    
    def __init__(self):
        pass
    
    def normalize(self, scores):
        """
        Normalize scores for a user's candidate items.
        
        Args:
            scores (np.array): Scores corresponding to a single user's candidates.
        
        Returns:
            np.array: The normalized scores.
        """
        scores = np.array(scores).flatten()
        
        if len(scores) == 0:
            return scores
        
        mean = np.mean(scores)
        std = np.std(scores) + 1e-8
        z_scores = (scores - mean) / std
        # Sigmoid to [0, 1]
        return 1 / (1 + np.exp(-z_scores))
