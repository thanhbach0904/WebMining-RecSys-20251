"""
Utilities for normalizing scores in ensemble learning.
Facilitates alignment of score distributions from diverse models prior to combination.
"""

import numpy as np
from scipy import stats #type: ignore


class ScoreNormalizer:
    """
    Standardizes scores from various models to a common scale.
    Supported methods: z-score standardization, min-max scaling, and rank percentile.
    """
    
    def __init__(self, method='zscore'):
        """
        Args:
            method (str): Normalization technique ('zscore', 'minmax', or 'rank_percentile').
        """
        self.method = method
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
        
        if self.method == 'zscore':
            self.model_stats[model_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores) + 1e-8  # Avoid division by zero
            }
        elif self.method == 'minmax':
            self.model_stats[model_name] = {
                'min': np.min(scores),
                'max': np.max(scores)
            }
        elif self.method == 'rank_percentile':
            # Store sorted scores for percentile lookup
            self.model_stats[model_name] = {
                'sorted_scores': np.sort(scores)
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
        
        if self.method == 'zscore':
            # Z-score then sigmoid to [0, 1]
            z_scores = (scores - stats['mean']) / stats['std']
            # Use sigmoid to map to [0, 1]
            normalized = 1 / (1 + np.exp(-z_scores))
        
        elif self.method == 'minmax':
            range_val = stats['max'] - stats['min'] + 1e-8
            normalized = (scores - stats['min']) / range_val
            normalized = np.clip(normalized, 0, 1)
        
        elif self.method == 'rank_percentile':
            # Convert to percentile rank
            sorted_scores = stats['sorted_scores']
            normalized = np.array([
                np.searchsorted(sorted_scores, s) / len(sorted_scores)
                for s in scores
            ])
        
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
    
    def __init__(self, method='zscore'):
        self.method = method
    
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
        
        if self.method == 'zscore':
            mean = np.mean(scores)
            std = np.std(scores) + 1e-8
            z_scores = (scores - mean) / std
            # Sigmoid to [0, 1]
            return 1 / (1 + np.exp(-z_scores))
        
        elif self.method == 'minmax':
            min_s = np.min(scores)
            max_s = np.max(scores)
            if max_s - min_s < 1e-8:
                return np.ones_like(scores) * 0.5
            return (scores - min_s) / (max_s - min_s)
        
        elif self.method == 'rank_percentile':
            # Rank-based percentile
            ranks = stats.rankdata(scores, method='average')
            return (ranks - 1) / (len(ranks) - 1 + 1e-8)
        
        return scores
