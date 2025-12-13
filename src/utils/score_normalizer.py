"""
Score normalization utilities for ensemble learning.
Aligns score distributions from different models before combining.
"""

import numpy as np
from scipy import stats #type: ignore


class ScoreNormalizer:
    """
    Normalizes scores from different models to comparable ranges.
    Supports z-score, min-max, and rank percentile normalization.
    """
    
    def __init__(self, method='zscore'):
        """
        Args:
            method: 'zscore', 'minmax', or 'rank_percentile'
        """
        self.method = method
        self.fitted = False
        
        # Per-model statistics (learned from validation set)
        self.model_stats = {}
    
    def fit(self, model_name, scores):
        """
        Fit normalizer on a set of scores from a specific model.
        
        Args:
            model_name: Identifier for the model (e.g., 'svd', 'ae')
            scores: numpy array of raw scores
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
        Transform scores to normalized range.
        
        Args:
            model_name: Identifier for the model
            scores: numpy array or list of raw scores
        
        Returns:
            numpy array: Normalized scores in [0, 1] range
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
        """Fit and transform in one step."""
        self.fit(model_name, scores)
        return self.transform(model_name, scores)


class PerUserNormalizer:
    """
    Per-user score normalization.
    Normalizes scores within each user's candidate set.
    """
    
    def __init__(self, method='zscore'):
        self.method = method
    
    def normalize(self, scores):
        """
        Normalize a user's scores across their candidate items.
        
        Args:
            scores: numpy array of scores for one user's candidates
        
        Returns:
            numpy array: Normalized scores
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
