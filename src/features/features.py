"""
Feature Engineering for ML-100k dataset.
Extracts user/item statistics, popularity, genre info for meta-learner.
"""

import numpy as np
import pandas as pd #type: ignore
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts engineered features for users and items.
    Features: user_mean, user_count, item_mean, item_count, item_popularity, genre_vector
    """
    
    def __init__(self, ratings_df, movies_df, users_df=None):
        """
        Initialize and precompute all features.
        
        Args:
            ratings_df: Training ratings DataFrame
            movies_df: Movies DataFrame with genre columns
            users_df: Optional user demographics DataFrame
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.users_df = users_df
        
        self.genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation',
            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
        # Precompute all statistics
        self._compute_user_stats()
        self._compute_item_stats()
        self._compute_genre_vectors()
        if users_df is not None:
            self._compute_user_demographics()
    
    def _compute_user_stats(self):
        """Compute per-user rating statistics."""
        user_groups = self.ratings_df.groupby('user_id')['rating']
        
        self.user_mean = user_groups.mean().to_dict()
        self.user_count = user_groups.count().to_dict()
        self.user_std = user_groups.std().fillna(0).to_dict()
        
        # Global stats for new users
        self.global_user_mean = self.ratings_df['rating'].mean()
        self.global_user_count = self.ratings_df.groupby('user_id').size().mean()
    
    def _compute_item_stats(self):
        """Compute per-item rating statistics and popularity."""
        item_groups = self.ratings_df.groupby('item_id')['rating']
        
        self.item_mean = item_groups.mean().to_dict()
        self.item_count = item_groups.count().to_dict()
        self.item_std = item_groups.std().fillna(0).to_dict()
        
        # Popularity: normalized rating count (0-1 range)
        max_count = max(self.item_count.values()) if self.item_count else 1
        self.item_popularity = {
            item_id: count / max_count 
            for item_id, count in self.item_count.items()
        }
        
        # Global stats for new items
        self.global_item_mean = self.ratings_df['rating'].mean()
        self.global_item_count = self.ratings_df.groupby('item_id').size().mean()
    
    def _compute_genre_vectors(self):
        """Extract genre binary vectors for all movies."""
        self.genre_vectors = {}
        for _, row in self.movies_df.iterrows():
            movie_id = row['movie_id']
            genre_vec = row[self.genre_columns].values.astype(np.float32)
            self.genre_vectors[movie_id] = genre_vec
    
    def _compute_user_demographics(self):
        """Extract user demographic features (age, gender, occupation)."""
        self.user_demographics = {}
        
        if self.users_df is None:
            return
        
        # One-hot encode occupation
        occupations = self.users_df['occupation'].unique()
        occ_to_idx = {occ: i for i, occ in enumerate(occupations)}
        
        for _, row in self.users_df.iterrows():
            user_id = row['user_id']
            
            # Normalize age to [0, 1]
            age_normalized = row['age'] / 100.0
            
            # Gender: 0 = M, 1 = F
            gender = 1.0 if row['gender'] == 'F' else 0.0
            
            # Occupation one-hot
            occ_vec = np.zeros(len(occupations), dtype=np.float32)
            occ_vec[occ_to_idx[row['occupation']]] = 1.0
            
            self.user_demographics[user_id] = {
                'age': age_normalized,
                'gender': gender,
                'occupation': occ_vec
            }
    
    def get_user_features(self, user_id):
        """
        Get feature vector for a user.
        
        Returns:
            numpy array: [user_mean, user_count_normalized, user_std]
        """
        user_mean = self.user_mean.get(user_id, self.global_user_mean)
        user_count = self.user_count.get(user_id, self.global_user_count)
        user_std = self.user_std.get(user_id, 0.0)
        
        # Normalize
        user_mean_norm = (user_mean - 1) / 4  # Scale 1-5 to 0-1
        user_count_norm = min(user_count / 200, 1.0)  # Cap at 200
        user_std_norm = user_std / 2  # Typical std is 0-2
        
        return np.array([user_mean_norm, user_count_norm, user_std_norm], dtype=np.float32)
    
    def get_item_features(self, item_id):
        """
        Get feature vector for an item.
        
        Returns:
            numpy array: [item_mean, item_count_norm, item_popularity, genre_vector]
        """
        item_mean = self.item_mean.get(item_id, self.global_item_mean)
        item_count = self.item_count.get(item_id, self.global_item_count)
        item_pop = self.item_popularity.get(item_id, 0.0)
        genre_vec = self.genre_vectors.get(item_id, np.zeros(len(self.genre_columns)))
        
        # Normalize
        item_mean_norm = (item_mean - 1) / 4
        item_count_norm = min(item_count / 200, 1.0)
        
        base_features = np.array([item_mean_norm, item_count_norm, item_pop], dtype=np.float32)
        
        return np.concatenate([base_features, genre_vec])
    
    def get_pair_features(self, user_id, item_id, svd_score=None, ae_score=None):
        """
        Get combined feature vector for a user-item pair (for meta-learner).
        
        Args:
            user_id: User ID
            item_id: Item ID
            svd_score: Normalized SVD score (optional)
            ae_score: Normalized AE score (optional)
        
        Returns:
            numpy array: Full feature vector for meta-learner
        """
        user_feats = self.get_user_features(user_id)
        item_feats = self.get_item_features(item_id)
        
        features = [user_feats, item_feats]
        
        if svd_score is not None:
            features.append(np.array([svd_score], dtype=np.float32))
        if ae_score is not None:
            features.append(np.array([ae_score], dtype=np.float32))
        
        return np.concatenate(features)
    
    def get_feature_dim(self, include_scores=True):
        """Get total feature dimension."""
        # user: 3, item: 3 + 19 genres = 22
        dim = 3 + 3 + len(self.genre_columns)
        if include_scores:
            dim += 2  # SVD and AE scores
        return dim
