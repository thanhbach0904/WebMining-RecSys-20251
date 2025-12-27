"""
Implements a Late-Fusion Hybrid Recommendation System.
Orchestrates parallel scoring from multiple models and combines them using normalized ensemble techniques or meta-learner stacking.
"""

import numpy as np
from src.data.loader import get_user_ratings
from src.utils.utils import get_user_rating_vector
from src.utils.score_normalizer import ScoreNormalizer, PerUserNormalizer
from config import NUM_MOVIES, NORM_METHOD


class LateFusionHybridRecommender:
    """
    Control class for the late-fusion hybrid system.
    
    Processing Pipeline:
        1. Candidate Generation: Use Content-based model to retrieve initial set (e.g., 300 items).
        2. Parallel Scoring: Evaluate candidates using SVD and Autoencoder models simultaneously.
        3. Score Normalization: Align scores using Z-score, Min-Max, or Rank-based methods.
        4. Fusion: Combine normalized scores via Meta-Learner (if available) or Weighted Average.
    """
    
    def __init__(self, content_model, svd_model, ae_model, 
                 feature_extractor=None, meta_learner=None,
                 weights=None, norm_method='zscore'):
        """
        Initialize the Hybrid Recommender.
        
        Args:
            content_model: Instance of ContentBasedRecommender.
            svd_model: Instance of CollaborativeFilteringSVD.
            ae_model: Instance of AutoEncoderTrainer.
            feature_extractor: (Optional) FeatureExtractor for meta-learning context.
            meta_learner: (Optional) Trained MetaLearner model. If None, defaults to weighted averaging.
            weights (list): Coefficients [w_svd, w_ae] for weighted averaging (default: [0.5, 0.5]).
            norm_method (str): Method for score normalization ('zscore', 'minmax', 'rank_percentile').
        """
        self.content_model = content_model
        self.svd_model = svd_model
        self.ae_model = ae_model
        self.feature_extractor = feature_extractor
        self.meta_learner = meta_learner
        
        self.weights = np.array(weights) if weights else np.array([0.5, 0.5])
        self.weights = self.weights / self.weights.sum()
        
        self.ratings_df = content_model.ratings_df
        
        # Score normalizers
        self.global_normalizer = ScoreNormalizer(method=norm_method)
        self.per_user_normalizer = PerUserNormalizer(method=norm_method)
        self.norm_method = norm_method
    
    def calibrate_normalizers(self, validation_df, sample_size=1000):
        """
        Calibrate score normalizers using the validation dataset.
        This step estimates the statistical distribution of scores for each constituent model.
        
        Args:
            validation_df (pd.DataFrame): DataFrame containing validation ratings.
            sample_size (int): Max number of samples to use for calibration efficiency.
        """
        print("Calibrating score normalizers...")
        
        svd_scores = []
        ae_scores = []
        
        # Sample user-item pairs
        users = validation_df['user_id'].unique()
        np.random.shuffle(users)
        
        count = 0
        for user_id in users:
            if count >= sample_size:
                break
            
            user_items = validation_df[validation_df['user_id'] == user_id]['item_id'].values
            user_rating_vec = get_user_rating_vector(user_id, self.ratings_df, NUM_MOVIES)
            
            for item_id in user_items[:10]:  # Sample up to 10 items per user
                if count >= sample_size:
                    break
                
                # Get raw scores
                svd_score = self.svd_model.predict(user_id, item_id)
                ae_predictions = self.ae_model.predict_ratings(user_rating_vec)
                ae_score = ae_predictions[item_id - 1] if item_id <= len(ae_predictions) else 0
                
                svd_scores.append(svd_score)
                ae_scores.append(ae_score)
                count += 1
        
        # Fit normalizers
        self.global_normalizer.fit('svd', svd_scores)
        self.global_normalizer.fit('ae', ae_scores)
        
        print(f"Calibrated on {count} user-item pairs")
        print(f"SVD score range: [{min(svd_scores):.3f}, {max(svd_scores):.3f}]")
        print(f"AE score range: [{min(ae_scores):.3f}, {max(ae_scores):.3f}]")
    
    def _get_candidate_scores(self, user_id, candidates):
        """
        Score all candidates with both SVD and AE.
        
        Returns:
            dict: {item_id: {'svd': score, 'ae': score, 'svd_norm': score, 'ae_norm': score}}
        """
        user_rating_vec = get_user_rating_vector(user_id, self.ratings_df, NUM_MOVIES)
        
        # Get AE predictions for all item
        ae_all_predictions = self.ae_model.predict_ratings(user_rating_vec)
        
        # Collect raw scores
        raw_svd = []
        raw_ae = []
        
        for item_id in candidates:
            svd_score = self.svd_model.predict(user_id, item_id)
            ae_score = ae_all_predictions[item_id - 1] if item_id <= len(ae_all_predictions) else 0
            raw_svd.append(svd_score)
            raw_ae.append(ae_score)
        
        # Normalize scores (per-user normalization within candidate set)
        norm_svd = self.per_user_normalizer.normalize(raw_svd)
        norm_ae = self.per_user_normalizer.normalize(raw_ae)
        
        scores = {}
        for i, item_id in enumerate(candidates):
            scores[item_id] = {
                'svd': raw_svd[i],
                'ae': raw_ae[i],
                'svd_norm': norm_svd[i],
                'ae_norm': norm_ae[i]
            }
        
        return scores
    
    def recommend(self, user_id, user_ratings=None, top_k=10, return_scores=False):
        """
        Generate recommendations with late-fusion scoring.
        
        Args:
            user_id: User ID
            user_ratings: Optional dict {movie_id: rating}
            top_k: Number of recommendations
            return_scores: If True, return (items, scores) tuple
        
        Returns:
            list: Top-K movie IDs (or tuple if return_scores=True)
        """
        if user_ratings is None:
            user_ratings = get_user_ratings(user_id, self.ratings_df)
        
        # Layer 1: Content-based candidate generation (300 items)
        candidates = self.content_model.recommend(user_id, top_k=300)
        
        if len(candidates) == 0:
            return ([], {}) if return_scores else []
        
        # Score all candidates with both models
        candidate_scores = self._get_candidate_scores(user_id, candidates)
        
        # Combine scores
        final_scores = {}
        
        if self.meta_learner is not None and self.feature_extractor is not None:
            # Use meta-learner for stacking
            features_list = []
            for item_id in candidates:
                scores = candidate_scores[item_id]
                pair_features = self.feature_extractor.get_pair_features(
                    user_id, item_id,
                    svd_score=scores['svd_norm'],
                    ae_score=scores['ae_norm']
                )
                features_list.append(pair_features)
            
            meta_scores = self.meta_learner.predict_proba(np.array(features_list))
            for i, item_id in enumerate(candidates):
                final_scores[item_id] = meta_scores[i]
        else:
            # Weighted average of normalized scores
            for item_id in candidates:
                scores = candidate_scores[item_id]
                final_scores[item_id] = (
                    self.weights[0] * scores['svd_norm'] +
                    self.weights[1] * scores['ae_norm']
                )
        
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in sorted_items[:top_k]]
        if return_scores:
            return top_items, {item_id: final_scores[item_id] for item_id in top_items}
        return top_items
    
    def get_training_data_for_meta(self, train_df, val_df, neg_sample_ratio=3):
        """
        Generate training data for meta-learner.
        
        Args:
            train_df: Training ratings for building models
            val_df: Validation ratings for generating labels
            neg_sample_ratio: Negative samples per positive
        
        Returns:
            X: Feature matrix
            y: Binary labels
        """
        print("Generating meta-learner training data...")
        
        X = []
        y = []
        
        users = val_df['user_id'].unique()
        
        for user_id in users:
            # Get user's validation items (positives)
            user_val = val_df[val_df['user_id'] == user_id]
            pos_items = set(user_val['item_id'].values)
            
            # Get candidates
            candidates = self.content_model.recommend(user_id, top_k=300)
            if len(candidates) == 0:
                continue
            
            # Score candidates
            scores = self._get_candidate_scores(user_id, candidates)
            # Positive examples
            for item_id in candidates:
                if item_id in pos_items:
                    features = self.feature_extractor.get_pair_features(
                        user_id, item_id,
                        svd_score=scores[item_id]['svd_norm'],
                        ae_score=scores[item_id]['ae_norm']
                    )
                    X.append(features)
                    y.append(1.0)
            # Negative examples (items not in val set)
            neg_candidates = [item_id for item_id in candidates if item_id not in pos_items]
            n_neg = min(len(neg_candidates), len(pos_items) * neg_sample_ratio)
            
            for item_id in np.random.choice(neg_candidates, n_neg, replace=False):
                features = self.feature_extractor.get_pair_features(
                    user_id, item_id,
                    svd_score=scores[item_id]['svd_norm'],
                    ae_score=scores[item_id]['ae_norm']
                )
                X.append(features)
                y.append(0.0)
        
        print(f"Generated {len(X)} training samples ({sum(y)} positives, {len(y)-sum(y)} negatives)")
        return np.array(X), np.array(y)


HybridRecommender = LateFusionHybridRecommender

