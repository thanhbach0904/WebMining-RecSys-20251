"""
Comparison Evaluation: With vs Without Content-Based Filtering.
Uses pre-trained models from models/ folder.

Usage:
    python -m evaluate_without_cbf.py
"""

import os
import time
import numpy as np
import torch
import pickle

import config
from src.data.loader import load_ratings_by_fold, load_movies_df
from src.models.content_based import ContentBasedRecommender
from src.models.svd_collaborative import CollaborativeFilteringSVD
from src.models.autoencoder import AutoEncoderTrainer
from src.models.hybrid import LateFusionHybridRecommender
from src.features.features import FeatureExtractor
from src.utils.utils import get_user_rating_vector, load_users_df
from src.utils.score_normalizer import PerUserNormalizer
from src.evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, coverage


class DirectHybridRecommender:
    """
    Hybrid recommender WITHOUT content-based filtering.
    Scores ALL items directly with SVD + Autoencoder.
    """
    
    def __init__(self, svd_model, ae_model, ratings_df, weights=None):
        self.svd_model = svd_model
        self.ae_model = ae_model
        self.ratings_df = ratings_df
        self.weights = np.array(weights) if weights else np.array([0.5, 0.5])
        self.weights = self.weights / self.weights.sum()
        self.per_user_normalizer = PerUserNormalizer()
        
        self.all_items = list(range(1, config.NUM_MOVIES + 1))
    
    def recommend(self, user_id, user_ratings=None, top_k=10, return_scores=False):
        """
        Generate recommendations by scoring ALL items directly.
        No content-based filtering layer.
        """
        user_data = self.ratings_df[self.ratings_df['user_id'] == user_id]
        rated_items = set(user_data['item_id'].values)
        
        # Candidate pool = ALL unrated items
        candidates = [item_id for item_id in self.all_items if item_id not in rated_items]
        
        if len(candidates) == 0:
            return ([], {}) if return_scores else []
        
        user_rating_vec = get_user_rating_vector(user_id, self.ratings_df, config.NUM_MOVIES)
        ae_all_predictions = self.ae_model.predict_ratings(user_rating_vec)
        
        raw_svd = []
        raw_ae = []
        
        for item_id in candidates:
            svd_score = self.svd_model.predict(user_id, item_id)
            ae_score = ae_all_predictions[item_id - 1] if item_id <= len(ae_all_predictions) else 0
            raw_svd.append(svd_score)
            raw_ae.append(ae_score)
        
        norm_svd = self.per_user_normalizer.normalize(raw_svd)
        norm_ae = self.per_user_normalizer.normalize(raw_ae)
        
        final_scores = {}
        for i, item_id in enumerate(candidates):
            final_scores[item_id] = (
                self.weights[0] * norm_svd[i] +
                self.weights[1] * norm_ae[i]
            )
        
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in sorted_items[:top_k]]
        if return_scores:
            return top_items, {item_id: final_scores[item_id] for item_id in top_items}
        return top_items


def load_fold_models(fold_num, model_dir='models'):
    """Load pre-trained SVD and AE models for a specific fold."""
    svd_path = os.path.join(model_dir, f'svd_fold{fold_num}_2812.pkl')
    with open(svd_path, 'rb') as f:
        svd_algo = pickle.load(f)
    
    svd_model = CollaborativeFilteringSVD(
        n_factors=config.SVD_N_FACTORS,
        n_epochs=config.SVD_N_EPOCHS,
        lr=config.SVD_LR,
        reg=config.SVD_REG
    )
    svd_model.model = svd_algo
    svd_model.is_trained = True
    
    ae_path = os.path.join(model_dir, f'autoencoder_fold{fold_num}_2812.pt')
    device = 'cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'
    checkpoint = torch.load(ae_path, map_location=device, weights_only=False)
    
    ae_trainer = AutoEncoderTrainer(
        n_items=checkpoint['n_items'],
        embedding_dim=config.AE_EMBEDDING_DIM,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
        lr=config.AE_LR,
        weight_decay=config.AE_WEIGHT_DECAY,
        device=device
    )
    ae_trainer.model.load_state_dict(checkpoint['model_state_dict'])
    ae_trainer.model.eval()
    
    return svd_model, ae_trainer


def load_saved_weights():
    """Load saved best weights."""
    weights_path = os.path.join(config.MODEL_DIR, 'best_weights.npy')
    if os.path.exists(weights_path):
        return np.load(weights_path).tolist()
    return [0.5, 0.5]


def evaluate_recommender(recommender, test_df, train_df, svd_model, ae_model, best_weights, k=10, label=""):
    """
    Evaluate a recommender and return metrics + timings.
    """
    users = test_df['user_id'].unique()
    
    precision_scores, recall_scores, ndcg_scores = [], [], []
    all_recommendations = []
    y_true_hybrid, y_pred_hybrid = [], []
    y_true_svd, y_pred_svd = [], []
    y_true_ae, y_pred_ae = [], []
    
    recommend_times = []
    
    for user_id in users:
        user_test_data = test_df[test_df['user_id'] == user_id]
        relevant_items = list(user_test_data['item_id'])
        relevant_with_ratings = dict(zip(user_test_data['item_id'], user_test_data['rating']))
        
        if len(relevant_items) == 0:
            continue
        
        try:
            # Time the recommendation
            start = time.time()
            recommendations = recommender.recommend(user_id, top_k=k)
            recommend_times.append(time.time() - start)
            
            if len(recommendations) == 0:
                continue
            
            all_recommendations.append(recommendations)
            
            # Ranking metrics
            precision_scores.append(precision_at_k(recommendations, relevant_items, k))
            recall_scores.append(recall_at_k(recommendations, relevant_items, k))
            ndcg_scores.append(ndcg_at_k(recommendations, relevant_with_ratings, k))
            
            # Rating prediction metrics
            user_rating_vec = get_user_rating_vector(user_id, train_df, config.NUM_MOVIES)
            ae_preds = ae_model.predict_ratings(user_rating_vec)
            
            for item_id in relevant_items:
                true_rating = relevant_with_ratings[item_id]
                
                pred_svd = svd_model.predict(user_id, item_id)
                y_true_svd.append(true_rating)
                y_pred_svd.append(pred_svd)
                
                if item_id - 1 < len(ae_preds):
                    pred_ae = ae_preds[item_id - 1] * 5.0
                    y_true_ae.append(true_rating)
                    y_pred_ae.append(pred_ae)
                    
                    pred_hybrid = best_weights[0] * pred_svd + best_weights[1] * pred_ae
                    y_true_hybrid.append(true_rating)
                    y_pred_hybrid.append(pred_hybrid)
                    
        except Exception as e:
            continue
    
    total_items = test_df['item_id'].nunique()
    
    results = {
        f'Precision@{k}': np.mean(precision_scores) if precision_scores else 0.0,
        f'Recall@{k}': np.mean(recall_scores) if recall_scores else 0.0,
        f'NDCG@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'Coverage': coverage(all_recommendations, total_items) if all_recommendations else 0.0,
        'RMSE_Hybrid': rmse(y_true_hybrid, y_pred_hybrid) if y_true_hybrid else 0.0,
        'MAE_Hybrid': mae(y_true_hybrid, y_pred_hybrid) if y_true_hybrid else 0.0,
        'RMSE_SVD': rmse(y_true_svd, y_pred_svd) if y_true_svd else 0.0,
        'RMSE_AE': rmse(y_true_ae, y_pred_ae) if y_true_ae else 0.0,
    }
    
    timings = {
        'avg_recommend_ms': np.mean(recommend_times) * 1000 if recommend_times else 0.0,
        'total_recommend_s': sum(recommend_times),
        'users_evaluated': len(precision_scores)
    }
    
    return results, timings


def run_comparison():
    """
    Compare WITH vs WITHOUT content-based filtering.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: WITH vs WITHOUT CONTENT-BASED FILTERING")
    print("=" * 70)
    
    movies_df = load_movies_df()
    users_df = load_users_df()
    best_weights = load_saved_weights()
    
    print(f"Using weights: SVD={best_weights[0]:.3f}, AE={best_weights[1]:.3f}")
    print(f"Content-based candidate pool: {config.CONTENT_TOP_K}")
    print(f"Total items: {config.NUM_MOVIES}")
    
    k = config.EVAL_K
    
    with_cb_results = []
    without_cb_results = []
    with_cb_timings = []
    without_cb_timings = []
    
    for fold_num in range(1, 6):
        print(f"\n{'─' * 50}")
        print(f"Fold {fold_num}/5")
        print(f"{'─' * 50}")
        
        train_df = load_ratings_by_fold(fold_name=f"u{fold_num}.base")
        test_df = load_ratings_by_fold(fold_name=f"u{fold_num}.test")
        
        svd_model, ae_model = load_fold_models(fold_num, config.MODEL_DIR)
        
        # === WITH Content-Based Filtering ===
        content_model = ContentBasedRecommender(movies_df, train_df)
        feature_extractor = FeatureExtractor(train_df, movies_df, users_df)
        
        hybrid_with_cb = LateFusionHybridRecommender(
            content_model=content_model,
            svd_model=svd_model,
            ae_model=ae_model,
            feature_extractor=feature_extractor,
            meta_learner=None,
            weights=best_weights
        )
        hybrid_with_cb.calibrate_normalizers(test_df, sample_size=500)
        
        print("  Evaluating WITH content-based filtering...")
        res_with, time_with = evaluate_recommender(
            hybrid_with_cb, test_df, train_df, svd_model, ae_model, best_weights, k=k
        )
        with_cb_results.append(res_with)
        with_cb_timings.append(time_with)
        
        # === WITHOUT Content-Based Filtering ===
        hybrid_without_cb = DirectHybridRecommender(
            svd_model=svd_model,
            ae_model=ae_model,
            ratings_df=train_df,
            weights=best_weights
        )
        
        print("  Evaluating WITHOUT content-based filtering...")
        res_without, time_without = evaluate_recommender(
            hybrid_without_cb, test_df, train_df, svd_model, ae_model, best_weights, k=k
        )
        without_cb_results.append(res_without)
        without_cb_timings.append(time_without)
        
        print(f"  WITH CB:    NDCG@{k}={res_with[f'NDCG@{k}']:.4f}, Time={time_with['avg_recommend_ms']:.2f}ms/user")
        print(f"  WITHOUT CB: NDCG@{k}={res_without[f'NDCG@{k}']:.4f}, Time={time_without['avg_recommend_ms']:.2f}ms/user")
    
    # === PRINT COMPARISON ===
    print("\n")
    print("=" * 70)
    print("                    COMPARISON RESULTS")
    print("=" * 70)
    
    metrics = [f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}', 'Coverage', 
               'RMSE_Hybrid', 'MAE_Hybrid', 'RMSE_SVD', 'RMSE_AE']
    
    print(f"\n{'Metric':<20} {'WITH CB':>15} {'WITHOUT CB':>15} {'Δ (%)':>12}")
    print("─" * 65)
    
    for metric in metrics:
        with_vals = [r[metric] for r in with_cb_results]
        without_vals = [r[metric] for r in without_cb_results]
        with_mean = np.mean(with_vals)
        without_mean = np.mean(without_vals)
        delta_pct = ((without_mean - with_mean) / with_mean * 100) if with_mean != 0 else 0
        print(f"{metric:<20} {with_mean:>15.4f} {without_mean:>15.4f} {delta_pct:>+11.2f}%")
    
    print("\n" + "─" * 65)
    print("TIMING COMPARISON")
    print("─" * 65)
    
    with_time = np.mean([t['avg_recommend_ms'] for t in with_cb_timings])
    without_time = np.mean([t['avg_recommend_ms'] for t in without_cb_timings])
    speedup = without_time / with_time if with_time > 0 else 0
    
    print(f"{'Avg time per user (ms)':<30} {with_time:>12.2f} {without_time:>12.2f}")
    print(f"{'Items scored per user':<30} {config.CONTENT_TOP_K:>12} {config.NUM_MOVIES:>12}")
    print(f"\nContent-based filtering provides {speedup:.2f}x speedup")
    print(f"Cost: Δ NDCG@{k} = {(np.mean([r[f'NDCG@{k}'] for r in with_cb_results]) - np.mean([r[f'NDCG@{k}'] for r in without_cb_results])):.4f}")
    
    print("\n" + "=" * 70)
    print("                    COMPARISON COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_comparison()