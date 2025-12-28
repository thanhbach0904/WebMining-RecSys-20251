"""
Final Evaluation Script for Hybrid Recommender System.
Loads pre-trained models from models/ folder and evaluates on 5 folds.

Usage:
    python -m src.evaluation.evaluate
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
from src.models.autoencoder import AutoEncoderTrainer, DenoisingAutoEncoder
from src.models.hybrid import LateFusionHybridRecommender
from src.features.features import FeatureExtractor
from src.utils.utils import get_user_rating_vector, load_users_df
from src.evaluation.metrics import (
    rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, coverage
)


def load_saved_weights():
    """Load saved best weights from models folder."""
    weights_path = os.path.join(config.MODEL_DIR, 'best_weights.npy')
    
    if os.path.exists(weights_path):
        best_weights = np.load(weights_path).tolist()
        print(f"Loaded best weights: SVD={best_weights[0]:.3f}, AE={best_weights[1]:.3f}")
    else:
        best_weights = [0.5, 0.5]
        print("No saved weights found, using default [0.5, 0.5]")
    
    return best_weights


def load_fold_models(fold_num, model_dir='models'):
    """
    Load pre-trained models for a specific fold.
    
    Args:
        fold_num: Fold number (1-5)
        model_dir: Directory where models are saved
    
    Returns:
        svd_model, ae_trainer
    """
    # Load SVD model
    svd_path = os.path.join(model_dir, f'svd_fold{fold_num}.pkl')
    if not os.path.exists(svd_path):
        raise FileNotFoundError(f"SVD model not found: {svd_path}. Run train.py first.")
    
    with open(svd_path, 'rb') as f:
        svd_algo = pickle.load(f)
    
    # Create SVD wrapper and attach loaded model
    svd_model = CollaborativeFilteringSVD(
        n_factors=config.SVD_N_FACTORS,
        n_epochs=config.SVD_N_EPOCHS,
        lr=config.SVD_LR,
        reg=config.SVD_REG
    )
    svd_model.model = svd_algo
    svd_model.is_trained = True
    
    # Load Autoencoder model
    ae_path = os.path.join(model_dir, f'autoencoder_fold{fold_num}.pt')
    if not os.path.exists(ae_path):
        raise FileNotFoundError(f"Autoencoder model not found: {ae_path}. Run train.py first.")
    
    device = 'cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'
    checkpoint = torch.load(ae_path, map_location=device)
    
    # Create AE trainer and load state
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


def evaluate_fold(hybrid, test_df, train_df, svd_model, ae_model, best_weights,  k=10):
    """
    Evaluate on a single fold with all metrics and timing.
    
    Returns:
        results dict, timings dict
    """
    users = test_df['user_id'].unique()
    
    # Metric accumulators
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    all_recommendations = []
    
    # For RMSE/MAE
    y_true_svd = []
    y_pred_svd = []
    y_true_ae = []
    y_pred_ae = []
    y_true_hybrid = []  # ADD THIS
    y_pred_hybrid = []
    # Timing accumulators
    content_times = []
    svd_times = []
    ae_times = []
    hybrid_times = []
    
    for user_id in users:
        user_test_data = test_df[test_df['user_id'] == user_id]
        relevant_items = list(user_test_data['item_id'])
        relevant_with_ratings = dict(zip(user_test_data['item_id'], user_test_data['rating']))
        
        if len(relevant_items) == 0:
            continue
        
        try:
            # Timing: Content-based
            start = time.time()
            candidates = hybrid.content_model.recommend(user_id, top_k=300)
            content_times.append(time.time() - start)
            
            if len(candidates) == 0:
                continue
            
            # Timing: SVD scoring (sample 50, extrapolate)
            start = time.time()
            for item_id in candidates[:50]:
                _ = svd_model.predict(user_id, item_id)
            svd_times.append((time.time() - start) * 6)
            
            # Timing: AE scoring
            user_rating_vec = get_user_rating_vector(user_id, train_df, config.NUM_MOVIES)
            start = time.time()
            ae_preds = ae_model.predict_ratings(user_rating_vec)
            ae_times.append(time.time() - start)
            
            # Timing: Full hybrid recommendation
            start = time.time()
            recommendations = hybrid.recommend(user_id, top_k=k)
            hybrid_times.append(time.time() - start)
            
            all_recommendations.append(recommendations)
            
            # Ranking metrics
            precision_scores.append(precision_at_k(recommendations, relevant_items, k))
            recall_scores.append(recall_at_k(recommendations, relevant_items, k))
            ndcg_scores.append(ndcg_at_k(recommendations, relevant_with_ratings, k))
            
            # Rating prediction metrics (RMSE/MAE)
            for item_id in relevant_items:
                true_rating = relevant_with_ratings[item_id]
                
                # SVD prediction
                pred_svd = svd_model.predict(user_id, item_id)
                y_true_svd.append(true_rating)
                y_pred_svd.append(pred_svd)
                
                # AE prediction (scale back to 1-5)
                if item_id - 1 < len(ae_preds):
                    pred_ae = ae_preds[item_id - 1] * 5.0
                    y_true_ae.append(true_rating)
                    y_pred_ae.append(pred_ae)

                    pred_hybrid = (best_weights[0] * pred_svd + best_weights[1] * pred_ae)
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
        'RMSE_SVD': rmse(y_true_svd, y_pred_svd) if y_true_svd else 0.0,
        'MAE_SVD': mae(y_true_svd, y_pred_svd) if y_true_svd else 0.0,
        'RMSE_AE': rmse(y_true_ae, y_pred_ae) if y_true_ae else 0.0,
        'MAE_AE': mae(y_true_ae, y_pred_ae) if y_true_ae else 0.0,
        'RMSE_Hybrid': rmse(y_true_hybrid, y_pred_hybrid) if y_true_hybrid else 0.0,
        'MAE_Hybrid': mae(y_true_hybrid, y_pred_hybrid) if y_true_hybrid else 0.0,
    }
    
    timings = {
        'content_based_ms': np.mean(content_times) * 1000 if content_times else 0.0,
        'svd_scoring_ms': np.mean(svd_times) * 1000 if svd_times else 0.0,
        'ae_scoring_ms': np.mean(ae_times) * 1000 if ae_times else 0.0,
        'hybrid_total_ms': np.mean(hybrid_times) * 1000 if hybrid_times else 0.0,
        'users_evaluated': len(precision_scores)
    }
    
    return results, timings


def run_evaluation():
    """
    Load pre-trained models and evaluate on all 5 folds.
    """
    print("\n" + "=" * 70)
    print("HYBRID RECOMMENDER - 5-FOLD EVALUATION (FROM SAVED MODELS)")
    print("=" * 70)
    
    # Check if models exist
    for fold_num in range(1, 6):
        svd_path = os.path.join(config.MODEL_DIR, f'svd_fold{fold_num}.pkl')
        ae_path = os.path.join(config.MODEL_DIR, f'autoencoder_fold{fold_num}.pt')
        if not os.path.exists(svd_path) or not os.path.exists(ae_path):
            print(f"\n ERROR: Models for fold {fold_num} not found!")
            print(f"   Expected: {svd_path}")
            print(f"   Expected: {ae_path}")
            print("\n   Please run 'python train.py' first to train and save models.")
            return None
    
    # Load weights
    best_weights = load_saved_weights()
    
    # Load static data
    movies_df = load_movies_df()
    users_df = load_users_df()
    
    all_fold_results = []
    all_fold_timings = []
    
    k = config.EVAL_K
    total_start = time.time()
    
    for fold_num in range(1, 6):
        print(f"\n{'─' * 50}")
        print(f"Evaluating Fold {fold_num}/5...")
        print(f"{'─' * 50}")
        
        train_df = load_ratings_by_fold(fold_name=f"u{fold_num}.base")
        test_df = load_ratings_by_fold(fold_name=f"u{fold_num}.test")
        print(f"  Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")
        
        # Load pre-trained models
        print(f"  Loading models...")
        svd_model, ae_model = load_fold_models(fold_num, config.MODEL_DIR)
        print(f"   Models loaded")
        
        content_model = ContentBasedRecommender(movies_df, train_df)
        
        feature_extractor = FeatureExtractor(train_df, movies_df, users_df)
        hybrid = LateFusionHybridRecommender(
            content_model=content_model,
            svd_model=svd_model,
            ae_model=ae_model,
            feature_extractor=feature_extractor,
            meta_learner=None,
            weights=best_weights,
        )
        
        # Calibrate normalizers (quick operation)
        hybrid.calibrate_normalizers(test_df, sample_size=500)
        
        # Evaluate
        results, timings = evaluate_fold(
            hybrid, test_df, train_df, svd_model, ae_model, best_weights, k=k
        )
        
        all_fold_results.append(results)
        all_fold_timings.append(timings)
        
        print(f"  Users evaluated: {timings['users_evaluated']}")
        print(f"  NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
    
    total_time = time.time() - total_start
    
    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    
    print("\n")
    print("=" * 70)
    print("                    FINAL EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n┌─ MODEL CONFIGURATION ─────────────────────────────────────────────┐")
    print(f"│  Ensemble Weights    : SVD = {best_weights[0]:.3f}, AE = {best_weights[1]:.3f}")
    print(f"│  Candidate Pool      : {config.CONTENT_TOP_K}")
    print(f"│  Evaluation K        : {k}")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ RANKING METRICS (averaged over 5 folds) ─────────────────────────┐")
    ranking_metrics = [f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}', 'Coverage']
    for metric in ranking_metrics:
        values = [r[metric] for r in all_fold_results]
        print(f"│  {metric:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ RATING PREDICTION METRICS (averaged over 5 folds) ────────────────┐")
    rating_metrics = [
        ('RMSE_Hybrid', 'RMSE (Hybrid)'), ('MAE_Hybrid', 'MAE (Hybrid)'),
        ('RMSE_SVD', 'RMSE (SVD)'), ('MAE_SVD', 'MAE (SVD)'), 
        ('RMSE_AE', 'RMSE (AE)'), ('MAE_AE', 'MAE (AE)')]
    for metric_key, metric_label in rating_metrics:
        values = [r[metric_key] for r in all_fold_results]
        print(f"│  {metric_label:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ INFERENCE TIMING (per user, averaged) ───────────────────────────┐")
    timing_labels = [
        ('content_based_ms', 'Content-Based'),
        ('svd_scoring_ms', 'SVD Scoring'),
        ('ae_scoring_ms', 'AE Scoring'),
        ('hybrid_total_ms', 'Hybrid Total')
    ]
    for key, label in timing_labels:
        values = [t[key] for t in all_fold_timings]
        print(f"│  {label:20s}: {np.mean(values):8.2f} ms")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ PER-FOLD BREAKDOWN ───────────────────────────────────────────────┐")
    print(f"│  {'Fold':<6} {'Prec@'+str(k):<10} {'Recall@'+str(k):<10} {'NDCG@'+str(k):<10} {'RMSE_SVD':<10} {'MAE_SVD':<10}")
    print(f"│  {'─'*60}")
    for i, res in enumerate(all_fold_results, 1):
        print(f"│  {i:<6} {res[f'Precision@{k}']:<10.4f} {res[f'Recall@{k}']:<10.4f} "
              f"{res[f'NDCG@{k}']:<10.4f} {res['RMSE_SVD']:<10.4f} {res['MAE_SVD']:<10.4f}")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ SUMMARY ─────────────────────────────────────────────────────────┐")
    print(f"│  Total Evaluation Time : {total_time:.1f} seconds")
    print(f"│  Device                : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    ndcg_values = [r[f'NDCG@{k}'] for r in all_fold_results]
    print(f"│  Best Fold (NDCG)      : Fold {np.argmax(ndcg_values)+1} ({max(ndcg_values):.4f})")
    print(f"│  Worst Fold (NDCG)     : Fold {np.argmin(ndcg_values)+1} ({min(ndcg_values):.4f})")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 70)
    print("                     EVALUATION COMPLETE")
    print("=" * 70 + "\n")
    
    return {
        'fold_results': all_fold_results,
        'fold_timings': all_fold_timings,
        'total_time': total_time
    }


if __name__ == "__main__":
    run_evaluation()
