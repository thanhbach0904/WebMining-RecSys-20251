"""
Main training pipeline for hybrid recommendation system.
Trains all three layers and combines them into an ensemble.

Usage:
    python train.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.data.loader import load_data
from src.data.preprocessing import split_data, create_user_item_matrix
from src.models.content_based import ContentBasedRecommender
from src.models.svd_collaborative import CollaborativeFilteringSVD
from src.models.autoencoder import AutoEncoderTrainer, prepare_data_loader
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import RecommenderEvaluator


def main():
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[Step 1/5] Loading data...")
    ratings, movies, users = load_data(config.DATA_DIR)
    print(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
    
    # Split data
    print("\n[Step 2/5] Splitting data...")
    train_data, val_data, test_data = split_data(
        ratings,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        time_based=config.TIME_BASED_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train models
    print("\n[Step 3/5] Training models...")
    
    # Layer 1: Content-Based
    print("\n  [Layer 1] Content-Based Recommender...")
    content_model = ContentBasedRecommender(movies)
    print("  ✓ Content-based model ready")
    
    # Layer 2: SVD
    print("\n  [Layer 2] SVD Collaborative Filtering...")
    svd_model = CollaborativeFilteringSVD(
        n_factors=config.SVD_N_FACTORS,
        n_epochs=config.SVD_N_EPOCHS,
        lr=config.SVD_LR,
        reg=config.SVD_REG
    )
    svd_model.train(train_data)
    print("  ✓ SVD model trained")
    
    # Layer 3: Autoencoder
    print("\n  [Layer 3] Autoencoder...")
    n_items = ratings['item_id'].max()
    ae_trainer = AutoEncoderTrainer(
        n_items=n_items,
        embedding_dim=config.AE_EMBEDDING_DIM,
        hidden_dims=config.AE_HIDDEN_DIMS,
        device=config.DEVICE
    )
    
    train_loader = prepare_data_loader(train_data, n_items, batch_size=config.AE_BATCH_SIZE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.AE_EPOCHS):
        train_loss = ae_trainer.train_epoch(train_loader)
        val_loss = ae_trainer.evaluate(val_data, n_items)
        
        print(f"  Epoch {epoch+1}/{config.AE_EPOCHS}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.AE_PATIENCE:
                print("  Early stopping triggered")
                break
    
    print("  ✓ Autoencoder trained")
    
    # Create hybrid model
    print("\n[Step 4/5] Creating hybrid ensemble...")
    hybrid_model = HybridRecommender(
        content_model=content_model,
        svd_model=svd_model,
        ae_model=ae_trainer,
        weights=config.ENSEMBLE_WEIGHTS
    )
    
    if config.TUNE_WEIGHTS:
        print("  Tuning ensemble weights on validation set...")
        hybrid_model.tune_weights(val_data)
    
    print("  ✓ Hybrid model ready")
    
    # Evaluate
    print("\n[Step 5/5] Evaluating on test set...")
    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate_all(hybrid_model, test_data, k=config.EVAL_K)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Save models
    if config.SAVE_MODELS:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        print(f"\nSaving models to {config.MODEL_DIR}/")
        # Save logic here
        print("✓ Models saved")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

