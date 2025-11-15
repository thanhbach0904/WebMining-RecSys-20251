# WebMining-RecSys-20251

## Hybrid Movie Recommendation System

A three-layer hybrid recommendation system combining content-based filtering, collaborative filtering (SVD), and deep learning (Autoencoder) on the MovieLens-100K dataset.

## Dataset

[MovieLens-100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data)

The `ml-100k` directory contains:
- `u.data`: 100,000 ratings (user_id, item_id, rating, timestamp)
- `u.item`: Movie metadata (movie_id, title, release_date, genres)
- `u.user`: User demographics (user_id, age, gender, occupation, zip_code)
- `u.genre`: Genre list

## Architecture Overview

```
Layer 1 (Content-Based + Inverted Index)
    ↓ Generates top 100 candidates
Layer 2 (SVD Collaborative Filtering)
    ↓ Re-ranks to top 50
Layer 3 (Autoencoder Embeddings)
    ↓ Final refinement to top 30
Weighted Ensemble → Top-K Recommendations
```

### Why This Design?

- **Progressive Complexity**: Easy to debug, can evaluate each layer independently
- **Practical Performance**: Fast candidate generation with accurate final ranking
- **Interpretable**: Clear contribution from each component
- **Cold Start Handling**: Content-based layer handles new items

## Project Structure

```
WebMining-RecSys-20251/
├── ml-100k/                          # Dataset directory
├── src/
│   ├── data/
│   │   ├── loader.py                 # Load ratings, movies, users data
│   │   └── preprocessing.py          # Train/val/test splits, user-item matrix
│   ├── models/
│   │   ├── content_based.py          # Layer 1: Inverted index recommender
│   │   ├── svd_collaborative.py      # Layer 2: SVD using Surprise library
│   │   ├── autoencoder.py            # Layer 3: PyTorch autoencoder
│   │   └── hybrid.py                 # Ensemble combining all three layers
│   ├── evaluation/
│   │   └── metrics.py                # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   └── visualization/
│       ├── graph.py                  # Movie similarity network visualization
│       └── embeddings.py             # t-SNE/UMAP embedding visualization
├── notebooks/
│   └── demo.ipynb                    # Interactive demo and experiments
├── train.py                          # Main training pipeline
├── recommend.py                      # Generate recommendations for users
├── config.py                         # Hyperparameters and configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Configuration and Hyperparameters

All hyperparameters are defined in `config.py`. Key parameters:

### Layer 1: Content-Based
- `CONTENT_TOP_K`: Number of candidates to generate (default: 100)
- `MIN_RATING_THRESHOLD`: Minimum rating to consider as preference (default: 4)

### Layer 2: SVD
- `SVD_N_FACTORS`: Number of latent factors (default: 50, range: 20-100)
- `SVD_N_EPOCHS`: Training epochs (default: 20, range: 10-30)
- `SVD_LR`: Learning rate (default: 0.005)
- `SVD_REG`: Regularization strength (default: 0.02)

### Layer 3: Autoencoder
- `AE_EMBEDDING_DIM`: Bottleneck dimension (default: 32, range: 16-64)
- `AE_BATCH_SIZE`: Batch size (default: 256)
- `AE_EPOCHS`: Maximum epochs (default: 50)
- `AE_LR`: Learning rate (default: 0.001)
- `AE_DROPOUT`: Dropout rate (default: 0.2)

### Ensemble
- `ENSEMBLE_WEIGHTS`: Initial weights [content, svd, ae] (default: [0.3, 0.5, 0.2])
- `TUNE_WEIGHTS`: Whether to optimize weights on validation set (default: True)

### Data Splitting
- `TRAIN_RATIO`: Training data ratio (default: 0.8)
- `VAL_RATIO`: Validation data ratio (default: 0.1)
- `TEST_RATIO`: Test data ratio (default: 0.1)
- `TIME_BASED_SPLIT`: Use temporal splitting (default: True)

## Customization Guide

### To Change Hyperparameters
Edit `config.py` and modify the relevant parameters.

### To Modify Model Architectures

**Content-Based (`src/models/content_based.py`)**:
- Change genre weight aggregation in `get_user_preferred_genres()`
- Modify candidate scoring logic in `recommend()`

**SVD (`src/models/svd_collaborative.py`)**:
- Adjust SVD parameters in the `__init__()` method
- Modify the Surprise library model (can swap SVD with SVD++, NMF, etc.)

**Autoencoder (`src/models/autoencoder.py`)**:
- Change network architecture in `AutoEncoderRecommender.__init__()`
- Modify layer sizes: [n_items → 512 → 128 → 32] can be adjusted
- Change activation functions (currently ReLU)
- Adjust dropout rates

**Hybrid Ensemble (`src/models/hybrid.py`)**:
- Change pipeline flow in `recommend()` (e.g., change top-K at each layer)
- Modify weight optimization in `tune_weights()`
- Add new models to the ensemble

### To Add New Evaluation Metrics
Add methods to `RecommenderEvaluator` class in `src/evaluation/metrics.py`.

### To Add Visualizations
Create new visualization functions in `src/visualization/` directory.


## Extensions

Possible improvements:
- Add user demographics as features
- Implement temporal weighting (recent ratings matter more)
- Add diversity re-ranking (MMR algorithm)
- Use attention mechanism in autoencoder
- Implement Neural Collaborative Filtering
- Add movie plot text embeddings (TF-IDF or BERT)


