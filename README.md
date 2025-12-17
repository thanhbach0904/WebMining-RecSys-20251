
# WebMining-RecSys-20251

## Hybrid Movie Recommendation System

A three-layer hybrid recommendation system combining content-based filtering, collaborative filtering (SVD), and deep learning (Autoencoder) on the MovieLens-100K dataset.

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Virtual Environment Setup

1. Create a virtual environment:

**Windows (PowerShell)**
```powershell
python -m venv webminingenv
```

**Linux/macOS**
```bash
python3 -m venv webminingenv
```

2. Activate the virtual environment:

**Windows (PowerShell)**
```powershell
.\webminingenv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
webminingenv\Scripts\activate.bat
```

**Linux/macOS**
```bash
source webminingenv/bin/activate
```

3. Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

4. To deactivate the virtual environment when done:

```bash
deactivate
```

### Verifying Installation

After installation, verify that key packages are installed:

```bash
python -c "import torch; import sklearn; import surprise; print('All packages installed successfully')"
```

## Dataset

[MovieLens-100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data)

The `ml-100k` directory contains:
- `u.data`: 100,000 ratings (user_id, item_id, rating, timestamp)
- `u.item`: Movie metadata (movie_id, title, release_date, genres)
- `u.user`: User demographics (user_id, age, gender, occupation, zip_code)
- `u.genre`: Genre list

## Architecture Overview


## System Architecture

Our system operates in two primary functional layers:

1. **Candidate Retrieval Layer:** A lightweight content-based filter rapidly reduces the search space from the entire item catalog to a relevant subset of candidates (in this project is top-300) for each user.

2. **Hybrid Ranking Layer:** This layer functions as a parallel scoring engine. The generated candidates are simultaneously scored by an SVD model and a Denoising Autoencoder. These scores are then normalized and fused via an ensemble mechanism to produce the final top-N recommendation list.

### Why This Design?

- **Progressive Complexity**: Easy to debug, can evaluate each layer independently
- **Practical Performance**: Fast candidate generation with accurate final ranking
- **Interpretable**: Clear contribution from each component
- **Cold Start Handling**: Content-based layer handles new items

## Project Structure

```
WebMining-RecSys-20251/
├── ml-100k/                          # Dataset directory
├── models/                           # Saved model files
├── src/
│   ├── data/
│   │   ├── loader.py                 # Load ratings, movies, users data
│   │   └── preprocessing.py          # Train/val/test splits, user-item matrix
│   ├── models/
│   │   ├── content_based.py          # Layer 1: Inverted index recommender
│   │   ├── svd_collaborative.py      # Layer 2: SVD using Surprise library
│   │   ├── autoencoder.py            # Layer 3: PyTorch autoencoder
│   │   ├── hybrid.py                 # Ensemble combining SVD + Autoencoder layer
│   │   └── meta_learner.py           # Stacking meta-learner
│   ├── evaluation/
│   │   ├── metrics.py                # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   │   └── evaluate.py               # Evaluation pipeline
│   ├── features/
│   │   └── features.py               # Feature engineering
│   ├── utils/
│   │   ├── utils.py                  # Utility functions
│   │   └── score_normalizer.py       # Score normalization methods
│   └── visualization/
│       ├── graph.py                  # Movie similarity network visualization
│       └── embeddings.py             # t-SNE/UMAP embedding visualization
├── notebooks/
│   ├── demo.ipynb                    # Interactive demo and experiments
│   └── loader.ipynb                  # Data loading examples
├── train.py                          # Main training pipeline
├── recommend.py                      # Generate recommendations for users
├── config.py                         # Hyperparameters and configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Usage

### Training the Model

```bash
python train.py
```

### Generating Recommendations

```bash
python recommend.py
```

### Running the Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## Configuration and Hyperparameters

All hyperparameters are defined in `config.py`. Key parameters:

### Layer 1: Content-Based
- `CONTENT_TOP_K`: Number of candidates to generate (default: 300)

### Layer 2: SVD
- `SVD_N_FACTORS`: Number of latent factors (default: 50, range: 20-100)
- `SVD_N_EPOCHS`: Training epochs (default: 20, range: 10-30)
- `SVD_LR`: Learning rate (default: 0.005)
- `SVD_REG`: Regularization strength (default: 0.02)

### Layer 3: Autoencoder
- `AE_EMBEDDING_DIM`: Bottleneck dimension (default: 32, range: 16-64)
- `AE_BATCH_SIZE`: Batch size (default: 256)
- `AE_EPOCHS`: Maximum epochs (default: 100)
- `AE_LR`: Learning rate (default: 0.001)
- `AE_DROPOUT`: Dropout rate (default: 0.3)
- `AE_PATIENCE`: Early stopping patience (default: 10)
- `AE_NOISE_RATIO`: Denoising noise ratio (default: 0.2)

### Meta-Learner
- `META_LEARNER_TYPE`: Type of meta-learner - 'logistic', 'mlp', or 'gbm' (default: 'logistic')
- `META_HIDDEN_DIM`: Hidden dimension for MLP meta-learner (default: 32)

### Score Normalization
- `NORM_METHOD`: Normalization method - 'zscore', 'minmax', or 'rank_percentile' (default: 'zscore')

### Data Splitting
- `TRAIN_RATIO`: Training data ratio (default: 0.8)
- `VAL_RATIO`: Validation data ratio (default: 0.2)
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
- Modify layer sizes: [n_items -> 512 -> 128 -> 32] can be adjusted
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


