"""
Configuration file for hybrid recommendation system.
Modify hyperparameters here to tune model performance.
"""

# Data paths
DATA_DIR = 'ml-100k'
RATINGS_FILE = 'u.data'
MOVIES_FILE = 'u.item'
USERS_FILE = 'u.user'

# Info
NUM_USERS = 943
NUM_MOVIES = 1682
NUM_RATINGS = 100000

# Data splitting
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
TIME_BASED_SPLIT = True  # If True, split by timestamp; else random split
RANDOM_SEED = 42

# Layer 1: Content-Based Recommender
CONTENT_TOP_K = 100  # Number of candidates to generate
MIN_RATING_THRESHOLD = 4  # Minimum rating to consider as user preference

# Layer 2: SVD Collaborative Filtering
SVD_N_FACTORS = 50  # Number of latent factors (20-100)
SVD_N_EPOCHS = 20  # Training epochs (10-30)
SVD_LR = 0.005  # Learning rate
SVD_REG = 0.02  # Regularization strength
SVD_TOP_K = 50  # Number of candidates to keep after SVD re-ranking

# Layer 3: Autoencoder
AE_EMBEDDING_DIM = 32  # Bottleneck dimension (16-64)
AE_HIDDEN_DIMS = [512, 128]  # Hidden layer dimensions
AE_BATCH_SIZE = 256  # Batch size
AE_EPOCHS = 50  # Maximum training epochs
AE_LR = 0.001  # Learning rate
AE_DROPOUT = 0.2  # Dropout rate (0.2-0.5)
AE_PATIENCE = 5  # Early stopping patience
AE_TOP_K = 30  # Number of candidates after autoencoder refinement

# Hybrid Ensemble
ENSEMBLE_WEIGHTS = [0.3, 0.5, 0.2]  # [content, svd, ae]
TUNE_WEIGHTS = True  # Optimize weights on validation set

# Evaluation
EVAL_K = 10  # K for Precision@K, Recall@K, NDCG@K metrics

# Device
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Model saving
MODEL_DIR = 'models'
SAVE_MODELS = True

