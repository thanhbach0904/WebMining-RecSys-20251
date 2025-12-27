"""
Central Configuration for the Recommender System.
Contains global constants, file paths, model hyperparameters, and execution flags.
Values defined here drive the behavior of training, evaluation, and inference.
"""

# --- File System Paths ---
DATA_DIR = 'ml-100k'
RATINGS_FILE = 'u.data'
MOVIES_FILE = 'u.item'
USERS_FILE = 'u.user'

# Info
NUM_USERS = 943
NUM_MOVIES = 1682
NUM_RATINGS = 100000
NUM_GENRES = 19

# --- Data Splitting Strategy ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2  # 20% of training data for AE validation
TEST_RATIO = 0.1
TIME_BASED_SPLIT = True
RANDOM_SEED = 42

# --- Model Layer 1: Content-Based ---
CONTENT_TOP_K = 300  # Increased candidate pool for late fusion

# --- Model Layer 2: Collaborative Filtering (SVD) ---
SVD_N_FACTORS = 50
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02

# --- Model Layer 3: Denoising Autoencoder ---
AE_EMBEDDING_DIM = 32
AE_HIDDEN_DIMS = [512, 128]
AE_BATCH_SIZE = 256
AE_EPOCHS = 100  # More epochs with early stopping
AE_LR = 0.001
AE_DROPOUT = 0.3  # Increased for regularization
AE_PATIENCE = 10  # Early stopping patience
AE_WEIGHT_DECAY = 1e-5  # L2 regularization
AE_NOISE_RATIO = 0.2  # Denoising: mask 20% of inputs

# --- Ensemble Normalization ---

# --- Meta-Learning ---
META_LEARNER_TYPE = 'logistic'  # 'logistic', 'mlp', or 'gbm'
META_HIDDEN_DIM = 32  # For MLP meta-learner
META_LR = 0.01
META_EPOCHS = 50

# --- Feature Engineering ---
USE_FEATURES = True
FEATURE_DIMS = {
    'user_mean': 1,
    'user_count': 1,
    'item_mean': 1,
    'item_count': 1,
    'item_popularity': 1,
    'genre_vector': NUM_GENRES,
}

# --- Evaluation Metrics ---
EVAL_K = 10

# Device
DEVICE = 'cuda'

# Model saving
MODEL_DIR = 'models'
SAVE_MODELS = True

