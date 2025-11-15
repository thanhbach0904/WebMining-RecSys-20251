"""
Data preprocessing and splitting utilities.
Handles train/val/test splits and sparse matrix creation.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def split_data(ratings_df, train_ratio=0.8, val_ratio=0.1, 
               time_based=True, random_seed=42):
    """
    Split ratings into train, validation, and test sets.
    
    Args:
        ratings_df: DataFrame with ratings
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        time_based: If True, split by timestamp; else random split
        random_seed: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: DataFrames
    """
    pass


def create_user_item_matrix(ratings_df, n_users=None, n_items=None):
    """
    Create sparse user-item rating matrix.
    
    Returns:
        csr_matrix: Sparse matrix of shape (n_users, n_items)
    """
    pass


def normalize_ratings(ratings_array, method='mean_center'):
    """
    Normalize rating values.
    
    Args:
        method: 'mean_center' or 'min_max'
    """
    pass

