"""
Collection of utility functions for data preparation and processing user rating vectors.
"""

from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd #type: ignore


def prepare_data_loader(ratings_df, n_users, n_items, batch_size=256):
    """
    Construct a PyTorch DataLoader from the ratings DataFrame.
    """
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    mask_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    
    for _, row in ratings_df.iterrows():
        user_idx = int(row['user_id']) - 1
        item_idx = int(row['item_id']) - 1
        rating = row['rating']
        
        if 0 <= user_idx < n_users and 0 <= item_idx < n_items:
            rating_matrix[user_idx, item_idx] = rating
            mask_matrix[user_idx, item_idx] = 1.0
    
    # Normalize ratings to [0, 1]
    rating_matrix = rating_matrix / 5.0
    
    ratings_tensor = torch.FloatTensor(rating_matrix)
    mask_tensor = torch.FloatTensor(mask_matrix)
    
    dataset = TensorDataset(ratings_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def prepare_train_val_loaders(ratings_df, n_users, n_items, val_ratio=0.2, 
                               batch_size=256, random_seed=42):
    """
    Create separate DataLoaders for training and validation by splitting user ratings.
    Typically reserves 20% of EACH USER's ratings for validation (stratified user split).
    
    Args:
        ratings_df (pd.DataFrame): The complete ratings dataset.
        n_users (int): Total count of users.
        n_items (int): Total count of items.
        val_ratio (float): Proportion of ratings to use for validation (default: 0.2).
        batch_size (int): Size of batches for the DataLoader.
        random_seed (int): Seed for random number generation to ensure reproducibility.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    np.random.seed(random_seed)
    
    train_rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    train_mask_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    val_rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    val_mask_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    
    # Split per-user
    for user_id in ratings_df['user_id'].unique():
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        n_ratings = len(user_ratings)
        
        # Shuffle and split
        indices = np.random.permutation(n_ratings)
        n_val = max(1, int(n_ratings * val_ratio))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        user_idx = int(user_id) - 1
        
        for i, (_, row) in enumerate(user_ratings.iterrows()):
            item_idx = int(row['item_id']) - 1
            rating = row['rating'] / 5.0
            
            if 0 <= user_idx < n_users and 0 <= item_idx < n_items:
                if i in val_indices:
                    val_rating_matrix[user_idx, item_idx] = rating
                    val_mask_matrix[user_idx, item_idx] = 1.0
                else:
                    train_rating_matrix[user_idx, item_idx] = rating
                    train_mask_matrix[user_idx, item_idx] = 1.0
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_rating_matrix),
        torch.FloatTensor(train_mask_matrix)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_rating_matrix),
        torch.FloatTensor(val_mask_matrix)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train/Val split: {train_mask_matrix.sum():.0f} / {val_mask_matrix.sum():.0f} ratings")
    
    return train_loader, val_loader


def get_user_rating_vector(user_id, ratings_df, n_items):
    """Generate a dense rating vector for a specific user."""
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    rating_vector = np.zeros(n_items, dtype=np.float32)
    
    for _, row in user_ratings.iterrows():
        item_idx = int(row['item_id']) - 1
        if 0 <= item_idx < n_items:
            rating_vector[item_idx] = row['rating'] / 5.0
    
    return rating_vector


def load_users_df(data_dir='ml-100k', file_name='u.user'):
    """Load user demographic information from the dataset."""
    import os
    from src.data.loader import get_project_root
    
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_dir, file_name)
    
    users_df = pd.read_csv(
        data_path,
        sep='|',
        header=None,
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )
    return users_df