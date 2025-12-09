"""
Layer 3: Autoencoder for learning dense user/item embeddings.
Handles sparse rating data with masked loss function.
Architecture: Input → Encoder → Bottleneck → Decoder → Output
"""
# Scripts
# python -m src.models.autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.data.loader import load_ratings_by_fold, load_movies_df
from src.models.content_based import ContentBasedRecommender
from src.models.svd_collaborative import CollaborativeFilteringSVD
from config import (NUM_USERS, NUM_MOVIES, AE_EMBEDDING_DIM, 
                    AE_HIDDEN_DIMS, AE_BATCH_SIZE, AE_EPOCHS, 
                    AE_LR, AE_DROPOUT, DEVICE)


class AutoEncoderRecommender(nn.Module):

    """
    Deep autoencoder for collaborative filtering.
    Learns compressed representations of sparse rating vectors.
    """
    
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128], dropout=0.2):
        """
        Initialize autoencoder architecture.
        Args:
            n_items: Number of items (input/output dimension)
            embedding_dim: Bottleneck dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """

        super(AutoEncoderRecommender, self).__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # Build encoder layers
        encoder_layers = []
        prev_dim = n_items
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, n_items))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through encoder and decoder.
        Returns:
            reconstruction: Reconstructed rating vector
            embedding: Bottleneck representation
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        """
        Get embedding representation without reconstruction.
        """
        return self.encoder(x)

class AutoEncoderTrainer:
    """
    Training wrapper for autoencoder with masked loss.
    Only computes loss on observed ratings.
    """
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128], 

                 dropout=0.2, lr=0.001, device='cpu'):
        """
        Initialize trainer with model and optimizer.
        """
        self.device = device
        self.n_items = n_items
        self.model = AutoEncoderRecommender(
            n_items=n_items,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.best_val_loss = float('inf')

    def masked_mse_loss(self, predictions, targets, mask):
        """
        Compute MSE loss only on observed ratings.
        Args:
            predictions: Reconstructed ratings
            targets: Original ratings
            mask: Binary mask (1 where rating exists, 0 otherwise)
        """
        masked_pred = predictions * mask
        masked_target = targets * mask
        # Count number of observed ratings
        n_observed = mask.sum()
        if n_observed == 0:
            return torch.tensor(0.0, device=self.device)

        # Compute MSE only on observed entries
        mse = ((masked_pred - masked_target) ** 2).sum() / n_observed
        return mse

    

    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            ratings, masks = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(ratings)
            loss = self.masked_mse_loss(reconstruction, ratings, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0

    def evaluate(self, val_loader):
        """
        Evaluate on validation set.
        Returns:
            float: Validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                ratings, masks = batch[0].to(self.device), batch[1].to(self.device)
                reconstruction, _ = self.model(ratings)
                loss = self.masked_mse_loss(reconstruction, ratings, masks)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0

    def train_full(self, train_loader, val_loader=None, epochs=50, patience=5):
        """
        Full training loop with early stopping.
        """

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(self.best_model_state)
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    

    def predict_ratings(self, user_rating_vector):

        """
        Predict all ratings for a user given their sparse rating vector.
        Args:
            user_rating_vector: numpy array or tensor of shape (n_items,)
        Returns:
            numpy array: Predicted ratings for all items
        """

        self.model.eval()

        if isinstance(user_rating_vector, np.ndarray):
            user_rating_vector = torch.FloatTensor(user_rating_vector)
        user_rating_vector = user_rating_vector.to(self.device)
        
        with torch.no_grad():
            if len(user_rating_vector.shape) == 1:
                user_rating_vector = user_rating_vector.unsqueeze(0)
            reconstruction, _ = self.model(user_rating_vector)

        return reconstruction.cpu().numpy().flatten()
    def recommend(self, user_rating_vector, candidate_items, top_k=30):
        """
        Final ranking of candidates using autoencoder predictions.
        Args:
            user_rating_vector: Sparse vector of user's ratings (shape: n_items)
            candidate_items: List of candidate movie IDs from Layer 2
            top_k: Number of final recommendations
        Returns:
            list: Top-K movie IDs sorted by predicted rating
        """
        # Get predicted ratings for all items
        predicted_ratings = self.predict_ratings(user_rating_vector)

        # Score only the candidate items
        # Note: movie_id to index conversion (movie_id starts from 1)
        predictions = []
        for item_id in candidate_items:
            idx = item_id - 1  # Convert to 0-based index
            if 0 <= idx < len(predicted_ratings):
                predictions.append((item_id, predicted_ratings[idx]))
        # Sort by predicted rating (descending) and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:top_k]]

    def get_user_embedding(self, user_rating_vector):
        """
        Get the learned embedding for a user.
        Useful for similarity computation and visualization.
        """

        self.model.eval()
        if isinstance(user_rating_vector, np.ndarray):
            user_rating_vector = torch.FloatTensor(user_rating_vector)
        user_rating_vector = user_rating_vector.to(self.device)
        
        with torch.no_grad():
            if len(user_rating_vector.shape) == 1:
                user_rating_vector = user_rating_vector.unsqueeze(0)
            embedding = self.model.get_embedding(user_rating_vector)
        return embedding.cpu().numpy().flatten()


def prepare_data_loader(ratings_df, n_users, n_items, batch_size=256):
    """
    Prepare PyTorch DataLoader from ratings DataFrame.
    Creates sparse rating vectors and binary masks for each user.
    Args:
        ratings_df: DataFrame with columns [user_id, item_id, rating]
        n_users: Total number of users
        n_items: Total number of items
        batch_size: Batch size for DataLoader
    Returns:
        DataLoader: PyTorch DataLoader with (ratings, masks) batches
    """
    # Create user-item matrix and mask
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    mask_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in ratings_df.iterrows():
        user_idx = int(row['user_id']) - 1  # 0-based indexing
        item_idx = int(row['item_id']) - 1  # 0-based indexing
        rating = row['rating']

        if 0 <= user_idx < n_users and 0 <= item_idx < n_items:
            rating_matrix[user_idx, item_idx] = rating
            mask_matrix[user_idx, item_idx] = 1.0

    # Normalize ratings to [0, 1] for better training
    rating_matrix = rating_matrix / 5.0
    # Convert to tensors
    ratings_tensor = torch.FloatTensor(rating_matrix)
    mask_tensor = torch.FloatTensor(mask_matrix)

    # Create dataset and dataloader
    dataset = TensorDataset(ratings_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader





def get_user_rating_vector(user_id, ratings_df, n_items):
    """
    Create a rating vector for a single user.

    Args:
        user_id: User ID
        ratings_df: Ratings DataFrame
        n_items: Total number of items

    Returns:
        numpy array: Sparse rating vector (normalized to [0, 1])
    """

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    rating_vector = np.zeros(n_items, dtype=np.float32)
    for _, row in user_ratings.iterrows():
        item_idx = int(row['item_id']) - 1
        if 0 <= item_idx < n_items:
            rating_vector[item_idx] = row['rating'] / 5.0  # Normalize

    return rating_vector





if __name__ == "__main__":

    # Load data
    ratings_df = load_ratings_by_fold()
    movies_df = load_movies_df()
    
    # Prepare data loader
    train_loader = prepare_data_loader(
        ratings_df, NUM_USERS, NUM_MOVIES, batch_size=AE_BATCH_SIZE
    )
    
    device = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    ae_trainer = AutoEncoderTrainer(
        n_items=NUM_MOVIES,
        embedding_dim=AE_EMBEDDING_DIM,
        hidden_dims=AE_HIDDEN_DIMS,
        dropout=AE_DROPOUT,
        lr=AE_LR,
        device=device
    )
    # Train (using same data for train/val for demo - in practice, split properly)
    ae_trainer.train_full(train_loader, val_loader=train_loader, epochs=10, patience=3)
    # Test the pipeline
    user_id = 1
    # Get candidates from content-based
    content_model = ContentBasedRecommender(movies_df, ratings_df)
    candidates_100 = content_model.recommend(user_id, top_k=100)
    print(f"Content-based candidates: {len(candidates_100)}")
    # Re-rank with SVD
    svd_model = CollaborativeFilteringSVD()
    svd_model.train(ratings_df)
    candidates_50 = svd_model.recommend(user_id, candidates_100, top_k=50)
    print(f"SVD re-ranked: {len(candidates_50)}")

    # Final ranking with autoencoder
    user_rating_vector = get_user_rating_vector(user_id, ratings_df, NUM_MOVIES)
    final_recommendations = ae_trainer.recommend(user_rating_vector, candidates_50, top_k=30)
    print(f"Final recommendations: {len(final_recommendations)}")
    # Print top 10
    print("\nTop 10 recommendations:")
    for i, movie_id in enumerate(final_recommendations[:10], 1):
        movie_row = movies_df[movies_df['movie_id'] == movie_id]
        if not movie_row.empty:
            title = movie_row.iloc[0]['movie_title']
            print(f"{i}. {title}")