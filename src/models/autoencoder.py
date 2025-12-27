"""
Layer 3: Denoising Autoencoder for user/item embedding learning.
Implements a CDAE-style architecture with configurable noise injection for robust representation learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DenoisingAutoEncoder(nn.Module):
    """
    Feed-forward Denoising Autoencoder for collaborative filtering.
    Applies input corruption (masking) to learn robust features.
    """
    
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128], 
                 dropout=0.3, noise_ratio=0.2):
        """
        Args:
            n_items (int): Dimension of input/output layer (corresponds to number of items).
            embedding_dim (int): Size of the bottleneck layer.
            hidden_dims (list): Dimensions of intermediate hidden layers.
            dropout (float): Probability of dropout.
            noise_ratio (float): Probability of input corruption.
        """
        super(DenoisingAutoEncoder, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.noise_ratio = noise_ratio
        
        encoder_layers = []
        prev_dim = n_items
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, n_items))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def add_noise(self, x, mask):
        """
        Inject noise into the input tensor to facilitate denoising training.
        Corrupts only the observed entries (where mask is 1).
        
        Args:
            x (torch.Tensor): Input rating vector.
            mask (torch.Tensor): Binary mask indicating observed ratings.
        
        Returns:
            torch.Tensor: The corrupted input vector.
        """
        if not self.training or self.noise_ratio == 0:
            return x
        
        # Create noise mask (only for observed entries)
        noise_mask = torch.rand_like(x) > self.noise_ratio
        
        # Zero out some observed ratings (Masking noise)
        corrupted = x * (noise_mask | (mask == 0)).float()
        
        return corrupted
    
    def forward(self, x, mask=None):
        """
        Resconstruct the input rating vector.
        
        Returns:
            tuple: (reconstruction, embedding)
        """
        if mask is not None:
            x_noisy = self.add_noise(x, mask)
        else:
            x_noisy = x
        
        embedding = self.encoder(x_noisy)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    
    def get_embedding(self, x):
        """Extract the bottleneck embedding for a given input."""
        return self.encoder(x)


class AutoEncoderTrainer:
    """
    Training wrapper for denoising autoencoder with proper validation.
    """
    
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128],
                 dropout=0.3, lr=0.001, weight_decay=1e-5,
                 noise_ratio=0.2, device='cpu'):
        """Initialize trainer with model and optimizer."""
        self.device = device
        self.n_items = n_items
        
        self.model = DenoisingAutoEncoder(
            n_items=n_items,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            noise_ratio=noise_ratio
        ).to(device)
        
        # Adam optimizer with L2 regularization (weight decay)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.best_model_state = None
        self.best_val_loss = float('inf')
    
    def masked_mse_loss(self, predictions, targets, mask):
        """Compute MSE loss only on observed ratings."""
        masked_pred = predictions * mask
        masked_target = targets * mask
        n_observed = mask.sum()
        
        if n_observed == 0:
            return torch.tensor(0.0, device=self.device)
        
        mse = ((masked_pred - masked_target) ** 2).sum() / n_observed
        return mse
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            ratings, masks = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(ratings, masks)
            loss = self.masked_mse_loss(reconstruction, ratings, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ratings, masks = batch[0].to(self.device), batch[1].to(self.device)
                reconstruction, _ = self.model(ratings, masks)
                loss = self.masked_mse_loss(reconstruction, ratings, masks)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def train_full(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Full training loop with early stopping on validation loss.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (REQUIRED)
            epochs: Maximum epochs
            patience: Early stopping patience
        """
        if val_loader is None:
            raise ValueError("Validation loader is required for proper training!")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_val_loss = best_val_loss
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.6f})")
                    break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model with val_loss: {self.best_val_loss:.6f}")
    
    def predict_ratings(self, user_rating_vector):
        """Predict all ratings for a user."""
        self.model.eval()
        
        if isinstance(user_rating_vector, np.ndarray):
            user_rating_vector = torch.FloatTensor(user_rating_vector)
        user_rating_vector = user_rating_vector.to(self.device)
        
        with torch.no_grad():
            if len(user_rating_vector.shape) == 1:
                user_rating_vector = user_rating_vector.unsqueeze(0)
            reconstruction, _ = self.model(user_rating_vector, mask=None)
        
        return reconstruction.cpu().numpy().flatten()
    
    def predict_batch(self, user_rating_matrix):
        """Predict ratings for multiple users at once."""
        self.model.eval()
        
        if isinstance(user_rating_matrix, np.ndarray):
            user_rating_matrix = torch.FloatTensor(user_rating_matrix)
        user_rating_matrix = user_rating_matrix.to(self.device)
        
        with torch.no_grad():
            reconstruction, _ = self.model(user_rating_matrix, mask=None)
        
        return reconstruction.cpu().numpy()
    
    def get_user_embedding(self, user_rating_vector):
        """Get the learned embedding for a user."""
        self.model.eval()
        
        if isinstance(user_rating_vector, np.ndarray):
            user_rating_vector = torch.FloatTensor(user_rating_vector)
        user_rating_vector = user_rating_vector.to(self.device)
        
        with torch.no_grad():
            if len(user_rating_vector.shape) == 1:
                user_rating_vector = user_rating_vector.unsqueeze(0)
            embedding = self.model.get_embedding(user_rating_vector)
        
        return embedding.cpu().numpy().flatten()
    
    def score_items(self, user_rating_vector, item_ids):
        """
        Score specific items for a user.
        
        Args:
            user_rating_vector: User's rating vector
            item_ids: List of item IDs to score
        
        Returns:
            dict: {item_id: score}
        """
        all_predictions = self.predict_ratings(user_rating_vector)
        scores = {}
        for item_id in item_ids:
            idx = item_id - 1  # 0-based index
            if 0 <= idx < len(all_predictions):
                scores[item_id] = all_predictions[idx]
            else:
                scores[item_id] = 0.0
        return scores