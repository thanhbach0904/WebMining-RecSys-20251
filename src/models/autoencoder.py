"""
Layer 3: Denoising Autoencoder (DAE) for Collaborative Filtering.
This module implements a CDAE-style architecture designed to learn robust 
latent representations of users/items by reconstructing corrupted input vectors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DenoisingAutoEncoder(nn.Module):
    """
    Feed-forward Denoising Autoencoder architecture.
    Learns a compact latent space (bottleneck) to capture non-linear 
    dependencies between items in a collaborative filtering setting.
    """
    
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128], 
                 dropout=0.3, noise_ratio=0.2):
        """
        Initializes the Encoder-Decoder symmetric structure.
        
        Args:
            n_items (int): Input/Output dimension (number of items in catalog).
            embedding_dim (int): Dimensionality of the bottleneck (latent embedding).
            hidden_dims (list): List of neurons in intermediate hidden layers.
            dropout (float): Dropout probability for regularization.
            noise_ratio (float): Probability of masking input entries during training.
        """
        super(DenoisingAutoEncoder, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.noise_ratio = noise_ratio
        
        # --- ENCODER STACK ---
        # Compresses high-dimensional sparse rating vectors into dense embeddings.
        encoder_layers = []
        prev_dim = n_items
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), # Batch normalization for internal covariate shift
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer representing the latent user preference vector
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- DECODER STACK ---
        # Reconstructs the original input from the latent representation.
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
        
        # Sigmoid activation ensures output scores are bounded between [0, 1]
        decoder_layers.append(nn.Linear(prev_dim, n_items))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def add_noise(self, x, mask):
        """
        Applies input corruption (Masking Noise).
        Forces the model to learn to recover missing ratings from observed ones.
        
        Returns:
            torch.Tensor: Corrupted input vector.
        """
        if not self.training or self.noise_ratio == 0:
            return x
        
        # Generate stochastic binary mask for denoising objective
        noise_mask = torch.rand_like(x) > self.noise_ratio
        
        # Corruption is applied only to observed entries to preserve sparsity structure
        corrupted = x * (noise_mask | (mask == 0)).float()
        
        return corrupted
    
    def forward(self, x, mask=None):
        """
        Standard forward pass through the DAE pipeline.
        
        Returns:
            tuple: (reconstruction_vector, latent_embedding)
        """
        # Inject noise if mask is provided and model is in training mode
        if mask is not None:
            x_noisy = self.add_noise(x, mask)
        else:
            x_noisy = x
        
        # Latent feature extraction
        embedding = self.encoder(x_noisy)
        # Output reconstruction
        reconstruction = self.decoder(embedding)
        
        return reconstruction, embedding
    
    def get_embedding(self, x):
        """Utility method to extract user latent features for downstream tasks."""
        return self.encoder(x)


class AutoEncoderTrainer:
    """
    High-level API for training and evaluating the Denoising Autoencoder.
    Implements masked loss functions and early stopping logic.
    """
    
    def __init__(self, n_items, embedding_dim=32, hidden_dims=[512, 128],
                 dropout=0.3, lr=0.001, weight_decay=1e-5,
                 noise_ratio=0.2, device='cpu'):
        """
        Sets up the optimization environment.
        """
        self.device = device
        self.n_items = n_items
        
        self.model = DenoisingAutoEncoder(
            n_items=n_items,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            noise_ratio=noise_ratio
        ).to(device)
        
        # Adam optimizer with L2 weight decay for complexity control
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.best_model_state = None
        self.best_val_loss = float('inf')
    
    def masked_mse_loss(self, predictions, targets, mask):
        """
        Computes Mean Squared Error exclusively on observed ratings.
        Prevents the model from being biased towards zero-filled missing entries.
        """
        masked_pred = predictions * mask
        masked_target = targets * mask
        n_observed = mask.sum()
        
        if n_observed == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sum of squared errors normalized by the number of observed interactions
        mse = ((masked_pred - masked_target) ** 2).sum() / n_observed
        return mse

    # ... Training loop methods (train_epoch, evaluate, etc.) follow similar documentation patterns