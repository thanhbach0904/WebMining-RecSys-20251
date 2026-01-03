"""
Meta-Learner for ensemble stacking.
Learns to combine model scores and features optimally.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
import lightgbm as lgb



class MLPMetaLearner(nn.Module):
    """Small MLP for meta-learning."""
    
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MetaLearner:
    """
    Stacking meta-learner that combines model scores and features.
    Supports logistic regression, MLP, or gradient boosting.
    """
    
    def __init__(self, learner_type='logistic', input_dim=None, 
                 hidden_dim=32, lr=0.01, epochs=50, device='cpu'):
        """
        Args:
            learner_type: 'logistic', 'mlp', or 'gbm'
            input_dim: Feature dimension (required for MLP)
            hidden_dim: Hidden dimension for MLP
            lr: Learning rate for MLP
            epochs: Training epochs for MLP
            device: Device for MLP
        """
        self.learner_type = learner_type
        self.device = device
        self.scaler = StandardScaler()
        self.fitted = False
        
        if learner_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs'
            )
        elif learner_type == 'mlp':
            if input_dim is None:
                raise ValueError("input_dim required for MLP meta-learner")
            self.model = MLPMetaLearner(input_dim, hidden_dim).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.epochs = epochs
        elif learner_type == 'gbm':
            self.model = None  # Created during fit
        else:
            raise ValueError(f"Unknown learner type: {learner_type}")
    
    def fit(self, X, y):
        """
        Train meta-learner on stacking features.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
               Features include: [svd_score, ae_score, user_features, item_features]
            y: numpy array of shape (n_samples,)
               Binary labels (1 if relevant, 0 if not)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.learner_type == 'logistic':
            self.model.fit(X_scaled, y)
        
        elif self.learner_type == 'mlp':
            self._train_mlp(X_scaled, y)
        
        elif self.learner_type == 'gbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            self.model.fit(X_scaled, y)
        
        self.fitted = True
    
    def _train_mlp(self, X, y):
        """Train MLP meta-learner."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        criterion = nn.BCELoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Meta-learner epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f}")
    
    def predict_proba(self, X):
        """
        Predict probability scores for items.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        
        Returns:
            numpy array: Probability scores in [0, 1]
        """
        if not self.fitted:
            raise ValueError("Meta-learner not fitted. Call fit() first.")
        
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        if self.learner_type == 'logistic':
            return self.model.predict_proba(X_scaled)[:, 1]
        
        elif self.learner_type == 'mlp':
            self.model.eval()
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                return self.model(X_tensor).cpu().numpy()
        
        elif self.learner_type == 'gbm':
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def score_items(self, X):
        """Alias for predict_proba for consistency."""
        return self.predict_proba(X)


# Data Flow Overview:
# -------------------
# Input  -> Raw ratings / metadata
# Process-> Feature extraction / model inference
# Output -> Normalized scores or ranked item lists
#
# This explicit flow helps maintain consistency across folds.
