"""
Layer 2: Collaborative Filtering with SVD (Singular Value Decomposition).
Uses Surprise library for matrix factorization to discover latent factors.
Re-ranks candidates from Layer 1 based on predicted ratings.
"""
# Scripts
# python -m src.models.svd_collaborative
from surprise import SVD, Dataset, Reader #type: ignore
import pandas as pd #type: ignore
from src.data.loader import load_ratings_by_fold, load_movies_df
from src.models.content_based import ContentBasedRecommender
from config import SVD_N_FACTORS, SVD_N_EPOCHS, SVD_LR, SVD_REG
import time
class CollaborativeFilteringSVD:
    """
    SVD-based collaborative filtering using Surprise library.
    Learns latent factors from user-item rating matrix.
    """
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        """
        Initialize SVD model with hyperparameters.
        
        Args:
            n_factors: Number of latent factors (20-100)
            n_epochs: Number of training epochs (10-30)
            lr: Learning rate for SGD
            reg: Regularization strength
        """
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr,
            reg_all=reg,
            verbose=True
        )
        self.is_trained = False
    
    def train(self, train_data):
        """
        Train SVD model on training data.
        
        Args:
            train_data: DataFrame with columns [user_id, item_id, rating]
        """
        # Surprise requires specific column order
        reader = Reader(rating_scale=(1, 5))
        
        # Rename columns if needed to match expected format
        if 'item_id' in train_data.columns:
            df = train_data[['user_id', 'item_id', 'rating']].copy()
        else:
            df = train_data[['user_id', 'movie_id', 'rating']].copy()
            df.columns = ['user_id', 'item_id', 'rating']
        
        # Create Surprise Dataset
        dataset = Dataset.load_from_df(df, reader)
        trainset = dataset.build_full_trainset()
        
        self.model.fit(trainset)
        self.is_trained = True
        print(f"SVD model trained on {trainset.n_users} users and {trainset.n_items} items")
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.
        
        Returns:
            float: Predicted rating (1-5 scale)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        prediction = self.model.predict(user_id, item_id)
        return prediction.est
    
    def recommend(self, user_id, candidate_items, top_k=50):
        """
        Re-rank candidate items from Layer 1 by predicted ratings.
        
        Args:
            user_id: User ID
            candidate_items: List of movie IDs from content-based layer
            top_k: Number of top items to return
        
        Returns:
            list: Top-K movie IDs sorted by predicted rating
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predict ratings for all candidates
        predictions = []
        for item_id in candidate_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating (descending) and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:top_k]]
    
    def get_user_factors(self, user_id):
        """
        Get learned latent factors for a user.
        Useful for visualization and analysis.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            inner_uid = self.model.trainset.to_inner_uid(user_id)
            return self.model.pu[inner_uid]
        except ValueError:
            return None
    
    def get_item_factors(self, item_id):
        """
        Get learned latent factors for an item.
        Useful for visualization and analysis.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            inner_iid = self.model.trainset.to_inner_iid(item_id)
            return self.model.qi[inner_iid]
        except ValueError:
            return None


if __name__ == "__main__":
    # Load data
    ratings_df = load_ratings_by_fold() #load the u1.base by default
    movies_df = load_movies_df()
    
    content_model = ContentBasedRecommender(movies_df, ratings_df)
    user_id = 1
    candidates = content_model.recommend(user_id, top_k=100)
    print(f"Content-based candidates: {len(candidates)}")
    
    # Train and test SVD
    svd_model = CollaborativeFilteringSVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr=SVD_LR,
        reg=SVD_REG
    )
    start_time = time.time()
    svd_model.train(ratings_df)
    train_time = time.time()
    print(f"SVD train time: {train_time - start_time}")
    # Re-rank candidates
    reranked = svd_model.recommend(user_id, candidates, top_k=50)
    print(f"SVD re-ranked: {len(reranked)}")
    
    # Show top 10 predictions
    for movie_id in reranked[:10]:
        pred = svd_model.predict(user_id, movie_id)
        print(f"Movie {movie_id}: Predicted rating = {pred:.2f}")