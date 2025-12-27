"""
Layer 1: Content-Based Recommender with Inverted Index.
Uses movie genres to build fast lookup structure for candidate generation.
O(n_genres × avg_movies_per_genre) instead of O(n_items × feature_dim).
"""
# Script:
# python -m src.models.content_based
 
from typing import Any
from collections import defaultdict
from config import NUM_USERS, NUM_MOVIES, NUM_RATINGS
from src.data.loader import load_movies_df, get_user_ratings, load_ratings_by_fold, get_movie_title
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Implements fast content-based filtering via an inverted genre index.
    Efficiently retrieves top-K candidate items that align with a user's genre preferences.
    """
    def __init__(self, movies_df, ratings_df):
        """
        Construct the inverted index upon initialization.
        
        Args:
            movies_df: DataFrame with movie metadata and genre columns
        """
        self.genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western']
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # Precompute genre matrix for cosine similarity
        self.movie_ids = self.movies_df['movie_id'].values
        self.genre_matrix = self.movies_df[self.genre_columns].values
    
    def recommend(self, user_id, top_k=100):
        # 1. Get User History
        user_ratings = get_user_ratings(user_id, self.ratings_df) # dict: {movie_id: rating}
        rated_movie_ids = list(user_ratings.keys())
        
        if not rated_movie_ids:
            return []

        # 2. Compute Similarity for each rated movie
        # Filter movies_df to get indices for rated movies
        rated_indices = self.movies_df[self.movies_df['movie_id'].isin(rated_movie_ids)].index
        
        # Calculate cosine similarity between rated movies (source) and ALL movies (candidates)
        # shape: (n_rated, n_total_movies)
        sim_matrix = cosine_similarity(self.genre_matrix[rated_indices], self.genre_matrix)
        
        candidate_scores = defaultdict(float)
        
        # 3. Aggregation (Summing similarities from top-10 neighbors for each rated movie)
        for i, rated_idx in enumerate(rated_indices):
            # Get similarity row for this rated movie
            sim_scores = sim_matrix[i]
            
            # Get indices of top-11 most similar (skipping itself which is index rated_idx)
            # We use top-11 because the movie itself will have sim=1.0
            top_indices = np.argsort(sim_scores)[::-1][:11] 
            
            for idx in top_indices:
                movie_id = self.movie_ids[idx]
                if movie_id not in user_ratings: # Exclude already rated
                    candidate_scores[movie_id] += sim_scores[idx]

        # 4. Sort and Return
        sorted_movies = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_movies[:top_k]]

if __name__ == "__main__":
    print(NUM_USERS)
    print(NUM_MOVIES)
    print(NUM_RATINGS)
    movies_df = load_movies_df()
    ratings_df = load_ratings_by_fold()
    
    user_id = 1
    recommender_engine = ContentBasedRecommender(movies_df, ratings_df)
    # print(recommender_engine.get_user_preferred_genres(user_id))
    start_time = time.time()
    recommended_movies = recommender_engine.recommend(user_id)
    end_time = time.time()
    print(f"Time to run content based recommender: {end_time - start_time}")
    for movie_id in recommended_movies:
        print(get_movie_title(movie_id, movies_df))