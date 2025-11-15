"""
Layer 1: Content-Based Recommender with Inverted Index.
Uses movie genres to build fast lookup structure for candidate generation.
O(n_genres × avg_movies_per_genre) instead of O(n_items × feature_dim).
"""

from collections import defaultdict


class ContentBasedRecommender:
    """
    Fast content-based filtering using inverted index on movie genres.
    Generates top-K candidates based on user's preferred genres.
    """
    
    def __init__(self, movies_df):
        """
        Initialize and build inverted index.
        
        Args:
            movies_df: DataFrame with movie metadata and genre columns
        """
        pass
    
    def _build_inverted_index(self):
        """
        Build genre -> movie_ids mapping.
        Assumes genre columns are binary indicators.
        """
        pass
    
    def _get_movie_genres(self, movie_id):
        """
        Get list of genres for a movie.
        """
        pass
    
    def get_user_preferred_genres(self, user_ratings):
        """
        Extract preferred genres from user's highly rated movies.
        
        Args:
            user_ratings: dict {movie_id: rating}
        
        Returns:
            dict: {genre: weight} based on rating aggregation
        """
        pass
    
    def recommend(self, user_ratings, top_k=100):
        """
        Generate candidate recommendations using inverted index.
        
        Args:
            user_ratings: dict {movie_id: rating} for the user
            top_k: Number of candidates to return
        
        Returns:
            list: Top-K movie IDs sorted by score
        """
        pass

