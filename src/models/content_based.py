"""
Layer 1: Content-Based Recommender with Inverted Index.
Uses movie genres to build fast lookup structure for candidate generation.
O(n_genres × avg_movies_per_genre) instead of O(n_items × feature_dim).
"""
# Script:
# python -m src.models.content_based
 
from collections import defaultdict
from config import NUM_USERS, NUM_MOVIES, NUM_RATINGS
from src.data.loader import load_movies_df, get_user_ratings
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
        self.movies_df = load_movies_df()
        self.genre_movieID_index = self._build_inverted_index()
        pass
    
    def _build_inverted_index(self):
        """
        Build genre -> movie_ids mapping.
        Assumes genre columns are binary indicators.
        """

        genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western']
        res = {}
        for genre in genre_columns:
            res[genre] = []
        
        return res
    
    def _get_movie_genres(self, movie_id):
        """
        Get list of genres for a movie.
        """
        pass
    
    def get_user_preferred_genres(self, user_ratings):
        """
        Extract preferred genres from user's highly rated movies.
        Count the times the genre appear in the user ratings
        For example if the user rates 5 movies, and 3 of them is Action movie, then the weight will be 3/5.
        The same for other genre
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

if __name__ == "__main__":
    print(NUM_USERS)
    print(NUM_MOVIES)
    print(NUM_RATINGS)