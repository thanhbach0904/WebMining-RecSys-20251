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

class ContentBasedRecommender:
    """
    Fast content-based filtering using inverted index on movie genres.
    Generates top-K candidates based on user's preferred genres.
    """
    def __init__(self, movies_df, ratings_df):
        """
        Initialize and build inverted index.
        
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
        self.genre_movieID_index = self._build_inverted_index()
        pass
    
    def _build_inverted_index(self):
        res = defaultdict(list)
        for _, row in self.movies_df.iterrows():
            movie_id = row['movie_id']
            for genre in self.genre_columns:
                if row[genre] == 1:
                    res[genre].append(movie_id)
        return res
    
    def _get_movie_genres(self, movie_id):
        movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if movie_row.empty:
            return []
        genres = []
        for genre in self.genre_columns:
            if movie_row.iloc[0][genre] == 1:
                genres.append(genre)
        return genres
    
    def get_user_preferred_genres(self, user_id, existed_user_rating = None):
        if existed_user_rating:
            user_ratings = existed_user_rating
        else:
            user_ratings = get_user_ratings(user_id, self.ratings_df)
        genre_count = defaultdict(int)
        total_movies = len(user_ratings)
        
        for movie_id in user_ratings.keys():
            genres = self._get_movie_genres(movie_id)
            for genre in genres:
                genre_count[genre] += 1
        
        genre_weights = {genre: round(count / total_movies, 2) for genre, count in genre_count.items()}
        return genre_weights
    
    def recommend(self, user_id, top_k=100):
        user_ratings = get_user_ratings(user_id, self.ratings_df)
        genre_weights = self.get_user_preferred_genres(user_id, existed_user_rating= user_ratings)
        
        movie_scores = defaultdict(float)
        for genre, weight in genre_weights.items():
            for movie_id in self.genre_movieID_index[genre]:
                if movie_id not in user_ratings:
                    movie_scores[movie_id] += weight
        
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_movies[:top_k]]

if __name__ == "__main__":
    print(NUM_USERS)
    print(NUM_MOVIES)
    print(NUM_RATINGS)
    movies_df = load_movies_df()
    ratings_df = load_ratings_by_fold()
    
    user_id = 1
    recommender_engine = ContentBasedRecommender(movies_df, ratings_df)
    print(recommender_engine.get_user_preferred_genres(user_id))
    start_time = time.time()
    recommended_movies = recommender_engine.recommend(user_id)
    end_time = time.time()
    print(f"Time to run content based recommender: {end_time - start_time}")
    for movie_id in recommended_movies:
        print(get_movie_title(movie_id, movies_df))