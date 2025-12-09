"""
Data loading utilities for MovieLens-100K dataset.
Handles reading ratings, movies, and user information from disk.
"""

import os
import pandas as pd # type:ignore

def get_project_root():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    return current_dir
def load_ratings_by_fold(data_dir='ml-100k', fold_name = "u1.base"):
    """
    Load MovieLens-100K dataset files by specific fold.
    
    Returns:
        ratings: DataFrame with columns [user_id, item_id, rating, timestamp]
    """
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_dir, fold_name)
    rating_df = pd.read_csv(data_path, sep = '\t', header = None, names = ['user_id', 'item_id', 'rating', 'timestamp'])

    return rating_df

def load_movies_df(data_dir="ml-100k", file_name="u.item"):
    """
    Load movies data from MovieLens-100K dataset.
    
    Returns:
        movies_df: DataFrame with movie information including genres
    """
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_dir, file_name)
    
    column_names = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 
        'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western']
    
    movies_df = pd.read_csv(
        data_path, 
        sep='|', 
        header=None, 
        names=column_names,
        encoding='latin-1')
    return movies_df

def get_movie_genre_mapping(data_dir="ml-100k", file_name="u.item"):
    """
    Create a mapping from movie_id to binary genre vector.
    
    Returns:
        dict: {movie_id: its binary vector (in numpy array)}
    """
    movies_df = load_movies_df(data_dir, file_name)
    
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western']
    
    movie_genre_map = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movie_id']
        genre_vector = row[genre_columns].values  
        movie_genre_map[movie_id] = genre_vector
    return movie_genre_map

def get_movie_title(movie_id: int, movies_df : pd.DataFrame):
    """
    Get movie title by ID.
    """
    try: 
        for _, row in movies_df.iterrows():
            if row['movie_id'] == movie_id:
                return row['movie_title']
    except Exception as e:
        print(f"Error when get the movie title by ID: {e}")


def get_user_ratings(user_id, ratings_df: pd.DataFrame):
    """
    Get all ratings for a specific user as a dictionary.
    
    Returns:
        dict: {movie_id: rating}
    """
    user_data = ratings_df[ratings_df['user_id'] == user_id][['item_id', 'rating']]
    return dict(zip(user_data['item_id'], user_data['rating']))

if __name__ == "__main__":
    print(get_project_root())
    ratings_df = load_ratings_by_fold()
    print(ratings_df)
    movies_df = load_movies_df()
    print(movies_df)
    print(get_movie_title(1, movies_df))
    print(get_user_ratings(1, ratings_df))