"""
Data loading utilities for MovieLens-100K dataset.
Handles reading ratings, movies, and user information from disk.
"""

import os
import pandas as pd


def load_data(data_dir='ml-100k'):
    """
    Load MovieLens-100K dataset files.
    
    Returns:
        ratings: DataFrame with columns [user_id, item_id, rating, timestamp]
        movies: DataFrame with movie metadata and genre indicators
        users: DataFrame with user demographics
    """
    pass


def get_movie_title(movie_id, movies_df):
    """
    Get movie title by ID.
    """
    pass


def get_user_ratings(user_id, ratings_df):
    """
    Get all ratings for a specific user.
    
    Returns:
        dict: {movie_id: rating}
    """
    pass

