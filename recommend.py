"""
Generate recommendations for specific users.

Usage:
    python recommend.py --user_id 1 --top_k 10
    python recommend.py --user_id 1 --top_k 10 --explain
"""

import argparse
import pandas as pd
from src.data.loader import load_data
from src.models.hybrid import HybridRecommender


def main():
    parser = argparse.ArgumentParser(description='Generate movie recommendations')
    parser.add_argument('--user_id', type=int, required=True, help='User ID')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--explain', action='store_true', help='Show explanation for recommendations')
    args = parser.parse_args()
    
    # Load data and model
    print(f"Loading data and models...")
    ratings, movies, users = load_data('ml-100k')
    
    # Load trained hybrid model
    # hybrid_model = load_hybrid_model('models/')
    
    print(f"\nGenerating recommendations for User {args.user_id}...")
    
    # Get user's rating history
    user_ratings = ratings[ratings['user_id'] == args.user_id]
    print(f"User has rated {len(user_ratings)} movies")
    
    # Generate recommendations
    # recommendations = hybrid_model.recommend(args.user_id, top_k=args.top_k)
    
    # Display results
    print(f"\nTop {args.top_k} Recommendations for User {args.user_id}:")
    print("-" * 60)
    # for i, (movie_id, score) in enumerate(recommendations, 1):
    #     movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    #     print(f"{i:2d}. {movie_title:50s} (Score: {score:.3f})")
    
    # if args.explain:
    #     print("\nExplanation:")
    #     # Show contribution from each layer


if __name__ == "__main__":
    main()

