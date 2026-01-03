"""
Streamlit Demo for Hybrid Recommendation System
Run with: streamlit run app.py
"""

"""
This module is part of the Web Mining Recommendation System project.

Purpose:
- Describe the responsibility of this module
- Explain how it fits into the overall pipeline

Note:
- This file contains no execution entry point
- All logic is imported and orchestrated elsewhere
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os

# Import project modules
from src.data.loader import load_ratings_by_fold, load_movies_df, get_movie_title
from src.models.content_based import ContentBasedRecommender
from src.models.svd_collaborative import CollaborativeFilteringSVD
from src.models.autoencoder import AutoEncoderTrainer
from src.models.hybrid import LateFusionHybridRecommender
from src.features.features import FeatureExtractor
from src.utils.utils import load_users_df
import config

# Page config
st.set_page_config(
    page_title="Movie Recommender Demo",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hit-movie {
        background-color: #2ECC71 !important;
        color: white !important;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        font-weight: 600;
    }
    .miss-movie {
        background-color: #34495E;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .ground-truth {
        background-color: #3498DB;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .history-movie {
        background-color: #9B59B6;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .stSelectbox > div > div {
        background-color: #2D2D2D;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_movies():
    """Load movies dataframe (cached)."""
    return load_movies_df()


@st.cache_resource
def load_users():
    """Load users dataframe (cached)."""
    return load_users_df()


@st.cache_data
def load_fold_data(fold_num):
    """Load train and test data for a specific fold."""
    train_df = load_ratings_by_fold(fold_name=f"u{fold_num}.base")
    test_df = load_ratings_by_fold(fold_name=f"u{fold_num}.test")
    return train_df, test_df


@st.cache_resource
def load_fold_models(fold_num, movies_df, train_df, users_df):
    """Load trained models for a specific fold."""
    model_dir = config.MODEL_DIR
    
    # Load SVD model
    svd_path = os.path.join(model_dir, f'svd_fold{fold_num}.pkl')
    with open(svd_path, 'rb') as f:
        svd_surprise_model = pickle.load(f)
    
    # Create SVD wrapper and set the loaded model
    svd_model = CollaborativeFilteringSVD()
    svd_model.model = svd_surprise_model
    svd_model.is_trained = True
    
    # Load Autoencoder model
    ae_path = os.path.join(model_dir, f'autoencoder_fold{fold_num}.pt')
    checkpoint = torch.load(ae_path, map_location='cpu')
    
    device = 'cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'
    ae_trainer = AutoEncoderTrainer(
        n_items=checkpoint['n_items'],
        embedding_dim=config.AE_EMBEDDING_DIM,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
        device=device
    )
    ae_trainer.model.load_state_dict(checkpoint['model_state_dict'])
    ae_trainer.model.eval()
    
    # Create content-based model
    content_model = ContentBasedRecommender(movies_df, train_df)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(train_df, movies_df, users_df)

    weights_path = "models\best_weights.npy"
    if os.path.exists(weights_path):
        best_weights = np.load(weights_path).tolist()
    else:
        best_weights = [0.5, 0.5]
    
    # Create hybrid recommender
    hybrid = LateFusionHybridRecommender(
        content_model=content_model,
        svd_model=svd_model,
        ae_model=ae_trainer,
        feature_extractor=feature_extractor,
        meta_learner=None,
        weights=best_weights,
        norm_method=config.NORM_METHOD
    )
    
    return hybrid


def get_ground_truth(user_id, test_df, rating_threshold=4):
    """
    Get ground truth items for a user (items rated >= threshold in test set).
    These are the items the user actually liked.
    """
    user_test = test_df[test_df['user_id'] == user_id]
    liked_items = user_test[user_test['rating'] >= rating_threshold]['item_id'].tolist()
    return set(liked_items)


def get_all_test_items(user_id, test_df):
    """Get all items in the test set for a user with their ratings."""
    user_test = test_df[test_df['user_id'] == user_id]
    return dict(zip(user_test['item_id'], user_test['rating']))


def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load static data
    movies_df = load_movies()
    users_df = load_users()
    
    # Sidebar for fold selection
    st.sidebar.header("⚙️ Configuration")
    fold_num = st.sidebar.selectbox(
        "Select Fold",
        options=[1, 2, 3, 4, 5],
        index=0,
        help="Choose which fold's model to load (1-5)"
    )
    
    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    rating_threshold = st.sidebar.slider(
        "Ground Truth Rating Threshold",
        min_value=3,
        max_value=5,
        value=4,
        help="Movies rated >= this value are considered 'liked'"
    )
    
    # Load fold data and models
    with st.spinner(f"Loading Fold {fold_num} data..."):
        train_df, test_df = load_fold_data(fold_num)
    
    with st.spinner(f"Loading Fold {fold_num} models..."):
        hybrid_model = load_fold_models(fold_num, movies_df, train_df, users_df)
    
    st.sidebar.success(f"✅ Fold {fold_num} loaded!")
    st.sidebar.info(f"Train: {len(train_df):,} ratings\nTest: {len(test_df):,} ratings")
    
    # Get unique users from test set
    test_users = sorted(test_df['user_id'].unique())
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Select User")
        
        # User selection
        selected_user = st.selectbox(
            "User ID",
            options=test_users,
            format_func=lambda x: f"User {x}",
            help="Select a user to generate recommendations"
        )
        
        # Show user info
        if selected_user:
            user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
            st.markdown("**User Info:**")
            st.write(f"- Age: {user_info['age']}")
            st.write(f"- Gender: {user_info['gender']}")
            st.write(f"- Occupation: {user_info['occupation']}")
            
            # Show user's training history
            user_train = train_df[train_df['user_id'] == selected_user]
            st.write(f"- Training ratings: {len(user_train)}")
            
            # Show user's test items
            user_test = test_df[test_df['user_id'] == selected_user]
            st.write(f"- Test items: {len(user_test)}")
            
            ground_truth = get_ground_truth(selected_user, test_df, rating_threshold)
            st.write(f"- Liked items (rating ≥ {rating_threshold}): {len(ground_truth)}")
    
    with col2:
        st.subheader("🎯 Recommendations & Comparison")
        
        if selected_user:
            # Generate recommendations
            with st.spinner("Generating recommendations..."):
                recommendations, scores = hybrid_model.recommend(selected_user, top_k=top_k, return_scores=True)
            
            # Get ground truth
            ground_truth = get_ground_truth(selected_user, test_df, rating_threshold)
            all_test_items = get_all_test_items(selected_user, test_df)
            
            # Calculate hits
            hits = set(recommendations) & ground_truth
            precision = len(hits) / len(recommendations) if recommendations else 0
            recall = len(hits) / len(ground_truth) if ground_truth else 0
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("🎯 Hits", f"{len(hits)}/{len(recommendations)}")
            with metric_col2:
                st.metric("📊 Precision", f"{precision:.2%}")
            with metric_col3:
                st.metric("📈 Recall", f"{recall:.2%}")
            
            st.markdown("---")
            
            # Display recommendations, ground truth, and user history side by side
            rec_col, gt_col, history_col = st.columns(3)
            
            with rec_col:
                st.markdown("### 🎬 Predicted Recommendations")
                st.caption("Green = correctly predicted (in ground truth)")
                
                for i, movie_id in enumerate(recommendations, 1):
                    movie_title = get_movie_title(movie_id, movies_df) or f"Movie {movie_id}"
                    is_hit = movie_id in ground_truth
                    pred_score = scores.get(movie_id, 0) * 5
                    
                    if is_hit:
                        st.markdown(
                            f'<div class="hit-movie">✅ {i}. {movie_title} (score: {pred_score:.3f})</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="miss-movie">{i}. {movie_title} (score: {pred_score:.3f})</div>',
                            unsafe_allow_html=True
                        )
            
            with gt_col:
                st.markdown("### 📋 Ground Truth (Test Set)")
                st.caption(f"Items rated ≥ {rating_threshold} by user, sorted by rating")
                
                if ground_truth:
                    # Sort ground truth by rating (descending)
                    sorted_ground_truth = sorted(
                        ground_truth, 
                        key=lambda x: all_test_items.get(x, 0), 
                        reverse=True
                    )
                    for movie_id in sorted_ground_truth:
                        movie_title = get_movie_title(movie_id, movies_df) or f"Movie {movie_id}"
                        rating = all_test_items.get(movie_id, "N/A")
                        is_recommended = movie_id in recommendations
                        
                        if is_recommended:
                            st.markdown(
                                f'<div class="hit-movie">⭐ {movie_title} (rated {rating})</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="ground-truth">⭐ {movie_title} (rated {rating})</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No items rated ≥ threshold in test set for this user.")
            with history_col:
                st.markdown("### 📚 User's Watch History")
                st.caption("Movies rated in training set (sorted by rating)")
                
                # Get user's training history
                user_history = train_df[train_df['user_id'] == selected_user].copy()
                user_history = user_history.sort_values('rating', ascending=False)
                
                # Show top rated movies from history
                for _, row in user_history.head(20).iterrows():
                    movie_id = row['item_id']
                    rating = row['rating']
                    movie_title = get_movie_title(movie_id, movies_df) or f"Movie {movie_id}"
                    
                    st.markdown(
                        f'<div class="history-movie">⭐ {movie_title} ({int(rating)})</div>',
                        unsafe_allow_html=True
                    )
                
                if len(user_history) > 20:
                    st.caption(f"... and {len(user_history) - 20} more movies")
            # Show all test items
            with st.expander("📊 View All Test Items for This User"):
                test_items_df = test_df[test_df['user_id'] == selected_user].copy()
                test_items_df['movie_title'] = test_items_df['item_id'].apply(
                    lambda x: get_movie_title(x, movies_df) or f"Movie {x}"
                )
                test_items_df['recommended'] = test_items_df['item_id'].isin(recommendations)
                st.dataframe(
                    test_items_df[['item_id', 'movie_title', 'rating', 'recommended']]
                    .sort_values('rating', ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True
                )


if __name__ == "__main__":
    main()