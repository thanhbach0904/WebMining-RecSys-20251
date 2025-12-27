"""
Metrics for evaluating recommendation performance.
Includes implementations for RMSE, MAE, Precision@K, Recall@K, NDCG@K, and Catalog Coverage.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error #type: ignore

    
def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted ratings.
    
    Args:
        y_true (list[float]): Ground truth ratings.
        y_pred (list[float]): Predicted ratings.
        
    Returns:
        float: The RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted ratings.
    
    Args:
        y_true (list[float]): Ground truth ratings.
        y_pred (list[float]): Predicted ratings.
    
    Returns:
        float: The MAE value.
    """
    return mean_absolute_error(y_true, y_pred)
    
def precision_at_k(recommended, relevant, k=10):
    """
    Compute Precision at K.
    Measures the proportion of recommended items in the top-K that are relevant.
        
    Args:
        recommended (list): Ordered list of recommended item IDs.
        relevant (list): List of relevant item IDs (ground truth).
        k (int): Number of top recommendations to consider.
        
    Returns:
        float: Precision score [0, 1].
    """
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    if len(recommended_k) == 0:
        return 0.0
    
    hits = len(recommended_k & relevant_set)
    return hits / len(recommended_k)
    
def recall_at_k(recommended, relevant, k=10):
    """
    Compute Recall at K.
    Measures the proportion of relevant items that are successfully recommended in the top-K.
    """
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    if len(relevant_set) == 0:
        return 0.0
    
    hits = len(recommended_k & relevant_set)
    return hits / len(relevant_set)
    
def ndcg_at_k(recommended, relevant_with_ratings, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    Evaluates ranking quality, giving higher importance to top ranks.
        
    Args:
        recommended (list): Ordered list of recommended item IDs.
        relevant_with_ratings (dict): Mapping of item_id to true rating (relevance).
        k (int): Rank cutoff.
        
    Returns:
        float: NDCG score [0, 1].
    """
    # DCG: sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, item_id in enumerate(recommended[:k]):
        if item_id in relevant_with_ratings:
            relevance = relevant_with_ratings[item_id]
            dcg += relevance / np.log2(i + 2)  # i+2 because position starts at 1
    
    # IDCG: DCG of ideal ranking (sorted by relevance)
    ideal_relevances = sorted(relevant_with_ratings.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        idcg += relevance / np.log2(i + 2)
    
    # NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
    
def coverage(all_recommendations, total_items):
    """
    Calculate Catalog Coverage.
    The percentage of total available items that were recommended to at least one user.
    """
    unique_items = set()
    for recommendations in all_recommendations:
        unique_items.update(recommendations)
    
    return len(unique_items) / total_items
    
def evaluate_all(model, test_data, k=10):
    """
    Perform a comprehensive evaluation using the test set.
    
    Returns:
        dict: A dictionary containing all computed metrics.
    """
    users = test_data['user_id'].unique()
    
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    all_recommendations = []
    
    for user_id in users:
        # Get ground truth
        user_test_data = test_data[test_data['user_id'] == user_id]
        relevant_items = list(user_test_data['item_id'])
        relevant_with_ratings = dict(zip(user_test_data['item_id'], user_test_data['rating']))
        
        if len(relevant_items) == 0:
            continue
        
        try:
            # Get recommendations
            recommendations = model.recommend(user_id, top_k=k)
            all_recommendations.append(recommendations)
            
            # Calculate metrics
            precision_scores.append(precision_at_k(recommendations, relevant_items, k))
            recall_scores.append(recall_at_k(recommendations, relevant_items, k))
            ndcg_scores.append(ndcg_at_k(recommendations, relevant_with_ratings, k))
        except Exception as e:
            continue
    
    # Get total items from test data
    total_items = test_data['item_id'].nunique()
    
    return {
        f'Precision@{k}': np.mean(precision_scores) if precision_scores else 0.0,
        f'Recall@{k}': np.mean(recall_scores) if recall_scores else 0.0,
        f'NDCG@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'Coverage': coverage(all_recommendations, total_items) if all_recommendations else 0.0
    }

if __name__ == "__main__":

    y_true = [0,1,2]
    y_pred = [0,2,1]
    print(rmse(y_true, y_pred))