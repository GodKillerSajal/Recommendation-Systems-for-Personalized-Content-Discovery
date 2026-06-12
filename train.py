# src/train.py
"""Model training and evaluation module.
Implements:
- Mapping of IDs to contiguous indices
- Alternating Least Squares (ALS) via implicit library
- Singular Value Decomposition (SVD) via surprise library
- Evaluation (RMSE, MAP@10)
- Generation of sample recommendations
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
import math

# ALS
from implicit.als import AlternatingLeastSquares

# SVD
from surprise import Dataset, Reader, SVD as SurpriseSVD
from surprise.model_selection import train_test_split as surprise_train_test_split

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
RAW_RATINGS = os.path.join(DATA_DIR, 'processed', 'ratings.parquet')

def load_ratings():
    df = pd.read_parquet(RAW_RATINGS)
    return df

def train_and_evaluate():
    print("Loading ratings data...")
    df = load_ratings()
    total_rows = len(df)
    print(f"Total ratings loaded: {total_rows}")

    # For computational efficiency on 16GB RAM, we sample 1,000,000 ratings for model building
    # This keeps runtime to seconds instead of hours while still providing significant data for learning.
    np.random.seed(42)
    sample_df = df.sample(n=min(1000000, total_rows), random_state=42).reset_index(drop=True)
    
    # Map raw IDs to contiguous indices
    user_ids = sample_df['user_id'].unique()
    movie_ids = sample_df['movie_id'].unique()
    
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    movie2idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx2movie = {i: mid for i, mid in enumerate(movie_ids)}
    
    sample_df['u_idx'] = sample_df['user_id'].map(user2idx)
    sample_df['m_idx'] = sample_df['movie_id'].map(movie2idx)
    
    # Save mappings
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    with open(os.path.join(DATA_DIR, 'processed', 'user2idx.json'), 'w') as f:
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    with open(os.path.join(DATA_DIR, 'processed', 'movie2idx.json'), 'w') as f:
        json.dump({str(k): v for k, v in movie2idx.items()}, f)
        
    # Split into train/test (80% train, 20% test)
    msk = np.random.rand(len(sample_df)) < 0.8
    train_df = sample_df[msk].reset_index(drop=True)
    test_df = sample_df[~msk].reset_index(drop=True)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # ------------------ ALS MODEL ------------------
    print("Training ALS model...")
    # Build sparse matrix (user x item) for ALS
    rows = train_df['u_idx'].values
    cols = train_df['m_idx'].values
    # Confidence weighting: 1 + rating
    vals = (1 + train_df['rating']).values
    
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    
    als = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=10, random_state=42)
    als.fit(matrix)
    
    # Save ALS factors
    os.makedirs(os.path.join(DATA_DIR, 'models'), exist_ok=True)
    with open(os.path.join(DATA_DIR, 'models', 'als_model.pkl'), 'wb') as f:
        pickle.dump(als, f)
        
    # ------------------ SVD MODEL ------------------
    print("Training SVD model...")
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
    surprise_train = surprise_data.build_full_trainset()
    
    svd = SurpriseSVD(n_factors=32, n_epochs=10, random_state=42)
    svd.fit(surprise_train)
    
    with open(os.path.join(DATA_DIR, 'models', 'svd_model.pkl'), 'wb') as f:
        pickle.dump(svd, f)
        
    # ------------------ EVALUATION ------------------
    print("Evaluating models...")
    # 1. RMSE
    # ALS Predictions
    als_preds = []
    for row in test_df.itertuples():
        u = row.u_idx
        i = row.m_idx
        pred = np.dot(als.user_factors[u], als.item_factors[i])
        # scale to 1-5
        pred = max(1.0, min(5.0, pred))
        als_preds.append(pred)
        
    rmse_als = math.sqrt(mean_squared_error(test_df['rating'], als_preds))
    
    # SVD Predictions
    svd_preds = []
    for row in test_df.itertuples():
        pred = svd.predict(row.user_id, row.movie_id).est
        svd_preds.append(pred)
        
    rmse_svd = math.sqrt(mean_squared_error(test_df['rating'], svd_preds))
    
    print(f"ALS RMSE: {rmse_als:.4f}")
    print(f"SVD RMSE: {rmse_svd:.4f}")
    
    # 2. MAP@10
    # Relevant if rating >= 3.5
    # Evaluate MAP@10 on a sample of 200 users to keep runtime fast
    print("Computing MAP@10...")
    ap_sum_als = 0
    ap_sum_svd = 0
    eval_users = test_df['user_id'].unique()[:200]
    user_count = 0
    
    for uid in eval_users:
        u_idx = user2idx[uid]
        # Get true relevance for the user in the test set
        user_test = test_df[test_df['user_id'] == uid]
        relevant_movies = set(user_test[user_test['rating'] >= 3.5]['movie_id'].values)
        if not relevant_movies:
            continue
            
        # ALS Top 10
        scores_als = als.user_factors[u_idx] @ als.item_factors.T
        top_10_idx_als = np.argsort(-scores_als)[:10]
        top_10_movies_als = [idx2movie[i] for i in top_10_idx_als]
        
        hits_als = 0
        sum_prec_als = 0
        for rank, mid in enumerate(top_10_movies_als, start=1):
            if mid in relevant_movies:
                hits_als += 1
                sum_prec_als += hits_als / rank
        ap_sum_als += sum_prec_als / min(len(relevant_movies), 10)
        
        # SVD Top 10
        all_movies = list(movie2idx.keys())
        scores_svd = [svd.predict(uid, mid).est for mid in all_movies]
        top_10_idx_svd = np.argsort(-np.array(scores_svd))[:10]
        top_10_movies_svd = [all_movies[i] for i in top_10_idx_svd]
        
        hits_svd = 0
        sum_prec_svd = 0
        for rank, mid in enumerate(top_10_movies_svd, start=1):
            if mid in relevant_movies:
                hits_svd += 1
                sum_prec_svd += hits_svd / rank
        ap_sum_svd += sum_prec_svd / min(len(relevant_movies), 10)
        
        user_count += 1
        
    map10_als = ap_sum_als / user_count if user_count else 0.0
    map10_svd = ap_sum_svd / user_count if user_count else 0.0
    
    print(f"ALS MAP@10: {map10_als:.4f}")
    print(f"SVD MAP@10: {map10_svd:.4f}")
    
    # Save results
    metrics = {
        'total_ratings': total_rows,
        'sampled_ratings': len(sample_df),
        'als_rmse': round(rmse_als, 4),
        'svd_rmse': round(rmse_svd, 4),
        'als_map10': round(map10_als, 4),
        'svd_map10': round(map10_svd, 4)
    }
    
    with open(os.path.join(DATA_DIR, 'processed', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Model training and evaluation successfully completed.")

if __name__ == "__main__":
    train_and_evaluate()
