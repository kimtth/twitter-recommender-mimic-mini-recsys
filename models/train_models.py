"""
Training script for all models
Trains models on synthetic dataset and saves checkpoints
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from models.engagement_model import EngagementModel
from models.safety_models import NSFWModel, ToxicityModel
from models.embedding_model import TwHINEmbeddingModel
from data_loader import get_data_loader


def extract_features_for_engagement(interaction, data_loader):
    """Extract features for engagement prediction"""
    user = data_loader.get_user(interaction['user_id'])
    tweet = data_loader.get_tweet(interaction['tweet_id'])
    
    # Skip if user or tweet not found
    if user is None or tweet is None:
        raise ValueError("User or tweet not found")
    
    author = data_loader.get_user(tweet.get('author_id'))
    if author is None:
        raise ValueError("Author not found")
    
    features = [
        # User features
        np.log1p(user.get('followers_count', 0)),
        np.log1p(user.get('following_count', 0)),
        user.get('account_age_days', 0) / 365.0,
        
        # Tweet features
        tweet.get('text_length', 0) / 280.0,
        1.0 if tweet.get('has_media', False) else 0.0,
        1.0 if tweet.get('has_link', False) else 0.0,
        tweet.get('hours_old', 0) / 168.0,  # normalize to weeks
        
        # Author features
        np.log1p(author.get('followers_count', 0)),
        1.0 if author.get('verified', False) else 0.0,
        
        # Interaction features (check if user follows author)
        1.0 if (interaction['user_id'], tweet.get('author_id')) in data_loader.follow_index else 0.0,
        
        # Temporal (use interaction timestamp, not tweet timestamp)
        (data_loader.max_timestamp - interaction.get('timestamp', data_loader.max_timestamp)) / 86400.0,  # days ago
        
        # Interest overlap (simplified)
        1.0 if tweet.get('category', '') in user.get('interests', []) else 0.0,
        
        # Quality indicators
        tweet.get('quality_score', 0.5),
        tweet.get('nsfw_score', 0.0),
        tweet.get('toxicity_score', 0.0),
        
        # Padding to 20 features
        0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    return features[:20]


def extract_features_for_safety(tweet):
    """Extract features for safety models (NSFW, Toxicity)"""
    features = [
        tweet.get('text_length', 0) / 280.0,
        1.0 if tweet.get('has_media', False) else 0.0,
        1.0 if tweet.get('has_link', False) else 0.0,
        tweet.get('hours_old', 0) / 168.0,  # normalize to weeks
        1.0 if tweet.get('category', '') == 'news' else 0.0,
        np.log1p(tweet.get('author_followers', 0)) / 10.0,
        1.0 if tweet.get('author_verified', False) else 0.0,
        np.random.random(),  # Placeholder for text embedding features
        np.random.random(),
        np.random.random()
    ]
    return features


def extract_features_for_embeddings(user=None, tweet=None):
    """Extract features for embedding model"""
    if user:
        return [
            np.log1p(user.get('followers_count', 0)),
            np.log1p(user.get('following_count', 0)),
            user.get('account_age_days', 0) / 365.0,
            len(user.get('interests', [])) / 10.0,
            1.0 if user.get('verified', False) else 0.0,
            user.get('avg_tweets_per_day', 0) / 10.0,
            *([0.1] * 4)  # Padding
        ]
    elif tweet:
        return [
            tweet.get('text_length', 0) / 280.0,
            1.0 if tweet.get('has_media', False) else 0.0,
            1.0 if tweet.get('has_link', False) else 0.0,
            tweet.get('hours_old', 0) / 168.0,
            tweet.get('quality_score', 0.5),
            np.log1p(tweet.get('author_followers', 0)) / 10.0,
            1.0 if tweet.get('author_verified', False) else 0.0,
            *([0.1] * 3)  # Padding
        ]


def train_engagement_model(data_loader):
    """Train engagement prediction model"""
    print("\n" + "="*70)
    print("Training Engagement Model (Multi-head Heavy Ranker)")
    print("="*70)
    if torch is None:
        print("[WARN] Skipping engagement model training (PyTorch missing).")
        return None
    
    # Prepare training data - ONLY use train split
    train_interactions = [i for i in data_loader.interactions if i.get('split') == 'train']
    
    features = []
    labels = []
    errors = []
    
    for interaction in train_interactions:
        try:
            feat = extract_features_for_engagement(interaction, data_loader)
            features.append(feat)
            labels.append(1.0 if interaction['interaction_type'] == 'like' else 0.5)
        except Exception as e:
            errors.append(str(e))
            continue
    
    if errors and len(features) == 0:
        print(f"[WARN] All {len(train_interactions)} interactions failed feature extraction. Sample errors:")
        for err in errors[:3]:
            print(f"  - {err}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Training on {len(features)} interactions")
    
    # Train model
    model = EngagementModel(input_dim=20)
    model.train_model(features, labels, epochs=10, lr=0.001, batch_size=128)
    
    # Save model
    torch.save(model.state_dict(), 'models/engagement_model.pt')
    print("[OK] Engagement model saved to models/engagement_model.pt")
    
    return model


def train_safety_models(data_loader):
    """Train NSFW and Toxicity models"""
    print("\n" + "="*70)
    print("Training Safety Models (NSFW & Toxicity)")
    print("="*70)
    if torch is None:
        print("[WARN] Skipping safety model training (PyTorch missing).")
        return None, None
    
    # Get tweets that appeared in train interactions only
    train_interactions = [i for i in data_loader.interactions if i.get('split') == 'train']
    train_tweet_ids = set(i['tweet_id'] for i in train_interactions)
    train_tweets = [t for t in data_loader.tweets if t['tweet_id'] in train_tweet_ids]
    
    # Prepare training data from tweets
    features = []
    nsfw_labels = []
    toxicity_labels = []
    
    for tweet in train_tweets:
        feat = extract_features_for_safety(tweet)
        features.append(feat)
        nsfw_labels.append(tweet.get('nsfw_score', 0.1))
        toxicity_labels.append(tweet.get('toxicity_score', 0.1))
    
    features = np.array(features)
    nsfw_labels = np.array(nsfw_labels)
    toxicity_labels = np.array(toxicity_labels)
    
    print(f"Training on {len(features)} tweets")
    
    # Train NSFW model
    print("\nTraining NSFW Model:")
    nsfw_model = NSFWModel(input_dim=10)
    nsfw_model.train_model(features, nsfw_labels, epochs=8, lr=0.001, batch_size=128)
    torch.save(nsfw_model.state_dict(), 'models/nsfw_model.pt')
    print("[OK] NSFW model saved to models/nsfw_model.pt")
    
    # Train Toxicity model
    print("\nTraining Toxicity Model:")
    toxicity_model = ToxicityModel(input_dim=10)
    toxicity_model.train_model(features, toxicity_labels, epochs=8, lr=0.001, batch_size=128)
    torch.save(toxicity_model.state_dict(), 'models/toxicity_model.pt')
    print("[OK] Toxicity model saved to models/toxicity_model.pt")
    
    return nsfw_model, toxicity_model


def train_embedding_model(data_loader):
    """Train TwHIN-style embedding model"""
    print("\n" + "="*70)
    print("Training TwHIN Embedding Model (Two-Tower)")
    print("="*70)
    if torch is None:
        print("[WARN] Skipping embedding model training (PyTorch missing).")
        return None
    
    # Prepare training data from train interactions only
    train_interactions = [i for i in data_loader.interactions if i.get('split') == 'train']
    
    user_features = []
    tweet_features = []
    labels = []
    
    for interaction in train_interactions:
        try:
            user = data_loader.get_user(interaction['user_id'])
            tweet = data_loader.get_tweet(interaction['tweet_id'])
            
            user_feat = extract_features_for_embeddings(user=user)
            tweet_feat = extract_features_for_embeddings(tweet=tweet)
            
            user_features.append(user_feat)
            tweet_features.append(tweet_feat)
            labels.append(1.0)  # Positive interaction
        except Exception:
            continue
    
    # Add negative samples (random user-tweet pairs)
    import random
    for _ in range(len(labels) // 2):
        try:
            user = random.choice(data_loader.users)
            tweet = random.choice(data_loader.tweets)

            user_feat = extract_features_for_embeddings(user=user)
            tweet_feat = extract_features_for_embeddings(tweet=tweet)

            user_features.append(user_feat)
            tweet_features.append(tweet_feat)
            labels.append(0.0)  # Negative sample
        except Exception:
            continue
    
    user_features = np.array(user_features)
    tweet_features = np.array(tweet_features)
    labels = np.array(labels)
    
    print(f"Training on {len(labels)} user-tweet pairs ({sum(labels)} positive)")
    
    # Train model
    model = TwHINEmbeddingModel(user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128)
    model.train_model(user_features, tweet_features, labels, epochs=10, lr=0.001, batch_size=128)
    
    # Save model
    torch.save(model.state_dict(), 'models/twhin_model.pt')
    print("[OK] TwHIN model saved to models/twhin_model.pt")
    
    return model


def main():
    print("="*70)
    print("Mini-RecSys Model Training")
    print("="*70)
    
    # Load data
    data_loader = get_data_loader()
    print(f"\n[OK] Dataset loaded: {len(data_loader.users)} users, {len(data_loader.tweets)} tweets, {len(data_loader.interactions)} interactions")
    
    # Create models directory if needed
    os.makedirs('models', exist_ok=True)
    
    # Train all models
    train_engagement_model(data_loader)
    train_safety_models(data_loader)
    train_embedding_model(data_loader)
    
    print("\n" + "="*70)
    print("[OK] All models trained successfully!")
    print("="*70)
    if torch is not None:
        print("\nSaved models:")
        print("  - models/engagement_model.pt")
        print("  - models/nsfw_model.pt")
        print("  - models/toxicity_model.pt")
        print("  - models/twhin_model.pt")
        print("\nThese models will be auto-loaded by the recommendation system.")
    else:
        print("\n[WARN] No models trained (PyTorch missing). System will use heuristic fallbacks.")
    print("="*70)


if __name__ == '__main__':
    main()
