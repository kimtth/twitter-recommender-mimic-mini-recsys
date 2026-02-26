"""Scoring and ranking — mimics the Heavy Ranker served by Navi.

Heavy Ranker source:
  home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/scoring_pipeline/
Navi model server:
  navi/
"""
import math
import os
import numpy as np
from data_loader import get_data_loader

class Scorer:
    def __init__(self, name):
        self.name = name
    
    def score(self, query, candidates):
        raise NotImplementedError

class MLScorer(Scorer):
    """Simulates Heavy Ranker with ~6000 features and multiple prediction heads"""
    def __init__(self, use_trained_model=True):
        super().__init__("MLScorer")
        self.data_loader = get_data_loader()
        self.use_trained_model = use_trained_model
        self.engagement_model = None
        
        self.source_weights = {
            "InNetwork": 0.8,
            "OutOfNetwork": 0.5,
            "Graph": 0.6,
            "FollowRecs": 0.4,
        }
        # Simulate learned model weights for different features
        self.feature_weights = {
            'simclusters_score': 0.25,
            'twhin_score': 0.20,
            'author_similarity': 0.15,
            'engagement_prob': 0.30,
            'follow_prob': 0.10,
        }
        
        # Load trained PyTorch model if available
        if use_trained_model:
            self._load_trained_model()
        else:
            self._train_engagement_model()
    
    def _load_trained_model(self):
        """Load trained PyTorch engagement model"""
        # Always compute engagement baseline for fallback scoring
        self._compute_engagement_baseline()
        
        model_path = 'models/engagement_model.pt'
        if os.path.exists(model_path):
            try:
                import torch
                from models.engagement_model import EngagementModel
                self.engagement_model = EngagementModel(input_dim=20)
                self.engagement_model.load_state_dict(torch.load(model_path))
                self.engagement_model.eval()
                print(f"[OK] Loaded trained engagement model from {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load trained model: {e}. Using dataset scores.")
        else:
            print(f"[WARN] Trained model not found at {model_path}. Using dataset scores.")
    
    def _compute_engagement_baseline(self):
        """Compute engagement baseline from historical interactions"""
        interactions = self.data_loader.interactions
        if not interactions:
            self.engagement_baseline = 0.1
            return
        
        # Calculate average engagement probability from dataset
        engagement_probs = [inter['engagement_prob'] for inter in interactions]
        self.engagement_baseline = np.mean(engagement_probs) if engagement_probs else 0.1
    
    def _train_engagement_model(self):
        """Learn engagement patterns from historical interactions (fallback)"""
        self._compute_engagement_baseline()
    
    def _extract_model_features(self, query, candidate, tweet, author, user):
        """Extract 20 features for PyTorch model (matches train_models.py feature extraction)"""
        follow_weight = self.data_loader.get_follow_weight(query.user_id, tweet.get('author_id'))
        
        features = [
            # User features
            np.log1p(user.get('followers_count', 0)) if user else 0.0,
            np.log1p(user.get('following_count', 0)) if user else 0.0,
            user.get('account_age_days', 0) / 365.0 if user else 0.0,
            
            # Tweet features
            tweet.get('text_length', 0) / 280.0,
            1.0 if tweet.get('has_media', False) else 0.0,
            1.0 if tweet.get('has_link', False) else 0.0,
            tweet.get('hours_old', 0) / 168.0,  # normalize to weeks
            
            # Author features
            np.log1p(author.get('followers_count', 0)) if author else 0.0,
            1.0 if author.get('verified', False) else 0.0 if author else 0.0,
            
            # Interaction features
            1.0 if follow_weight > 0 else 0.0,
            
            # Temporal (placeholder - would need interaction timestamp)
            0.0,
            
            # Interest overlap
            1.0 if user and tweet.get('category', '') in user.get('interests', []) else 0.0,
            
            # Quality indicators
            tweet.get('quality_score', 0.5),
            tweet.get('nsfw_score', 0.0),
            tweet.get('toxicity_score', 0.0),
            
            # Padding to 20 features
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        return features[:20]
    
    def compute_features(self, query, candidate):
        """Extract features from candidate using dataset"""
        features = candidate.features.copy()
        
        # Get tweet data
        tweet = self.data_loader.get_tweet(candidate.id)
        if not tweet:
            return features
        
        # Content features from dataset
        features['has_media'] = 1 if tweet['has_media'] else 0
        features['has_link'] = 1 if tweet['has_link'] else 0
        features['text_length'] = tweet['text_length']
        features['quality_score'] = tweet['quality_score']
        
        # Temporal features
        features['recency_hours'] = tweet['hours_old']
        features['recency_score'] = math.exp(-tweet['hours_old'] / 24.0)
        
        # Author features
        author_id = tweet['author_id']
        author = self.data_loader.get_user(author_id)
        if author:
            features['author_followers'] = author['followers_count']
            features['author_verified'] = 1 if author['verified'] else 0
            features['author_tweets_per_day'] = author['avg_tweets_per_day']
        
        # Relationship features
        follow_weight = self.data_loader.get_follow_weight(query.user_id, author_id)
        features['follows_author'] = 1 if follow_weight > 0 else 0
        features['follow_weight'] = follow_weight
        
        # Interest match features
        user = self.data_loader.get_user(query.user_id)
        if user and author:
            user_interests = set(user['interests'])
            tweet_category = tweet['category']
            features['interest_match'] = 1 if tweet_category in user_interests else 0
        
        # Predicted engagement from dataset patterns
        base_engagement = self.engagement_baseline
        if follow_weight > 0:
            base_engagement += 0.4 * follow_weight
        if features.get('interest_match'):
            base_engagement += 0.3
        base_engagement += 0.2 * tweet['quality_score']
        base_engagement += 0.15 * features.get('has_media', 0)
        features['engagement_prob'] = min(0.95, base_engagement)
        
        # Follow probability (for out-of-network)
        if follow_weight == 0:
            simcluster_sim = features.get('simclusters_score', 0)
            features['follow_prob'] = simcluster_sim * 0.3
        else:
            features['follow_prob'] = 0.0
        
        return features
    
    def score(self, query, candidates):
        for candidate in candidates:
            # Compute all features
            features = self.compute_features(query, candidate)
            candidate.features.update(features)
            
            # Use trained PyTorch model if available
            if self.engagement_model is not None:
                tweet = self.data_loader.get_tweet(candidate.id)
                if tweet:
                    author = self.data_loader.get_user(tweet['author_id'])
                    user = self.data_loader.get_user(query.user_id)
                    
                    # Extract features for model
                    model_features = self._extract_model_features(query, candidate, tweet, author, user)
                    model_features_array = np.array([model_features], dtype=np.float32)
                    
                    # Get predictions from trained model
                    predictions = self.engagement_model.predict(model_features_array)
                    engagement_score = float(predictions['engagement'][0])
                    follow_score = float(predictions['follow'][0])
                else:
                    engagement_score = features.get('engagement_prob', 0.5)
                    follow_score = features.get('follow_prob', 0.3)
            else:
                # Fallback to dataset scores
                engagement_score = features.get('engagement_prob', 0.5)
                if features.get('has_media'):
                    engagement_score *= 1.2
                if features.get('is_trending'):
                    engagement_score *= 1.1
                follow_score = features.get('follow_prob', 0.3)
            
            # Head 3: Embedding similarity
            embedding_score = (
                features.get('simclusters_score', 0) * self.feature_weights['simclusters_score'] +
                features.get('twhin_score', 0) * self.feature_weights['twhin_score'] +
                features.get('author_similarity', 0) * self.feature_weights['author_similarity']
            )
            
            # Combine predictions (weighted by learned coefficients)
            base_score = self.source_weights.get(candidate.source, 0.5)
            combined_score = (
                0.5 * engagement_score +
                0.2 * follow_score +
                0.3 * embedding_score
            ) * base_score
            
            # Apply calibration
            candidate.score = self._calibrate(combined_score)
        
        return candidates
    
    def _calibrate(self, score):
        """Apply sigmoid calibration"""
        return 1.0 / (1.0 + math.exp(-5 * (score - 0.5)))

class DiversityScorer(Scorer):
    """Penalizes source repetition (InNetwork/OutOfNetwork/Graph) to diversify the result set."""
    def __init__(self):
        super().__init__("Diversity")
    
    def score(self, query, candidates):
        seen_sources = {}
        for candidate in candidates:
            # Penalize repeated sources
            count = seen_sources.get(candidate.source, 0)
            penalty = 0.9 ** count
            candidate.score *= penalty
            seen_sources[candidate.source] = count + 1
        return candidates

class RecencyScorer(Scorer):
    """Time-decay boost — newer tweets rank higher."""
    def __init__(self):
        super().__init__("Recency")
        self.data_loader = get_data_loader()
    
    def score(self, query, candidates):
        for candidate in candidates:
            tweet = self.data_loader.get_tweet(candidate.id)
            if tweet:
                hours_old = tweet.get('hours_old', 168)
                recency_boost = math.exp(-hours_old / 72.0)  # 72h half-life
            else:
                recency_boost = 0.5
            candidate.score *= (0.8 + 0.4 * recency_boost)
        return candidates
