"""Graph embeddings - mimics SimClusters and TwHIN"""
import math
import os
import numpy as np
from data_loader import get_data_loader

class SimClusters:
    """
    Simplified interest-based clustering (hash-based proxy, NOT actual SimClusters).
    
    Production SimClusters uses Metropolis-Hastings sampling on a bipartite
    follow graph to detect ~145K communities via Producer-Producer similarity.

    This mimic uses deterministic hash-based clustering on user interests:
    - hash(interest) % num_clusters -> cluster_id
    - Preserves the sparse embedding structure (1-3 active clusters per user)
    - Does NOT perform actual community detection or graph analysis
    
    Trade-off: Speed and simplicity over semantic accuracy.
    """
    def __init__(self, num_clusters=150):
        self.num_clusters = num_clusters
        self.data_loader = get_data_loader()
        self.tweet_clusters = {}
        self.user_clusters_cache = {}
    
    def get_user_embedding(self, user_id):
        """Compute sparse cluster embedding from user interests"""
        if user_id in self.user_clusters_cache:
            return self.user_clusters_cache[user_id]
        
        user = self.data_loader.get_user(user_id)
        if not user:
            return {0: 1.0}
        
        # Map interests to clusters (hash-based deterministic mapping)
        clusters = []
        scores = []
        for interest in user['interests']:
            cluster_id = hash(interest) % self.num_clusters
            clusters.append(cluster_id)
            # Score based on relative importance (could be learned)
            scores.append(1.0)
        
        # Normalize scores
        if scores:
            total = sum(scores)
            scores = [s / total for s in scores]
            result = {c: s for c, s in zip(clusters, scores)}
        else:
            result = {0: 1.0}
        
        self.user_clusters_cache[user_id] = result
        return result
    
    def get_tweet_embedding(self, tweet_id):
        """Get sparse cluster embedding for tweet based on author"""
        if tweet_id not in self.tweet_clusters:
            tweet = self.data_loader.get_tweet(tweet_id)
            if tweet:
                # Tweet inherits clusters from author
                author_emb = self.get_user_embedding(tweet['author_id'])
                # Add some noise to differentiate tweet from author
                self.tweet_clusters[tweet_id] = author_emb
            else:
                self.tweet_clusters[tweet_id] = {0: 1.0}
        return self.tweet_clusters[tweet_id]
    
    def cosine_similarity(self, user_id, tweet_id):
        """Compute similarity between user and tweet in cluster space"""
        user_emb = self.get_user_embedding(user_id)
        tweet_emb = self.get_tweet_embedding(tweet_id)
        
        # Sparse dot product
        common_clusters = set(user_emb.keys()) & set(tweet_emb.keys())
        if not common_clusters:
            return 0.0
        
        dot_product = sum(user_emb[c] * tweet_emb[c] for c in common_clusters)
        
        # Compute norms
        norm_user = math.sqrt(sum(s**2 for s in user_emb.values()))
        norm_tweet = math.sqrt(sum(s**2 for s in tweet_emb.values()))
        
        return dot_product / (norm_user * norm_tweet + 1e-8)

class TwHIN:
    """
    Two-tower embedding model (inspired by TwHIN, not identical architecture).
    
    Production TwHIN uses knowledge graph embeddings (TransE-style) trained on
    a heterogeneous graph of users, tweets, hashtags, and entities.
    See: https://github.com/twitter/the-algorithm-ml/blob/main/projects/twhin/README.md
    
    This mimic uses a simpler two-tower (dual encoder) architecture:
    - User Tower: user features -> 128-dim embedding
    - Tweet Tower: tweet features -> 128-dim embedding
    - Similarity: cosine distance in shared latent space
    
    The two-tower approach achieves similar goals (user-tweet similarity)
    but via metric learning rather than knowledge graph link prediction.
    """
    def __init__(self, dim=128, use_trained_model=True):
        self.dim = dim
        self.data_loader = get_data_loader()
        self.tweet_embeddings = {}
        self.user_embeddings = {}
        self.model = None
        
        if use_trained_model:
            self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained TwHIN PyTorch model"""
        model_path = 'models/twhin_model.pt'
        if os.path.exists(model_path):
            try:
                import torch
                from models.embedding_model import TwHINEmbeddingModel
                self.model = TwHINEmbeddingModel(user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                # ASCII-only status for Windows console compatibility
                print(f"[OK] Loaded trained TwHIN model from {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load TwHIN model: {e}. Using feature-based embeddings.")
    
    def _extract_user_features(self, user):
        """Extract features for user embedding"""
        return np.array([
            np.log1p(user['followers_count']),
            np.log1p(user['following_count']),
            user['account_age_days'] / 365.0,
            len(user['interests']) / 10.0,
            *([0.1] * 6)  # Padding
        ], dtype=np.float32)
    
    def _extract_tweet_features(self, tweet):
        """Extract features for tweet embedding"""
        return np.array([
            tweet['text_length'] / 280.0,
            1.0 if tweet['has_media'] else 0.0,
            len(tweet.get('hashtags', [])) / 5.0,
            len(tweet.get('mentions', [])) / 5.0,
            len([tweet['category']]) / 5.0,
            *([0.1] * 5)  # Padding
        ], dtype=np.float32)
    
    def get_user_embedding(self, user_id):
        """Compute dense embedding for user using trained model"""
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id]
        
        user = self.data_loader.get_user(user_id)
        if not user:
            return np.zeros(self.dim)
        
        if self.model is not None:
            # Use trained model
            features = self._extract_user_features(user)
            embedding = self.model.encode_users(features.reshape(1, -1))[0]
        else:
            # Fallback: simple feature-based embedding
            embedding = self._extract_user_features(user)
            # Pad to full dimension
            if len(embedding) < self.dim:
                embedding = np.pad(embedding, (0, self.dim - len(embedding)))
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        self.user_embeddings[user_id] = embedding
        return embedding
    
    def get_tweet_embedding(self, tweet_id):
        """Compute dense embedding for tweet using trained model"""
        if tweet_id in self.tweet_embeddings:
            return self.tweet_embeddings[tweet_id]
        
        tweet = self.data_loader.get_tweet(tweet_id)
        if not tweet:
            return np.zeros(self.dim)
        
        if self.model is not None:
            # Use trained model
            features = self._extract_tweet_features(tweet)
            embedding = self.model.encode_tweets(features.reshape(1, -1))[0]
        else:
            # Fallback: inherit from author with noise
            author_emb = self.get_user_embedding(tweet['author_id'])
            noise = np.random.randn(self.dim) * 0.1
            embedding = author_emb + noise
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        self.tweet_embeddings[tweet_id] = embedding
        return embedding
    
    def cosine_similarity(self, user_id, tweet_id):
        """Compute similarity between user and tweet"""
        user_emb = self.get_user_embedding(user_id)
        tweet_emb = self.get_tweet_embedding(tweet_id)
        
        dot = np.dot(user_emb, tweet_emb)
        norm_u = np.linalg.norm(user_emb)
        norm_t = np.linalg.norm(tweet_emb)
        
        return dot / (norm_u * norm_t + 1e-8)

class RealGraph:
    """
    User-user interaction prediction based on social graph (rule-based).
    
    Production RealGraph is a learned model predicting interaction likelihood
    between user pairs based on historical engagement patterns.
    See: src/scala/com/twitter/interaction_graph/README.md
    
    This mimic uses rule-based heuristics:
    - Follow weight from dataset (explicit signal)
    - Interest overlap via Jaccard similarity (fallback)
    - Neighborhood retrieval via follow relationship strength
    
    Simplification: No learned model, uses direct graph signals.
    """
    def __init__(self):
        self.data_loader = get_data_loader()
        self.interaction_cache = {}
    
    def predict_interaction(self, user_a, user_b):
        """Predict likelihood of user_a interacting with user_b's content"""
        # Check if they follow each other
        follow_weight = self.data_loader.get_follow_weight(user_a, user_b)
        if follow_weight > 0:
            return follow_weight
        
        # Compute interest similarity as fallback
        user_a_obj = self.data_loader.get_user(user_a)
        user_b_obj = self.data_loader.get_user(user_b)
        
        if user_a_obj and user_b_obj:
            a_interests = set(user_a_obj['interests'])
            b_interests = set(user_b_obj['interests'])
            if a_interests and b_interests:
                similarity = len(a_interests & b_interests) / len(a_interests | b_interests)
                return max(0.0, similarity * 0.5)
        
        return 0.0
    
    def get_neighborhood(self, user_id, k=10):
        """Get top-k users by follow relationship strength"""
        followed_users = self.data_loader.get_followed_users(user_id)
        
        # Get weights and sort
        user_weights = [
            (followed_id, self.data_loader.get_follow_weight(user_id, followed_id))
            for followed_id in followed_users
        ]
        user_weights.sort(key=lambda x: x[1], reverse=True)
        
        return [user_id for user_id, _ in user_weights[:k]]

class GraphFeatureService:
    """Compute graph-based features between users"""
    def __init__(self):
        self.data_loader = get_data_loader()
    
    def get_features(self, user_a, user_b):
        """Compute graph features between two users"""
        follow_weight = self.data_loader.get_follow_weight(user_a, user_b)
        follow_back_weight = self.data_loader.get_follow_weight(user_b, user_a)
        
        # Get user info
        user_a_obj = self.data_loader.get_user(user_a)
        user_b_obj = self.data_loader.get_user(user_b)
        
        # Compute mutual follows (simplified)
        a_follows = set(self.data_loader.get_followed_users(user_a))
        b_follows = set(self.data_loader.get_followed_users(user_b))
        mutual_follows = len(a_follows & b_follows)
        
        # Common interests
        if user_a_obj and user_b_obj:
            a_interests = set(user_a_obj['interests'])
            b_interests = set(user_b_obj['interests'])
            common_interests = len(a_interests & b_interests)
        else:
            common_interests = 0
        
        features = {
            'mutual_follows': mutual_follows,
            'follow_back_prob': follow_back_weight,
            'follows': 1.0 if follow_weight > 0 else 0.0,
            'common_interests_score': common_interests / 7.0,  # max 7 categories
        }
        return features
