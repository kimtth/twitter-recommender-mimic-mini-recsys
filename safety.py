"""Trust and safety models — content filtering.

Mimics trust_and_safety_models/ from twitter/the-algorithm.
Production uses Twitter-BERT + vision models; this proxy uses
metadata MLP classifiers on synthetic features.
"""
import os
import numpy as np
from data_loader import get_data_loader

class NSFWModel:
    """Detects NSFW content using vision model"""
    def __init__(self, use_trained_model=True):
        self.data_loader = get_data_loader()
        self.model = None
        
        if use_trained_model:
            self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained PyTorch NSFW model"""
        model_path = 'models/nsfw_model.pt'
        if os.path.exists(model_path):
            try:
                import torch
                from models.safety_models import NSFWModel as NSFWModelPyTorch
                self.model = NSFWModelPyTorch(input_dim=10)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"[OK] Loaded trained NSFW model from {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load NSFW model: {e}. Using dataset scores.")
    
    def _extract_features(self, tweet):
        """Extract features for NSFW model (matches train_models.py)"""
        return [
            tweet.get('text_length', 0) / 280.0,
            1.0 if tweet.get('has_media', False) else 0.0,
            1.0 if tweet.get('has_link', False) else 0.0,
            tweet.get('hours_old', 0) / 168.0,  # normalize to weeks
            1.0 if tweet.get('category', '') == 'news' else 0.0,
            np.log1p(tweet.get('author_followers', 0)) / 10.0,
            1.0 if tweet.get('author_verified', False) else 0.0,
            tweet.get('quality_score', 0.5),      # deterministic placeholder
            tweet.get('nsfw_score', 0.0),          # deterministic placeholder
            tweet.get('toxicity_score', 0.0),      # deterministic placeholder
        ]
    
    def predict(self, candidate):
        """Returns probability of NSFW content"""
        tweet = self.data_loader.get_tweet(candidate.id)
        if not tweet:
            return 0.0
        
        # Use trained model if available
        if self.model is not None:
            features = self._extract_features(tweet)
            features_array = np.array([features], dtype=np.float32)
            score = self.model.predict(features_array)
            return float(score[0])
        
        # Fallback to dataset score
        return tweet['nsfw_score']

class ToxicityModel:
    """Detects toxic/abusive content using NLP model"""
    def __init__(self, use_trained_model=True):
        self.data_loader = get_data_loader()
        self.model = None
        
        if use_trained_model:
            self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained PyTorch toxicity model"""
        model_path = 'models/toxicity_model.pt'
        if os.path.exists(model_path):
            try:
                import torch
                from models.safety_models import ToxicityModel as ToxicityModelPyTorch
                self.model = ToxicityModelPyTorch(input_dim=10)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"[OK] Loaded trained Toxicity model from {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load Toxicity model: {e}. Using dataset scores.")
    
    def _extract_features(self, tweet):
        """Extract features for toxicity model (matches train_models.py)"""
        return [
            tweet.get('text_length', 0) / 280.0,
            1.0 if tweet.get('has_media', False) else 0.0,
            1.0 if tweet.get('has_link', False) else 0.0,
            tweet.get('hours_old', 0) / 168.0,  # normalize to weeks
            1.0 if tweet.get('category', '') == 'news' else 0.0,
            np.log1p(tweet.get('author_followers', 0)) / 10.0,
            1.0 if tweet.get('author_verified', False) else 0.0,
            tweet.get('quality_score', 0.5),      # deterministic placeholder
            tweet.get('nsfw_score', 0.0),          # deterministic placeholder
            tweet.get('toxicity_score', 0.0),      # deterministic placeholder
        ]
    
    def predict(self, candidate):
        """Returns toxicity score"""
        tweet = self.data_loader.get_tweet(candidate.id)
        if not tweet:
            return 0.0
        
        # Use trained model if available
        if self.model is not None:
            features = self._extract_features(tweet)
            features_array = np.array([features], dtype=np.float32)
            score = self.model.predict(features_array)
            return float(score[0])
        
        # Fallback to dataset score
        return tweet['toxicity_score']

class SafetyScorer:
    """Combined safety scoring"""
    def __init__(self):
        self.nsfw_model = NSFWModel()
        self.toxicity_model = ToxicityModel()
        self.nsfw_threshold = 0.8
        self.toxicity_threshold = 0.7
    
    def score(self, query, candidates):
        """Add safety scores and penalize unsafe content"""
        for candidate in candidates:
            nsfw_score = self.nsfw_model.predict(candidate)
            toxicity_score = self.toxicity_model.predict(candidate)
            
            candidate.features['nsfw_score'] = nsfw_score
            candidate.features['toxicity_score'] = toxicity_score
            
            # Heavy penalty for unsafe content
            if nsfw_score > self.nsfw_threshold:
                candidate.score *= 0.1
            if toxicity_score > self.toxicity_threshold:
                candidate.score *= 0.1
            
            # Moderate penalty for borderline content
            elif nsfw_score > 0.5:
                candidate.score *= 0.7
            elif toxicity_score > 0.5:
                candidate.score *= 0.8
        
        return candidates
