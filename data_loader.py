"""Load and access synthetic dataset.

Mimics data stores backing Twitter's recommendation pipeline:
- Tweetypie (tweet metadata)
- Manhattan (key-value user data)
- Interaction Graph (follow/engagement edges)

See: tweetypie/, src/scala/com/twitter/interaction_graph/
"""
import os
import numpy as np
import pandas as pd

class DataLoader:
    """Singleton data loader for synthetic dataset"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance
    
    def _load_data(self):
        """Load all dataset files from Parquet format"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        # Pre-initialize attributes to avoid partially constructed state on failure
        self.users = []
        self.follows = []
        self.tweets = []
        self.interactions = []
        self.user_index = {}
        self.tweet_index = {}
        self.follow_index = {}
        self.user_interactions = {}
        self.max_timestamp = 0

        if not os.path.exists(data_dir):
            # Use ASCII-only output for Windows console compatibility
            print("[WARN] Dataset not found. Run 'python prep/generate_dataset.py' first.")
            return

        try:
            self.users = pd.read_parquet(os.path.join(data_dir, 'users.parquet')).to_dict('records')
            self.follows = pd.read_parquet(os.path.join(data_dir, 'follows.parquet')).to_dict('records')
            self.tweets = pd.read_parquet(os.path.join(data_dir, 'tweets.parquet')).to_dict('records')
            self.interactions = pd.read_parquet(os.path.join(data_dir, 'interactions.parquet')).to_dict('records')
            self._build_indices()
            print(f"[OK] Dataset loaded: {len(self.users)} users, {len(self.tweets)} tweets, {len(self.interactions)} interactions")
        except FileNotFoundError:
            print("[WARN] Dataset files incomplete. Run 'python prep/generate_dataset.py' first.")
        except Exception as e:
            # Handle missing parquet engine or other read errors gracefully
            print(f"[WARN] Dataset load failed ({e}). Continuing with empty dataset.")
    
    def _create_empty_data(self):
        """Create empty data structures"""
        self.users = []
        self.follows = []
        self.tweets = []
        self.interactions = []
        self.user_index = {}
        self.tweet_index = {}
        self.follow_index = {}
        self.user_interactions = {}
    
    def _build_indices(self):
        """Build lookup indices for fast access"""
        # User index
        self.user_index = {u['user_id']: u for u in self.users}
        
        # Tweet index
        self.tweet_index = {t['tweet_id']: t for t in self.tweets}
        
        # Follow index: (follower, followed) -> weight
        self.follow_index = {
            (f['follower_id'], f['followed_id']): f['follow_weight']
            for f in self.follows
        }
        
        # Follower -> [followed_ids] for O(1) get_followed_users
        self.follower_index = {}
        for f in self.follows:
            fid = f['follower_id']
            if fid not in self.follower_index:
                self.follower_index[fid] = []
            self.follower_index[fid].append(f['followed_id'])
        
        # User interactions for history
        self.user_interactions = {}
        for inter in self.interactions:
            user_id = inter['user_id']
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = []
            self.user_interactions[user_id].append(inter)
        
        # Compute max timestamp for temporal features (from interactions, not tweets)
        if self.interactions:
            self.max_timestamp = max(i.get('timestamp', 0) for i in self.interactions)
        else:
            self.max_timestamp = 0
    
    def get_user(self, user_id):
        """Get user by ID"""
        return self.user_index.get(user_id)
    
    def get_tweet(self, tweet_id):
        """Get tweet by ID"""
        return self.tweet_index.get(tweet_id)
    
    def get_follow_weight(self, follower_id, followed_id):
        """Get follow weight between users"""
        return self.follow_index.get((follower_id, followed_id), 0.0)
    
    def get_followed_users(self, user_id):
        """Get list of users followed by user_id"""
        return self.follower_index.get(user_id, [])
    
    def get_user_tweets(self, user_id):
        """Get tweets authored by user"""
        return [t for t in self.tweets if t['author_id'] == user_id]
    

    def get_user_interactions(self, user_id):
        """Get interaction history for user"""
        return self.user_interactions.get(user_id, [])
    
    def get_tweets_by_category(self, category, limit=100):
        """Get tweets in a specific category"""
        matching = [t for t in self.tweets if t['category'] == category]
        return matching[:limit]
    
    def get_recent_tweets(self, max_hours=168, limit=1000):
        """Get recent tweets (within max_hours)"""
        recent = [t for t in self.tweets if t['hours_old'] <= max_hours]
        # Sort by recency
        recent.sort(key=lambda t: t['hours_old'])
        return recent[:limit]
    
# Singleton instance
_data_loader = None

def get_data_loader():
    """Get singleton data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
