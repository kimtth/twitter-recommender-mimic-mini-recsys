"""Candidate sources — mimics Earlybird, UTEG, CR-Mixer, FRS.

Each source corresponds to a CandidatePipeline in Twitter's Product Mixer.
See: home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/
"""
from pipeline import CandidateWithDetails
from embeddings import SimClusters, TwHIN, RealGraph
from data_loader import get_data_loader

class CandidateSource:
    def __init__(self, name):
        self.name = name
    
    def fetch(self, query):
        raise NotImplementedError

class InNetworkSource(CandidateSource):
    """Earlybird search index — tweets from followed users.

    Source: home-mixer/.../ScoredTweetsInNetworkCandidatePipelineConfig.scala
    """
    def __init__(self):
        super().__init__("InNetwork")
        self.data_loader = get_data_loader()
    
    def fetch(self, query):
        # Get tweets from users that query.user_id follows
        followed_users = self.data_loader.get_followed_users(query.user_id)
        
        candidates = []
        for followed_id in followed_users[:50]:  # Limit to top 50 follows
            tweets = self.data_loader.get_user_tweets(followed_id)
            for tweet in tweets[:5]:  # Max 5 tweets per user
                candidate = CandidateWithDetails(tweet['tweet_id'], self.name)
                candidate.features['in_network'] = 1.0
                candidate.features['author_id'] = tweet['author_id']
                candidates.append(candidate)
                if len(candidates) >= 200:
                    break
            if len(candidates) >= 200:
                break
        
        return candidates[:200]

class OutOfNetworkSource(CandidateSource):
    """CR-Mixer — out-of-network discovery via SimClusters + TwHIN.

    Source: cr-mixer/server/src/main/scala/com/twitter/cr_mixer/
    Pipeline: home-mixer/.../ScoredTweetsTweetMixerCandidatePipelineConfig.scala
    """
    def __init__(self):
        super().__init__("OutOfNetwork")
        self.data_loader = get_data_loader()
        self.simclusters = SimClusters()
        self.twhin = TwHIN()
    
    def fetch(self, query):
        # Get recent tweets (not from followed users)
        followed_users = set(self.data_loader.get_followed_users(query.user_id))
        recent_tweets = self.data_loader.get_recent_tweets(max_hours=168, limit=500)
        
        # Filter to out-of-network tweets
        oon_tweets = [
            t for t in recent_tweets 
            if t['author_id'] not in followed_users and t['author_id'] != query.user_id
        ]
        
        candidates = []
        for tweet in oon_tweets[:300]:  # Sample 300 candidates
            candidate = CandidateWithDetails(tweet['tweet_id'], self.name)
            # Compute similarity using both embedding models
            candidate.features['simclusters_score'] = self.simclusters.cosine_similarity(
                query.user_id, tweet['tweet_id']
            )
            candidate.features['twhin_score'] = self.twhin.cosine_similarity(
                query.user_id, tweet['tweet_id']
            )
            candidate.features['author_id'] = tweet['author_id']
            candidates.append(candidate)
        
        # Rank by embedding similarity (approximate nearest neighbor)
        candidates.sort(key=lambda c: c.features.get('simclusters_score', 0), reverse=True)
        return candidates[:200]

class GraphSource(CandidateSource):
    """User Tweet Entity Graph (UTEG) — graph-based recommendations via RealGraph.

    Source: src/scala/com/twitter/recos/user_tweet_entity_graph/
    Pipeline: home-mixer/.../ScoredTweetsUtegCandidatePipelineConfig.scala
    """
    def __init__(self):
        super().__init__("Graph")
        self.data_loader = get_data_loader()
        self.real_graph = RealGraph()
    
    def fetch(self, query):
        # GraphJet traversal: find tweets from similar users in social graph
        similar_users = self.real_graph.get_neighborhood(query.user_id, k=20)
        
        # Get tweets from similar users' network
        candidates = []
        for similar_user in similar_users:
            tweets = self.data_loader.get_user_tweets(similar_user)
            for tweet in tweets[:10]:  # Max 10 tweets per similar user
                candidate = CandidateWithDetails(tweet['tweet_id'], self.name)
                candidate.features['author_similarity'] = self.real_graph.predict_interaction(
                    query.user_id, tweet['author_id']
                )
                candidate.features['author_id'] = tweet['author_id']
                candidates.append(candidate)
                if len(candidates) >= 100:
                    break
            if len(candidates) >= 100:
                break
        
        return candidates[:100]

class FollowRecommendationSource(CandidateSource):
    """Follow Recommendation Service (FRS) — user recommendations.

    Source: follow-recommendations-service/
    Pipeline: home-mixer/.../ScoredTweetsFrsCandidatePipelineConfig.scala
    """
    def __init__(self):
        super().__init__("FollowRecs")
        self.data_loader = get_data_loader()
        self.real_graph = RealGraph()
    
    def fetch(self, query):
        """Find users the requester may want to follow, then surface their tweets."""
        followed_users = set(self.data_loader.get_followed_users(query.user_id))
        # Score unconnected users by interest overlap
        scored_users = []
        for user in self.data_loader.users[:200]:  # sample pool
            uid = user['user_id']
            if uid == query.user_id or uid in followed_users:
                continue
            score = self.real_graph.predict_interaction(query.user_id, uid)
            if score > 0:
                scored_users.append((uid, score))
        scored_users.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for uid, _ in scored_users[:5]:
            tweets = self.data_loader.get_user_tweets(uid)
            for tweet in tweets[:3]:
                candidate = CandidateWithDetails(tweet['tweet_id'], self.name)
                candidate.features['author_id'] = tweet['author_id']
                candidates.append(candidate)
        return candidates
