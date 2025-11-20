"""Generate synthetic dataset mimicking Twitter's interaction data"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

np.random.seed(42)

def generate_users(n_users=1000):
    """Generate synthetic user profiles"""
    users = []
    for user_id in range(1, n_users + 1):
        # Power law distribution for followers (few users have many followers)
        followers = int(np.random.pareto(1.5) * 100 + 10)
        following = int(np.random.pareto(2.0) * 80 + 5)
        
        # User categories affect their content preferences
        categories = ['tech', 'sports', 'news', 'entertainment', 'politics', 'art', 'science']
        interests = np.random.choice(categories, size=np.random.randint(1, 4), replace=False).tolist()
        
        users.append({
            'user_id': user_id,
            'followers_count': followers,
            'following_count': following,
            'account_age_days': np.random.randint(30, 3650),
            'verified': np.random.random() < 0.05,
            'interests': interests,
            'avg_tweets_per_day': max(0.1, np.random.exponential(2.0)),
        })
    return users

def generate_follow_graph(users, avg_follows=20):
    """Generate follow relationships (social graph)"""
    follows = []
    for user in users:
        n_follows = min(int(np.random.poisson(avg_follows)), len(users) - 1)
        # Homophily: users follow others with similar interests
        same_interest_users = [
            u for u in users 
            if u['user_id'] != user['user_id'] and 
            any(interest in u['interests'] for interest in user['interests'])
        ]
        other_users = [u for u in users if u not in same_interest_users and u['user_id'] != user['user_id']]
        
        # 70% same interest, 30% random
        n_similar = int(n_follows * 0.7)
        n_random = n_follows - n_similar
        
        follows_similar = np.random.choice(
            [u['user_id'] for u in same_interest_users], 
            size=min(n_similar, len(same_interest_users)), 
            replace=False
        ).tolist() if same_interest_users else []
        
        follows_random = np.random.choice(
            [u['user_id'] for u in other_users],
            size=min(n_random, len(other_users)),
            replace=False
        ).tolist() if other_users else []
        
        for followed_id in follows_similar + follows_random:
            follows.append({
                'follower_id': user['user_id'],
                'followed_id': followed_id,
                'follow_weight': np.random.beta(2, 5)  # interaction strength
            })
    return follows

def generate_tweets(users, n_tweets=10000):
    """Generate synthetic tweets"""
    tweets = []
    
    for tweet_id in range(1, n_tweets + 1):
        # Select author (power law: popular users tweet more)
        author_weights = np.array([u['avg_tweets_per_day'] for u in users])
        author_weights = author_weights / author_weights.sum()
        author = np.random.choice(users, p=author_weights)
        
        # Tweet characteristics
        category = np.random.choice(author['interests'])
        has_media = np.random.random() < 0.3
        has_link = np.random.random() < 0.4
        text_length = int(np.random.gamma(5, 30))  # Gamma distribution for text length
        
        # Content quality affects engagement
        quality_score = np.random.beta(2, 5)
        
        # NSFW and toxicity (rare)
        nsfw_score = np.random.beta(1, 20)  # Most tweets are safe
        toxicity_score = np.random.beta(1, 15)
        
        # Recency (tweets from last 7 days)
        hours_old = np.random.exponential(12)  # Recent tweets more common
        
        tweets.append({
            'tweet_id': tweet_id,
            'author_id': author['user_id'],
            'category': category,
            'text_length': min(text_length, 280),
            'has_media': has_media,
            'has_link': has_link,
            'quality_score': quality_score,
            'nsfw_score': nsfw_score,
            'toxicity_score': toxicity_score,
            'hours_old': hours_old,
            'author_followers': author['followers_count'],
            'author_verified': author['verified'],
        })
    return tweets

def generate_interactions(users, tweets, follows, n_interactions=50000):
    """
    Generate user-tweet interactions (likes, retweets, replies)
    
    Temporal split: First 80% for training, last 20% for testing
    This ensures proper train/test separation for model evaluation
    """
    interactions = []
    follow_dict = {(f['follower_id'], f['followed_id']): f['follow_weight'] 
                   for f in follows}
    
    # Add timestamp to simulate temporal ordering
    current_timestamp = 1700000000  # Unix timestamp (Nov 2023)
    
    for _ in range(n_interactions):
        # Select user (active users interact more)
        user = np.random.choice(users)
        
        # Select tweet (recent tweets + followed authors + interest match)
        tweet = np.random.choice(tweets)
        
        # Calculate engagement probability based on multiple factors
        engagement_prob = 0.1  # base rate
        
        # In-network boost
        if (user['user_id'], tweet['author_id']) in follow_dict:
            engagement_prob += 0.4 * follow_dict[(user['user_id'], tweet['author_id'])]
        
        # Interest match boost
        if tweet['category'] in user['interests']:
            engagement_prob += 0.3
        
        # Quality boost
        engagement_prob += 0.2 * tweet['quality_score']
        
        # Recency boost (decay over time)
        recency_boost = np.exp(-tweet['hours_old'] / 24.0) * 0.2
        engagement_prob += recency_boost
        
        # Media boost
        if tweet['has_media']:
            engagement_prob += 0.15
        
        # Author popularity boost (diminishing returns)
        popularity_boost = min(0.15, np.log1p(tweet['author_followers']) / 50)
        engagement_prob += popularity_boost
        
        # Clip to valid probability
        engagement_prob = min(0.95, max(0.01, engagement_prob))
        
        # Decide if interaction happens
        if np.random.random() < engagement_prob:
            interaction_type = np.random.choice(
                ['like', 'retweet', 'reply', 'click'],
                p=[0.6, 0.2, 0.1, 0.1]
            )
            
            # Add timestamp (increment for temporal ordering)
            current_timestamp += np.random.randint(60, 3600)  # 1-60 min apart
            
            interactions.append({
                'user_id': user['user_id'],
                'tweet_id': tweet['tweet_id'],
                'interaction_type': interaction_type,
                'engagement_prob': engagement_prob,
                'timestamp': current_timestamp,
            })
    
    # Sort by timestamp to ensure temporal ordering
    interactions.sort(key=lambda x: x['timestamp'])
    
    # Mark split point (80% train, 20% test)
    split_idx = int(len(interactions) * 0.8)
    for i, inter in enumerate(interactions):
        inter['split'] = 'train' if i < split_idx else 'test'
    
    return interactions

def main():
    print("Generating synthetic dataset...")
    
    # Generate components
    print("  - Users...")
    users = generate_users(n_users=1000)
    
    print("  - Follow graph...")
    follows = generate_follow_graph(users, avg_follows=30)
    
    print("  - Tweets...")
    tweets = generate_tweets(users, n_tweets=10000)
    
    print("  - Interactions...")
    interactions = generate_interactions(users, tweets, follows, n_interactions=50000)
    
    # Save to files (Parquet format for efficient storage)
    # Note: Embeddings are computed on-the-fly by trained models, not pre-generated
    print("\nSaving to Parquet files...")
    pd.DataFrame(users).to_parquet('data/users.parquet', compression='snappy', index=False)
    pd.DataFrame(follows).to_parquet('data/follows.parquet', compression='snappy', index=False)
    pd.DataFrame(tweets).to_parquet('data/tweets.parquet', compression='snappy', index=False)
    pd.DataFrame(interactions).to_parquet('data/interactions.parquet', compression='snappy', index=False)
    
    # Print statistics
    # Split statistics
    train_interactions = [i for i in interactions if i['split'] == 'train']
    test_interactions = [i for i in interactions if i['split'] == 'test']
    
    print("\nDataset Statistics:")
    print(f"  Users: {len(users)}")
    print(f"  Follows: {len(follows)}")
    print(f"  Tweets: {len(tweets)}")
    print(f"  Interactions: {len(interactions)}")
    print(f"    - Train: {len(train_interactions)} ({len(train_interactions)/len(interactions)*100:.1f}%)")
    print(f"    - Test: {len(test_interactions)} ({len(test_interactions)/len(interactions)*100:.1f}%)")
    print(f"  Avg followers per user: {np.mean([u['followers_count'] for u in users]):.1f}")
    print(f"  Avg follows per user: {len(follows) / len(users):.1f}")
    print(f"  Interaction rate: {len(interactions) / (len(users) * len(tweets)) * 100:.3f}%")
    
    # Compute engagement statistics
    interaction_counts = {}
    for inter in interactions:
        tweet_id = inter['tweet_id']
        interaction_counts[tweet_id] = interaction_counts.get(tweet_id, 0) + 1
    
    print(f"  Avg interactions per tweet: {np.mean(list(interaction_counts.values())):.2f}")
    print(f"  Max interactions per tweet: {max(interaction_counts.values())}")
    
    print("\n[OK] Dataset generation complete!")
    print("  Files saved in data/ directory")

if __name__ == "__main__":
    import os
    # Create data directory in parent folder (mini-recsys/data/)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    # Change to parent directory so data/ path works correctly
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    main()
