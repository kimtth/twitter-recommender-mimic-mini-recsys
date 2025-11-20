"""
TwHIN two-tower embedding model

Trainable neural network for learning user and tweet embeddings.
Saved to `twhin_model.pt` after training.

Simplifications vs production:
    - Flat feature vectors (no graph meta-paths)
    - On-demand computation (no offline pre-computation or ANN index)
    - Basic contrastive loss (cosine similarity + BCE)

Note: `SimClusters` and `RealGraph` in `embeddings.py` are rule-based (not trained).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwHINEmbeddingModel(nn.Module):
    """
    Simplified two-tower embedding model for users and tweets.

    Production-inspired design notes:
        - Mimics core TwHIN pattern (separate user/item towers + cosine similarity)
        - Omits heterogeneous graph meta-path / feature fusion layers
        - Provides on-demand embedding computation suitable for this educational scale

    Towers:
        - User tower: maps user feature vector -> 128-dim embedding
        - Tweet tower: maps tweet feature vector -> 128-dim embedding

    Training objective:
        - BCEWithLogits over cosine similarity for (user, tweet) interaction pairs
        - Negative samples assumed provided in calling code via label=0 rows

    Returned values:
        - Normalized user embeddings
        - Normalized tweet embeddings
        - Raw cosine similarity scores (pre-sigmoid)
    """
    
    def __init__(self, user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Tweet tower
        self.tweet_tower = nn.Sequential(
            nn.Linear(tweet_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, user_features, tweet_features):
        """
        Args:
            user_features: [batch_size, user_feature_dim]
            tweet_features: [batch_size, tweet_feature_dim]
            
        Returns:
            user_embeddings, tweet_embeddings, similarity_scores
        """
        user_emb = self.user_tower(user_features)
        tweet_emb = self.tweet_tower(tweet_features)
        
        # Normalize embeddings
        user_emb = F.normalize(user_emb, p=2, dim=1)
        tweet_emb = F.normalize(tweet_emb, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(user_emb * tweet_emb, dim=1)
        
        return user_emb, tweet_emb, similarity
    
    def train_model(self, user_features, tweet_features, labels, epochs=10, lr=0.001, batch_size=64):
        """
        Train the embedding model
        
        Args:
            user_features: np.array [n_samples, user_feature_dim]
            tweet_features: np.array [n_samples, tweet_feature_dim]
            labels: np.array [n_samples] (1 for interaction, 0 for no interaction)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        X_user = torch.FloatTensor(user_features)
        X_tweet = torch.FloatTensor(tweet_features)
        y = torch.FloatTensor(labels)

        if X_user.shape[0] == 0:
            print("[WARN] No embedding training pairs provided. Skipping TwHIN training and using uninitialized model weights.")
            return
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_user), batch_size):
                batch_user = X_user[i:i+batch_size]
                batch_tweet = X_tweet[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                _, _, similarity = self.forward(batch_user, batch_tweet)
                
                # Loss: predict interaction from similarity
                loss = criterion(similarity, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 2 == 0:
                print(f"  TwHIN Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def encode_users(self, user_features):
        """Generate embeddings for users"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(user_features)
            embeddings = self.user_tower(X)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.numpy()
    
    def encode_tweets(self, tweet_features):
        """Generate embeddings for tweets"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(tweet_features)
            embeddings = self.tweet_tower(X)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.numpy()
