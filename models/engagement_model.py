"""
Multi-head engagement prediction model
Mimics Twitter's Heavy Ranker architecture
"""
import torch
import torch.nn as nn
import numpy as np


class EngagementModel(nn.Module):
    """
    Multi-head neural network for engagement prediction
    
    Architecture mirrors Twitter's Heavy Ranker:
    - Shared feature extraction layers
    - Multiple prediction heads (engagement, follow, embedding similarity)
    - Fusion layer for final score
    """
    
    def __init__(self, input_dim=20, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        
        # Shared feature extraction tower
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.shared_tower = nn.Sequential(*layers)
        
        # Engagement head (likes, retweets, replies)
        self.engagement_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Follow head (follow probability)
        self.follow_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Embedding similarity head
        self.embedding_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.2, 0.2]))
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] tensor of features
            
        Returns:
            dict with keys: engagement, follow, embedding, final_score
        """
        # Shared feature extraction
        shared = self.shared_tower(features)
        
        # Multi-head predictions
        engagement = self.engagement_head(shared).squeeze(-1)
        follow = self.follow_head(shared).squeeze(-1)
        embedding = self.embedding_head(shared).squeeze(-1)
        
        # Fusion (weighted combination)
        weights = torch.softmax(self.fusion_weights, dim=0)
        final_score = (
            weights[0] * engagement +
            weights[1] * follow +
            weights[2] * embedding
        )
        
        return {
            'engagement': engagement,
            'follow': follow,
            'embedding': embedding,
            'final_score': final_score
        }
    
    def train_model(self, train_features, train_labels, epochs=10, lr=0.001, batch_size=64):
        """
        Train the model on interaction data
        
        Args:
            train_features: np.array of shape [n_samples, input_dim]
            train_labels: np.array of shape [n_samples] (binary engagement labels)
            epochs: number of training epochs
            lr: learning rate
            batch_size: batch size for training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X = torch.FloatTensor(train_features)
        y = torch.FloatTensor(train_labels)

        # Handle empty training data gracefully
        if X.shape[0] == 0:
            print("[WARN] No engagement training data provided. Skipping training and using uninitialized model weights.")
            return
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                
                # Loss on engagement prediction
                loss = criterion(outputs['engagement'], batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, features):
        """
        Predict engagement scores
        
        Args:
            features: np.array of shape [n_samples, input_dim]
            
        Returns:
            dict with numpy arrays for each prediction head
        """
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features)
            outputs = self.forward(X)
            
            return {
                'engagement': outputs['engagement'].numpy(),
                'follow': outputs['follow'].numpy(),
                'embedding': outputs['embedding'].numpy(),
                'final_score': outputs['final_score'].numpy()
            }
