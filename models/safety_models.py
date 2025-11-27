"""
Safety models for content filtering

Simplified placeholders for Twitter's Trust & Safety models.

Production models use:
- Twitter-BERT / BERTweet text encoders for NLP-based content analysis
- Vision models (CNN/ViT) for image/video NSFW detection
- Multi-task learning across abuse types

This mimic uses:
- Simple MLP on numeric metadata features (text_length, has_media, etc.)
- Cannot detect actual NSFW/toxic content from text
- Predicts based on statistical correlates in synthetic data

See: trust_and_safety_models/ in twitter/the-algorithm
"""
import torch
import torch.nn as nn


class NSFWModel(nn.Module):
    """
    Metadata-based NSFW scoring (placeholder for production BERT + vision model).
    
    Production uses Twitter-BERT text encoder + image classifiers trained on
    labeled NSFW content. This mimic uses numeric features only, which cannot
    detect actual NSFW content—it predicts based on correlates like media presence.
    
    Input: 10 numeric features (text_length, has_media, has_link, etc.)
    Output: Probability score [0, 1]
    """
    
    def __init__(self, input_dim=10):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] tensor
            
        Returns:
            NSFW probability scores [batch_size]
        """
        return self.model(features).squeeze(-1)
    
    def train_model(self, train_features, train_labels, epochs=10, lr=0.001, batch_size=64):
        """Train NSFW detection model"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X = torch.FloatTensor(train_features)
        y = torch.FloatTensor(train_labels)

        if X.shape[0] == 0:
            print("[WARN] No NSFW training data provided. Skipping training and using uninitialized model weights.")
            return
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 2 == 0:
                print(f"  NSFW Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def predict(self, features):
        """Predict NSFW scores"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features)
            return self.forward(X).numpy()


class ToxicityModel(nn.Module):
    """
    Metadata-based toxicity scoring (placeholder for production BERT model).
    
    Production uses Twitter-BERT / BERTweet with fine-tuning on labeled
    toxic content (hate speech, harassment, abuse). This mimic uses numeric
    features only, which cannot detect actual toxicity from text.
    
    Input: 10 numeric features (text_length, has_media, has_link, etc.)
    Output: Probability score [0, 1]
    """
    
    def __init__(self, input_dim=10):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] tensor
            
        Returns:
            Toxicity probability scores [batch_size]
        """
        return self.model(features).squeeze(-1)
    
    def train_model(self, train_features, train_labels, epochs=10, lr=0.001, batch_size=64):
        """Train toxicity detection model"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X = torch.FloatTensor(train_features)
        y = torch.FloatTensor(train_labels)

        if X.shape[0] == 0:
            print("[WARN] No toxicity training data provided. Skipping training and using uninitialized model weights.")
            return
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 2 == 0:
                print(f"  Toxicity Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def predict(self, features):
        """Predict toxicity scores"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features)
            return self.forward(X).numpy()
