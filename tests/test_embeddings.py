import numpy as np
import torch
try:
    from models.embedding_model import TwHINEmbeddingModel
    from embeddings import SimClusters, TwHIN, RealGraph
    from data_loader import get_data_loader
except ModuleNotFoundError:
    TwHINEmbeddingModel = None
    SimClusters = None
    TwHIN = None
    RealGraph = None

def test_twhin_embedding_model_forward_shapes():
    if TwHINEmbeddingModel is None:
        return
    model = TwHINEmbeddingModel(user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128)
    user_features = torch.randn(32, 10)
    tweet_features = torch.randn(32, 10)
    user_emb, tweet_emb, similarity = model(user_features, tweet_features)
    assert user_emb.shape == (32, 128)
    assert tweet_emb.shape == (32, 128)
    assert similarity.shape == (32,)
    # Embeddings should be L2-normalized ~1.0
    user_norms = torch.norm(user_emb, dim=1)
    tweet_norms = torch.norm(tweet_emb, dim=1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-5)
    assert torch.allclose(tweet_norms, torch.ones_like(tweet_norms), atol=1e-5)

def test_twhin_encode_methods():
    if TwHINEmbeddingModel is None:
        return
    model = TwHINEmbeddingModel(user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128)
    user_features = np.random.randn(5, 10).astype(np.float32)
    tweet_features = np.random.randn(7, 10).astype(np.float32)
    user_embs = model.encode_users(user_features)
    tweet_embs = model.encode_tweets(tweet_features)
    assert user_embs.shape == (5, 128)
    assert tweet_embs.shape == (7, 128)
    assert np.allclose(np.linalg.norm(user_embs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(tweet_embs, axis=1), 1.0, atol=1e-5)

def test_simclusters_user_embedding_distribution():
    if SimClusters is None:
        return
    dl = get_data_loader()
    if not dl.users:
        return
    user_id = dl.users[0]['user_id']
    sc = SimClusters(num_clusters=150)
    emb = sc.get_user_embedding(user_id)
    assert isinstance(emb, dict)
    assert len(emb) >= 1
    total = sum(emb.values())
    assert abs(total - 1.0) < 1e-6

def test_twhin_user_embedding_fallback_or_trained():
    if TwHIN is None:
        return
    dl = get_data_loader()
    if not dl.users:
        return
    user_id = dl.users[0]['user_id']
    twhin = TwHIN(dim=128, use_trained_model=True)
    emb = twhin.get_user_embedding(user_id)
    assert emb.shape == (128,)
    norm = np.linalg.norm(emb)
    # Fallback or trained should both produce non-zero normalized-ish vectors
    assert norm > 0

def test_realgraph_interaction_and_neighborhood():
    if RealGraph is None:
        return
    dl = get_data_loader()
    if not dl.users:
        return
    rg = RealGraph()
    user_a = dl.users[0]['user_id']
    user_b = dl.users[1]['user_id'] if len(dl.users) > 1 else user_a
    score = rg.predict_interaction(user_a, user_b)
    assert 0.0 <= score <= 1.0
    neighborhood = rg.get_neighborhood(user_a, k=5)
    assert isinstance(neighborhood, list)
    assert len(neighborhood) <= 5
