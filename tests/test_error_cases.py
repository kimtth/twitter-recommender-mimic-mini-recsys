import numpy as np
try:
    from embeddings import SimClusters, TwHIN, RealGraph
except ModuleNotFoundError:
    SimClusters = None
    TwHIN = None
    RealGraph = None

INVALID_USER = 99999999
INVALID_TWEET = 99999999


def test_simclusters_missing_user_returns_default():
    if SimClusters is None:
        return
    sc = SimClusters(num_clusters=150)
    emb = sc.get_user_embedding(INVALID_USER)
    assert emb == {0: 1.0}


def test_twhin_missing_user_returns_zero_vector():
    if TwHIN is None:
        return
    twhin = TwHIN(dim=128, use_trained_model=False)
    emb = twhin.get_user_embedding(INVALID_USER)
    assert emb.shape == (128,)
    assert np.allclose(emb, 0.0)


def test_twhin_missing_tweet_returns_zero_vector():
    if TwHIN is None:
        return
    twhin = TwHIN(dim=128, use_trained_model=False)
    emb = twhin.get_tweet_embedding(INVALID_TWEET)
    assert emb.shape == (128,)
    assert np.allclose(emb, 0.0)


def test_realgraph_missing_users_interaction_zero():
    if RealGraph is None:
        return
    rg = RealGraph()
    score = rg.predict_interaction(INVALID_USER, INVALID_USER)
    assert score == 0.0
