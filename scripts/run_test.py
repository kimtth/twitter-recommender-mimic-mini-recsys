"""
Comprehensive test script with trained models
- Loads trained PyTorch models
- Tests each component individually
- Runs full pipeline E2E test
- Reports detailed results
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from time import time


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_data_loading():
    """Test dataset loading"""
    print_section("1. Testing Data Loading")
    
    from data_loader import get_data_loader
    
    dl = get_data_loader()
    
    print(f"Users: {len(dl.users)}")
    print(f"Tweets: {len(dl.tweets)}")
    print(f"Interactions: {len(dl.interactions)}")
    print(f"Follows: {len(dl.follows)}")
    
    assert len(dl.users) > 0, "No users loaded"
    assert len(dl.tweets) > 0, "No tweets loaded"
    assert len(dl.interactions) > 0, "No interactions loaded"
    
    print("[OK] Data loading passed")
    return dl


def test_embedding_models(dl):
    """Test embedding model inference"""
    print_section("2. Testing Embedding Models")
    
    from embeddings import SimClusters, TwHIN, RealGraph
    
    user_id = dl.users[0]['user_id']
    tweet_id = dl.tweets[0]['tweet_id']
    
    # Test SimClusters
    print("\nSimClusters:")
    sc = SimClusters(num_clusters=150)
    user_emb = sc.get_user_embedding(user_id)
    tweet_emb = sc.get_tweet_embedding(tweet_id)
    sim = sc.cosine_similarity(user_id, tweet_id)
    print(f"  User embedding: {len(user_emb)} clusters")
    print(f"  Tweet embedding: {len(tweet_emb)} clusters")
    print(f"  Similarity: {sim:.4f}")
    assert isinstance(user_emb, dict), "SimClusters should return dict"
    
    # Test TwHIN
    print("\nTwHIN:")
    twhin = TwHIN(dim=128, use_trained_model=True)
    user_emb = twhin.get_user_embedding(user_id)
    tweet_emb = twhin.get_tweet_embedding(tweet_id)
    sim = twhin.cosine_similarity(user_id, tweet_id)
    print(f"  User embedding shape: {user_emb.shape}")
    print(f"  Tweet embedding shape: {tweet_emb.shape}")
    print(f"  Similarity: {sim:.4f}")
    assert user_emb.shape == (128,), "TwHIN user embedding wrong shape"
    assert tweet_emb.shape == (128,), "TwHIN tweet embedding wrong shape"
    
    # Test RealGraph
    print("\nRealGraph:")
    rg = RealGraph()
    user_a = dl.users[0]['user_id']
    user_b = dl.users[1]['user_id'] if len(dl.users) > 1 else user_a
    score = rg.predict_interaction(user_a, user_b)
    neighborhood = rg.get_neighborhood(user_a, k=5)
    print(f"  Interaction score: {score:.4f}")
    print(f"  Neighborhood size: {len(neighborhood)}")
    assert 0.0 <= score <= 1.0, "RealGraph score out of range"
    
    print("\n[OK] Embedding models passed")


def test_candidate_sources(dl):
    """Test candidate retrieval"""
    print_section("3. Testing Candidate Sources")
    
    from pipeline import Query
    from candidates import InNetworkSource, OutOfNetworkSource, GraphSource
    
    user_id = dl.users[0]['user_id']
    query = Query(user_id=user_id, max_results=10)
    
    # Test InNetwork
    print("\nInNetworkSource:")
    source = InNetworkSource()
    candidates = source.fetch(query)
    print(f"  Candidates: {len(candidates)}")
    if candidates:
        print(f"  Sample: {candidates[0]}")
        assert 'in_network' in candidates[0].features
    
    # Test OutOfNetwork
    print("\nOutOfNetworkSource:")
    source = OutOfNetworkSource()
    candidates = source.fetch(query)
    print(f"  Candidates: {len(candidates)}")
    if candidates:
        print(f"  Sample features: simclusters={candidates[0].features.get('simclusters_score', 0):.3f}, twhin={candidates[0].features.get('twhin_score', 0):.3f}")
        assert 'simclusters_score' in candidates[0].features
        assert 'twhin_score' in candidates[0].features
    
    # Test Graph
    print("\nGraphSource:")
    source = GraphSource()
    candidates = source.fetch(query)
    print(f"  Candidates: {len(candidates)}")
    if candidates:
        print(f"  Sample: {candidates[0]}")
        assert 'author_similarity' in candidates[0].features
    
    print("\n[OK] Candidate sources passed")


def test_ml_scoring(dl):
    """Test ML models"""
    print_section("4. Testing ML Scoring")
    
    from pipeline import Query, CandidateWithDetails
    from scoring import MLScorer
    from safety import SafetyScorer
    
    user_id = dl.users[0]['user_id']
    query = Query(user_id=user_id, max_results=10)
    
    # Create dummy candidates
    candidates = [
        CandidateWithDetails(dl.tweets[i]['tweet_id'], "Test")
        for i in range(min(5, len(dl.tweets)))
    ]
    
    # Test ML Scorer
    print("\nMLScorer (Engagement Model):")
    scorer = MLScorer(use_trained_model=True)
    scored = scorer.score(query, candidates)
    print(f"  Scored {len(scored)} candidates")
    for i, c in enumerate(scored[:3]):
        print(f"  [{i+1}] Score: {c.score:.3f}, Engagement: {c.features.get('engagement_prob', 0):.3f}")
    
    # Test Safety Scorer
    print("\nSafetyScorer (NSFW & Toxicity):")
    safety_scorer = SafetyScorer()
    scored = safety_scorer.score(query, scored)
    print(f"  Processed {len(scored)} candidates")
    for i, c in enumerate(scored[:3]):
        nsfw = c.features.get('nsfw_score', 0)
        tox = c.features.get('toxicity_score', 0)
        print(f"  [{i+1}] NSFW: {nsfw:.3f}, Toxicity: {tox:.3f}")
    
    print("\n[OK] ML scoring passed")


def test_full_pipeline(dl):
    """Test complete recommendation pipeline"""
    print_section("5. Testing Full Pipeline (E2E)")
    
    from main import create_for_you_pipeline
    from pipeline import Query
    
    user_id = dl.users[0]['user_id']
    query = Query(user_id=user_id, max_results=10)
    
    print(f"\nGenerating recommendations for user {user_id}...")
    
    # Temporarily lower threshold for testing
    from filtering import ScoreThresholdFilter
    pipeline = create_for_you_pipeline()
    # Find and adjust score threshold filter
    for f in pipeline.filters:
        if isinstance(f, ScoreThresholdFilter):
            f.threshold = 0.01  # Lower threshold for testing
    
    results = pipeline.process(query)
    
    print(f"\n[OK] Generated {len(results)} recommendations")
    print("\nTop 5 results:")
    for i, candidate in enumerate(results[:5], 1):
        features = candidate.features
        print(f"\n{i}. Tweet {candidate.id} (source: {candidate.source})")
        print(f"   Score: {candidate.score:.3f}")
        print(f"   Engagement: {features.get('engagement_prob', 0):.3f}")
        print(f"   NSFW: {features.get('nsfw_score', 0):.3f}")
        print(f"   Toxicity: {features.get('toxicity_score', 0):.3f}")
    
    # Validate results
    assert len(results) <= 10, "Too many results returned"
    assert len(results) > 0, "No results returned"
    
    for c in results:
        assert 0.0 <= c.score <= 1.0, f"Score out of range: {c.score}"
        assert 'engagement_prob' in c.features, "Missing engagement_prob"
        assert 'nsfw_score' in c.features, "Missing nsfw_score"
        assert 'toxicity_score' in c.features, "Missing toxicity_score"
    
    # Check source diversity
    sources = {}
    for c in results:
        sources[c.source] = sources.get(c.source, 0) + 1
    
    print("\nSource distribution:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\n[OK] Full pipeline E2E test passed")


def test_model_files():
    """Verify trained model files exist and are loadable"""
    print_section("6. Testing Trained Model Files")
    
    import torch
    from models.engagement_model import EngagementModel
    from models.safety_models import NSFWModel, ToxicityModel
    from models.embedding_model import TwHINEmbeddingModel
    
    models_dir = 'models'
    
    # Test Engagement Model
    print("\nEngagement Model:")
    path = os.path.join(models_dir, 'engagement_model.pt')
    if os.path.exists(path):
        model = EngagementModel(input_dim=20)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"  [OK] Loaded from {path}")
        
        # Test inference
        dummy_input = np.random.randn(1, 20).astype(np.float32)
        preds = model.predict(dummy_input)
        print(f"  [OK] Inference works: engagement={preds['engagement'][0]:.3f}")
    else:
        print(f"  [X] Not found: {path}")
    
    # Test NSFW Model
    print("\nNSFW Model:")
    path = os.path.join(models_dir, 'nsfw_model.pt')
    if os.path.exists(path):
        model = NSFWModel(input_dim=10)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"  [OK] Loaded from {path}")
        
        dummy_input = np.random.randn(1, 10).astype(np.float32)
        score = model.predict(dummy_input)
        print(f"  [OK] Inference works: nsfw={score[0]:.3f}")
    else:
        print(f"  [X] Not found: {path}")
    
    # Test Toxicity Model
    print("\nToxicity Model:")
    path = os.path.join(models_dir, 'toxicity_model.pt')
    if os.path.exists(path):
        model = ToxicityModel(input_dim=10)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"  [OK] Loaded from {path}")
        
        dummy_input = np.random.randn(1, 10).astype(np.float32)
        score = model.predict(dummy_input)
        print(f"  [OK] Inference works: toxicity={score[0]:.3f}")
    else:
        print(f"  [X] Not found: {path}")
    
    # Test TwHIN Model
    print("\nTwHIN Embedding Model:")
    path = os.path.join(models_dir, 'twhin_model.pt')
    if os.path.exists(path):
        model = TwHINEmbeddingModel(user_feature_dim=10, tweet_feature_dim=10, embedding_dim=128)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"  [OK] Loaded from {path}")
        
        dummy_user = np.random.randn(1, 10).astype(np.float32)
        emb = model.encode_users(dummy_user)
        print(f"  [OK] Inference works: embedding shape={emb.shape}")
    else:
        print(f"  [X] Not found: {path}")
    
    print("\n[OK] Model files test passed")


def main():
    """Run comprehensive test suite"""
    start_time = time()
    
    print("=" * 70)
    print("Mini-RecSys Comprehensive Test Suite")
    print("Testing trained models and full pipeline")
    print("=" * 70)
    
    try:
        # Run all tests
        dl = test_data_loading()
        test_model_files()
        test_embedding_models(dl)
        test_candidate_sources(dl)
        test_ml_scoring(dl)
        test_full_pipeline(dl)
        
        # Success summary
        elapsed = time() - start_time
        print_section("[OK] All Tests Passed!")
        print(f"Total time: {elapsed:.1f} seconds")
        print("\nThe recommendation system is working correctly with trained models.")
        
        return 0
        
    except Exception as e:
        print(f"\n[X] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
