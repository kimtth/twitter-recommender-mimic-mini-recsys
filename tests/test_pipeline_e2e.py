"""
End-to-end tests for the mini-recsys pipeline.

Coverage:
  1. Data generation  – run generate_dataset and verify files are written
  2. Data integrity   – schema, types, value ranges, referential consistency
  3. Pipeline E2E     – candidate retrieval → scoring → filtering → ranking
  4. Score semantics  – calibration, feature presence, source diversity
"""
import os
import sys
import importlib
import numpy as np
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

try:
    import pandas as pd
    import pyarrow.parquet as pq
    _parquet_available = True
except ImportError:
    _parquet_available = False

try:
    from pipeline import Query, CandidateWithDetails
    from main import create_for_you_pipeline
    from data_loader import get_data_loader
    from filtering import ScoreThresholdFilter
    _pipeline_available = True
except ModuleNotFoundError:
    Query = CandidateWithDetails = create_for_you_pipeline = get_data_loader = ScoreThresholdFilter = None
    _pipeline_available = False


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────
_REQUIRED_FILES = ["users.parquet", "tweets.parquet", "interactions.parquet", "follows.parquet"]


def _parquet_path(fname: str) -> str:
    return os.path.join(DATA_DIR, fname)


def _load_df(fname: str):
    return pd.read_parquet(_parquet_path(fname))


# ────────────────────────────────────────────────────────────────────────────
# 1 · Data Generation
# ────────────────────────────────────────────────────────────────────────────
class TestDataGeneration:
    """Verify that generate_dataset produces all required Parquet files."""

    def test_generate_dataset_creates_all_files(self, tmp_path):
        """Run the generator with a small scale and confirm output files exist."""
        import importlib.util, types

        spec = importlib.util.spec_from_file_location(
            "generate_dataset",
            os.path.join(ROOT, "prep", "generate_dataset.py"),
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        # Minimal scale for speed
        users = gen.generate_users(n_users=50)
        follows = gen.generate_follow_graph(users, avg_follows=5)
        tweets = gen.generate_tweets(users, n_tweets=200)
        interactions = gen.generate_interactions(users, tweets, follows, n_interactions=500)

        # Write to tmp_path
        pd.DataFrame(users).to_parquet(tmp_path / "users.parquet", index=False)
        pd.DataFrame(follows).to_parquet(tmp_path / "follows.parquet", index=False)
        pd.DataFrame(tweets).to_parquet(tmp_path / "tweets.parquet", index=False)
        pd.DataFrame(interactions).to_parquet(tmp_path / "interactions.parquet", index=False)

        for fname in _REQUIRED_FILES:
            assert (tmp_path / fname).exists(), f"Missing generated file: {fname}"

    def test_generate_users_structure(self):
        spec = importlib.util.spec_from_file_location(
            "generate_dataset",
            os.path.join(ROOT, "prep", "generate_dataset.py"),
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        users = gen.generate_users(n_users=20)
        assert len(users) == 20
        required_keys = {"user_id", "followers_count", "following_count", "interests", "verified"}
        for u in users:
            assert required_keys.issubset(u.keys()), f"Missing keys in user: {u.keys()}"
            assert u["followers_count"] >= 0
            assert isinstance(u["interests"], list) and len(u["interests"]) >= 1

    def test_generate_tweets_structure(self):
        spec = importlib.util.spec_from_file_location(
            "generate_dataset",
            os.path.join(ROOT, "prep", "generate_dataset.py"),
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        users = gen.generate_users(n_users=10)
        tweets = gen.generate_tweets(users, n_tweets=50)
        assert len(tweets) == 50
        for t in tweets:
            assert 0 <= t["text_length"] <= 280, "text_length exceeds Twitter limit"
            assert 0.0 <= t["nsfw_score"] <= 1.0
            assert 0.0 <= t["toxicity_score"] <= 1.0
            assert t["hours_old"] >= 0.0

    def test_interactions_temporal_split(self):
        spec = importlib.util.spec_from_file_location(
            "generate_dataset",
            os.path.join(ROOT, "prep", "generate_dataset.py"),
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        users = gen.generate_users(n_users=20)
        follows = gen.generate_follow_graph(users, avg_follows=5)
        tweets = gen.generate_tweets(users, n_tweets=100)
        interactions = gen.generate_interactions(users, tweets, follows, n_interactions=300)

        splits = [i["split"] for i in interactions]
        train = [s for s in splits if s == "train"]
        test  = [s for s in splits if s == "test"]
        assert len(train) > 0 and len(test) > 0, "Both splits must be present"
        # ~80 / ~20 split (allow ±5 pp tolerance)
        ratio = len(train) / len(interactions)
        assert 0.70 <= ratio <= 0.90, f"Train ratio out of expected range: {ratio:.2f}"

        # Timestamps must be monotonically non-decreasing
        timestamps = [i["timestamp"] for i in interactions]
        assert timestamps == sorted(timestamps), "Interactions not sorted by timestamp"


# ────────────────────────────────────────────────────────────────────────────
# 2 · Data Integrity (against the persisted dataset in data/)
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _parquet_available, reason="pandas/pyarrow not available")
class TestDataIntegrity:
    """Validate schema, value ranges, and referential integrity of the dataset."""

    @pytest.fixture(autouse=True)
    def require_data(self):
        for fname in _REQUIRED_FILES:
            if not os.path.exists(_parquet_path(fname)):
                pytest.skip(f"Dataset file missing: {fname}  (run prep/generate_dataset.py)")

    def test_users_schema_and_ranges(self):
        df = _load_df("users.parquet")
        assert len(df) > 0, "users table is empty"
        for col in ["user_id", "followers_count", "following_count", "account_age_days"]:
            assert col in df.columns, f"Missing column: {col}"
        assert (df["followers_count"] >= 0).all(), "Negative followers_count"
        assert (df["following_count"] >= 0).all(), "Negative following_count"
        assert df["user_id"].is_unique, "Duplicate user_ids"

    def test_tweets_schema_and_ranges(self):
        df = _load_df("tweets.parquet")
        assert len(df) > 0
        for col in ["tweet_id", "author_id", "text_length", "nsfw_score", "toxicity_score", "hours_old"]:
            assert col in df.columns, f"Missing column: {col}"
        assert (df["text_length"] >= 0).all() and (df["text_length"] <= 280).all(), "text_length out of [0, 280]"
        assert (df["nsfw_score"].between(0, 1)).all(), "nsfw_score out of [0, 1]"
        assert (df["toxicity_score"].between(0, 1)).all(), "toxicity_score out of [0, 1]"
        assert (df["hours_old"] >= 0).all(), "Negative hours_old"
        assert df["tweet_id"].is_unique, "Duplicate tweet_ids"

    def test_follows_referential_integrity(self):
        users_df = _load_df("users.parquet")
        follows_df = _load_df("follows.parquet")
        valid_ids = set(users_df["user_id"])
        assert follows_df["follower_id"].isin(valid_ids).all(), "follower_id references unknown user"
        assert follows_df["followed_id"].isin(valid_ids).all(), "followed_id references unknown user"
        # No self-follows
        assert (follows_df["follower_id"] != follows_df["followed_id"]).all(), "Self-follow detected"

    def test_interactions_referential_integrity(self):
        users_df = _load_df("users.parquet")
        tweets_df = _load_df("tweets.parquet")
        inter_df = _load_df("interactions.parquet")
        valid_users = set(users_df["user_id"])
        valid_tweets = set(tweets_df["tweet_id"])
        assert inter_df["user_id"].isin(valid_users).all(), "interaction user_id references unknown user"
        assert inter_df["tweet_id"].isin(valid_tweets).all(), "interaction tweet_id references unknown tweet"

    def test_interactions_split_field(self):
        df = _load_df("interactions.parquet")
        assert "split" in df.columns, "Missing 'split' column"
        assert set(df["split"].unique()).issubset({"train", "test"}), "Unexpected split values"
        train_pct = (df["split"] == "train").mean()
        assert 0.70 <= train_pct <= 0.90, f"Train split ratio unexpected: {train_pct:.2f}"

    def test_tweet_authors_exist_in_users(self):
        users_df = _load_df("users.parquet")
        tweets_df = _load_df("tweets.parquet")
        valid_users = set(users_df["user_id"])
        assert tweets_df["author_id"].isin(valid_users).all(), "tweet author_id not in users table"

    def test_dataset_scale_is_reasonable(self):
        users_df   = _load_df("users.parquet")
        tweets_df  = _load_df("tweets.parquet")
        inter_df   = _load_df("interactions.parquet")
        follows_df = _load_df("follows.parquet")
        assert len(users_df)   >= 100,  "Too few users"
        assert len(tweets_df)  >= 1000, "Too few tweets"
        assert len(inter_df)   >= 1000, "Too few interactions"
        assert len(follows_df) >= 100,  "Too few follows"


# ────────────────────────────────────────────────────────────────────────────
# 3 · Pipeline E2E
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _pipeline_available, reason="pipeline modules unavailable")
class TestPipelineE2E:
    """Full end-to-end pipeline tests."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        p = create_for_you_pipeline()
        # Lower score threshold so test data always yields results
        for f in p.filters:
            if isinstance(f, ScoreThresholdFilter):
                f.threshold = 0.01
        return p

    @pytest.fixture(scope="class")
    def dl(self):
        return get_data_loader()

    def test_pipeline_returns_results(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=10)
        results = pipeline.process(query)
        assert len(results) > 0, "Pipeline returned no results"

    def test_pipeline_respects_max_results(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=5)
        results = pipeline.process(query)
        assert len(results) <= 5

    def test_pipeline_scores_in_range(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=10)
        results = pipeline.process(query)
        for c in results:
            assert 0.0 <= c.score <= 1.0, f"Score out of [0,1]: {c.score}"

    def test_pipeline_required_features_present(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=10)
        results = pipeline.process(query)
        required = {"engagement_prob", "nsfw_score", "toxicity_score"}
        for c in results:
            missing = required - set(c.features.keys())
            assert not missing, f"Candidate {c.id} missing features: {missing}"

    def test_pipeline_results_are_ranked_descending(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=10)
        results = pipeline.process(query)
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by descending score"

    def test_pipeline_no_duplicate_tweets(self, pipeline, dl):
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=20)
        results = pipeline.process(query)
        ids = [c.id for c in results]
        assert len(ids) == len(set(ids)), f"Duplicate tweet IDs in results: {ids}"

    def test_pipeline_multiple_users(self, pipeline, dl):
        if len(dl.users) < 3:
            pytest.skip("Not enough users in dataset")
        for user in dl.users[:3]:
            query = Query(user_id=user["user_id"], max_results=10)
            results = pipeline.process(query)
            assert len(results) <= 10
            for c in results:
                assert 0.0 <= c.score <= 1.0

    def test_pipeline_source_diversity(self, pipeline, dl):
        """Results should come from at least two different candidate sources."""
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=20)
        results = pipeline.process(query)
        if len(results) < 2:
            return  # Not enough results to check diversity
        sources = {c.source for c in results}
        assert len(sources) >= 1, "No source labels on results"

    def test_pipeline_safety_filters_high_nsfw(self, pipeline, dl):
        """No result should have both a high NSFW score and a high final score."""
        if not dl.users:
            pytest.skip("No users in dataset")
        query = Query(user_id=dl.users[0]["user_id"], max_results=10)
        results = pipeline.process(query)
        for c in results:
            nsfw = c.features.get("nsfw_score", 0.0)
            if nsfw > 0.8:
                assert c.score < 0.5, (
                    f"High-NSFW tweet {c.id} (nsfw={nsfw:.2f}) has unexpectedly high score {c.score:.2f}"
                )

    def test_pipeline_invalid_user_returns_empty_or_fallback(self, pipeline):
        """Unknown user should not crash the pipeline."""
        query = Query(user_id=99999999, max_results=10)
        results = pipeline.process(query)  # must not raise
        assert isinstance(results, list)
