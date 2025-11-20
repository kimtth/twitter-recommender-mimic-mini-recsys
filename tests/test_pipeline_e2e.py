try:
    from pipeline import Query
    from main import create_for_you_pipeline
except ModuleNotFoundError:
    Query = None
    create_for_you_pipeline = None

def test_pipeline_end_to_end():
    if create_for_you_pipeline is None:
        return
    pipeline = create_for_you_pipeline()
    query = Query(user_id=1, max_results=10)
    results = pipeline.process(query)
    assert len(results) <= 10
    # Each candidate should have a calibrated score between 0 and 1
    for c in results:
        assert 0.0 <= c.score <= 1.0
        # Core features expected after scoring
        assert 'engagement_prob' in c.features
        assert 'nsfw_score' in c.features
        assert 'toxicity_score' in c.features
