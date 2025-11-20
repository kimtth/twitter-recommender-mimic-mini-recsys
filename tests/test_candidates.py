from pipeline import Query
from candidates import InNetworkSource, OutOfNetworkSource, GraphSource
from data_loader import get_data_loader


def test_in_network_source_candidates():
    dl = get_data_loader()
    if not dl.users:
        return  # Skip if dataset unavailable
    user_id = dl.users[0]['user_id']
    src = InNetworkSource()
    query = Query(user_id=user_id, max_results=10)
    candidates = src.fetch(query)
    assert len(candidates) <= 200
    if candidates:  # Only assert features if we actually have follows
        c = candidates[0]
        assert 'in_network' in c.features


def test_out_of_network_source_similarity_features():
    dl = get_data_loader()
    if not dl.users:
        return
    user_id = dl.users[0]['user_id']
    src = OutOfNetworkSource()
    query = Query(user_id=user_id, max_results=10)
    candidates = src.fetch(query)
    # Should return up to 200
    assert len(candidates) <= 200
    if candidates:
        c = candidates[0]
        assert 'simclusters_score' in c.features
        assert 'twhin_score' in c.features


def test_graph_source_author_similarity():
    dl = get_data_loader()
    if not dl.users:
        return
    user_id = dl.users[0]['user_id']
    src = GraphSource()
    query = Query(user_id=user_id, max_results=10)
    candidates = src.fetch(query)
    assert len(candidates) <= 100
    if candidates:
        c = candidates[0]
        assert 'author_similarity' in c.features
