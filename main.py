"""Example usage - mimics For You Timeline"""
from pipeline import Query, MixerPipeline, CandidatePipeline, ScoringPipeline
from candidates import InNetworkSource, OutOfNetworkSource, GraphSource
from scoring import MLScorer, DiversityScorer, RecencyScorer
from safety import SafetyScorer
from filtering import (
    DeduplicationFilter, ScoreThresholdFilter, ContentBalanceFilter,
    TopKSelector, SourceDiversitySelector
)

def create_for_you_pipeline():
    """Creates a mini For You Timeline pipeline"""
    
    # Candidate Pipelines (parallel fetch)
    in_network_pipeline = CandidatePipeline(
        identifier="InNetwork",
        source=InNetworkSource(),
        filters=[DeduplicationFilter()]
    )
    
    oon_pipeline = CandidatePipeline(
        identifier="OutOfNetwork",
        source=OutOfNetworkSource(),
        filters=[DeduplicationFilter()]
    )
    
    graph_pipeline = CandidatePipeline(
        identifier="Graph",
        source=GraphSource()
    )
    
    # Scoring Pipeline (multiple models)
    scoring = ScoringPipeline(
        identifier="Scoring",
        scorers=[
            MLScorer(),           # Heavy Ranker (multi-head: engagement, follow, embeddings)
            SafetyScorer(),       # Trust & Safety (NSFW, toxicity)
            RecencyScorer(),      # Time decay
            DiversityScorer(),    # Source diversity (InNetwork/OutOfNetwork/Graph)
        ]
    )
    
    # Mixer Pipeline (orchestrates everything)
    mixer = MixerPipeline(
        identifier="ForYouMixer",
        candidate_pipelines=[
            in_network_pipeline,
            oon_pipeline,
            graph_pipeline,
        ],
        scoring_pipeline=scoring,
        selectors=[
            SourceDiversitySelector(min_per_source=2),
            TopKSelector(k=30),
        ],
        filters=[
            ScoreThresholdFilter(threshold=0.2),
            ContentBalanceFilter(in_network_ratio=0.5),
        ]
    )
    
    return mixer

def main():
    # Create pipeline
    pipeline = create_for_you_pipeline()
    
    # Simulate user request
    query = Query(user_id=1, max_results=10)
    
    print(f"Processing For You Timeline for user {query.user_id}")
    print("=" * 60)
    
    # Process
    results = pipeline.process(query)
    
    # Display results
    print(f"\nTop {len(results)} recommendations:")
    print("-" * 60)
    for i, candidate in enumerate(results, 1):
        features = candidate.features
        engagement = features.get('engagement_prob', 0)
        follow = features.get('follow_prob', 0)
        simclusters = features.get('simclusters_score', 0)
        twhin = features.get('twhin_score', 0)
        nsfw = features.get('nsfw_score', 0)
        toxicity = features.get('toxicity_score', 0)
        
        print(f"{i:2d}. {candidate}")
        print(f"    ML: engagement={engagement:.3f}, follow={follow:.3f}")
        print(f"    Embeddings: simclusters={simclusters:.3f}, twhin={twhin:.3f}")
        print(f"    Safety: nsfw={nsfw:.3f}, toxicity={toxicity:.3f}")
    
    # Show source distribution
    print("\n" + "=" * 60)
    print("Source Distribution:")
    print("-" * 60)
    source_counts = {}
    for candidate in results:
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
