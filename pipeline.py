"""Product Mixer framework — simplified port of twitter/product-mixer.

Implements the core pipeline hierarchy from Twitter's Product Mixer:
  ProductPipeline -> MixerPipeline -> CandidatePipeline / ScoringPipeline

Source:
  product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/Pipeline.scala
  product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/product/ProductPipeline.scala
  product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/mixer/MixerPipeline.scala
  product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/candidate/CandidatePipeline.scala
  product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/scoring/ScoringPipeline.scala
"""


class Pipeline:
    """Base pipeline — all pipeline types extend this.

    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/Pipeline.scala
    """
    def __init__(self, identifier):
        self.identifier = identifier
    
    def process(self, query):
        raise NotImplementedError


class Query:
    """Encapsulates a timeline request.

    Maps to PipelineQuery in Product Mixer.
    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/PipelineQuery.scala
    """
    def __init__(self, user_id, max_results=10):
        self.user_id = user_id
        self.max_results = max_results
        self.features = {}


class CandidateWithDetails:
    """A scored candidate with source attribution and feature map.

    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/model/common/presentation/CandidateWithDetails.scala
    """
    def __init__(self, candidate_id, source, score=0.0):
        self.id = candidate_id
        self.source = source
        self.score = score
        self.features = {}
    
    def __repr__(self):
        return f"Candidate({self.id}, source={self.source}, score={self.score:.3f})"


class Gate:
    """Controls whether a pipeline executes for a given query.

    Production gates check feature flags, user eligibility, and experiments.
    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/functional_component/gate/Gate.scala
    """
    def __init__(self, name):
        self.name = name

    def should_continue(self, query):
        return True


class ProductPipeline(Pipeline):
    """Top-level entry point — selects which MixerPipeline to run.

    In production this resolves the product surface (ForYou, Following, etc.)
    and delegates to the appropriate mixer.
    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/product/ProductPipeline.scala
    """
    def __init__(self, identifier, mixer_pipeline, gates=None):
        super().__init__(identifier)
        self.mixer_pipeline = mixer_pipeline
        self.gates = gates or []

    def process(self, query):
        for gate in self.gates:
            if not gate.should_continue(query):
                return []
        return self.mixer_pipeline.process(query)


class MixerPipeline(Pipeline):
    """Orchestrates candidate retrieval, scoring, selection, and filtering.

    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/mixer/MixerPipeline.scala
    Home Mixer entry: home-mixer/server/src/main/scala/com/twitter/home_mixer/product/for_you/ForYouScoredTweetsMixerPipelineConfig.scala
    """
    def __init__(self, identifier, candidate_pipelines, scoring_pipeline, selectors, filters, gates=None):
        super().__init__(identifier)
        self.candidate_pipelines = candidate_pipelines
        self.scoring_pipeline = scoring_pipeline
        self.selectors = selectors
        self.filters = filters
        self.gates = gates or []
    
    def process(self, query):
        # Gate check
        for gate in self.gates:
            if not gate.should_continue(query):
                return []

        # Fetch candidates from all sources
        all_candidates = []
        for pipeline in self.candidate_pipelines:
            candidates = pipeline.process(query)
            all_candidates.extend(candidates)
        
        # Score candidates
        scored = self.scoring_pipeline.process(query, all_candidates)
        
        # Apply selectors
        for selector in self.selectors:
            scored = selector.apply(query, scored)
        
        # Apply filters
        for filter_fn in self.filters:
            scored = filter_fn.apply(query, scored)
        
        return scored[:query.max_results]


class CandidatePipeline(Pipeline):
    """Retrieves and pre-filters candidates from a single source.

    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/candidate/CandidatePipeline.scala
    """
    def __init__(self, identifier, source, filters=None, gates=None):
        super().__init__(identifier)
        self.source = source
        self.filters = filters or []
        self.gates = gates or []
    
    def process(self, query):
        for gate in self.gates:
            if not gate.should_continue(query):
                return []
        candidates = self.source.fetch(query)
        for filter_fn in self.filters:
            candidates = filter_fn.apply(query, candidates)
        return candidates


class ScoringPipeline(Pipeline):
    """Applies one or more scorers sequentially, then sorts by score.

    Source: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/scoring/ScoringPipeline.scala
    """
    def __init__(self, identifier, scorers):
        super().__init__(identifier)
        self.scorers = scorers
    
    def process(self, query, candidates):
        for scorer in self.scorers:
            candidates = scorer.score(query, candidates)
        return sorted(candidates, key=lambda c: c.score, reverse=True)
