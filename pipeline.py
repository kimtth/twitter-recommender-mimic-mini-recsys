"""Core pipeline framework - mimics Product Mixer"""

class Pipeline:
    def __init__(self, identifier):
        self.identifier = identifier
    
    def process(self, query):
        raise NotImplementedError

class Query:
    def __init__(self, user_id, max_results=10):
        self.user_id = user_id
        self.max_results = max_results
        self.features = {}

class CandidateWithDetails:
    def __init__(self, candidate_id, source, score=0.0):
        self.id = candidate_id
        self.source = source
        self.score = score
        self.features = {}
    
    def __repr__(self):
        return f"Candidate({self.id}, source={self.source}, score={self.score:.3f})"

class MixerPipeline(Pipeline):
    def __init__(self, identifier, candidate_pipelines, scoring_pipeline, selectors, filters):
        super().__init__(identifier)
        self.candidate_pipelines = candidate_pipelines
        self.scoring_pipeline = scoring_pipeline
        self.selectors = selectors
        self.filters = filters
    
    def process(self, query):
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
    def __init__(self, identifier, source, filters=None):
        super().__init__(identifier)
        self.source = source
        self.filters = filters or []
    
    def process(self, query):
        candidates = self.source.fetch(query)
        for filter_fn in self.filters:
            candidates = filter_fn.apply(query, candidates)
        return candidates

class ScoringPipeline(Pipeline):
    def __init__(self, identifier, scorers):
        super().__init__(identifier)
        self.scorers = scorers
    
    def process(self, query, candidates):
        for scorer in self.scorers:
            candidates = scorer.score(query, candidates)
        return sorted(candidates, key=lambda c: c.score, reverse=True)
