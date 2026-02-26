"""Filters and selectors — mimics VisibilityLibrary and Product Mixer selectors.

Visibility filters: visibilitylib/
Selectors: product-mixer/core/src/main/scala/com/twitter/product_mixer/core/functional_component/selector/
"""

class Filter:
    def __init__(self, name):
        self.name = name
    
    def apply(self, query, candidates):
        raise NotImplementedError

class ScoreThresholdFilter(Filter):
    """Remove low-quality candidates"""
    def __init__(self, threshold=0.1):
        super().__init__("ScoreThreshold")
        self.threshold = threshold
    
    def apply(self, query, candidates):
        return [c for c in candidates if c.score >= self.threshold]

class DeduplicationFilter(Filter):
    """Remove duplicates"""
    def __init__(self):
        super().__init__("Deduplication")
    
    def apply(self, query, candidates):
        seen = set()
        unique = []
        for candidate in candidates:
            if candidate.id not in seen:
                seen.add(candidate.id)
                unique.append(candidate)
        return unique

class ContentBalanceFilter(Filter):
    """Ensure balance between in-network and out-of-network"""
    def __init__(self, in_network_ratio=0.5):
        super().__init__("ContentBalance")
        self.ratio = in_network_ratio
    
    def apply(self, query, candidates):
        in_network = [c for c in candidates if c.source == "InNetwork"]
        out_of_network = [c for c in candidates if c.source != "InNetwork"]
        
        # Calculate target counts
        total = min(len(candidates), query.max_results)
        in_target = int(total * self.ratio)
        oon_target = total - in_target
        
        # Balance the mix
        balanced = in_network[:in_target] + out_of_network[:oon_target]
        return sorted(balanced, key=lambda c: c.score, reverse=True)

class Selector:
    """Selects subset of candidates to process"""
    def __init__(self, name):
        self.name = name
    
    def apply(self, query, candidates):
        return candidates

class TopKSelector(Selector):
    """Select top K by score"""
    def __init__(self, k=50):
        super().__init__("TopK")
        self.k = k
    
    def apply(self, query, candidates):
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        return sorted_candidates[:self.k]

class SourceDiversitySelector(Selector):
    """Ensure representation from all sources"""
    def __init__(self, min_per_source=2):
        super().__init__("SourceDiversity")
        self.min_per_source = min_per_source
    
    def apply(self, query, candidates):
        by_source = {}
        for candidate in candidates:
            if candidate.source not in by_source:
                by_source[candidate.source] = []
            by_source[candidate.source].append(candidate)
        
        # Take minimum from each source first
        selected = []
        for source_candidates in by_source.values():
            sorted_source = sorted(source_candidates, key=lambda c: c.score, reverse=True)
            selected.extend(sorted_source[:self.min_per_source])
        
        # Fill remaining with top scores
        remaining = [c for c in candidates if c not in selected]
        remaining_sorted = sorted(remaining, key=lambda c: c.score, reverse=True)
        selected.extend(remaining_sorted[:query.max_results - len(selected)])
        
        return sorted(selected, key=lambda c: c.score, reverse=True)
