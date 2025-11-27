# Twitter Recommender Mimic: Codebase Guide for Agents

This document provides a structured overview of the `twitter-recommender-mimic-mini-recsys` codebase, designed to help AI agents understand the system's architecture, component mapping, and implementation details relative to the original Twitter algorithm.

## 1. Project Overview
- **Goal**: A high-fidelity, simplified implementation of Twitter's "For You" timeline architecture.
- **Reference**: [twitter/the-algorithm](https://github.com/twitter/the-algorithm)
- **Key Characteristic**: Runs on a single machine using synthetic data while preserving the core "Funnel" architecture.

## 2. Component Mapping
This table maps original Twitter components to their implementation in this codebase.

| Concept | Original Component | Mimic Implementation | File Path |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | Home Mixer | `MixerPipeline` | `pipeline.py` |
| **Retrieval (In-Network)** | Earlybird Search Index | `InNetworkSource` | `candidates.py` |
| **Retrieval (Embedding)** | CR-Mixer (SimClusters + TwHIN) | `OutOfNetworkSource` | `candidates.py` |
| **Retrieval (Graph)** | UTEG (User Tweet Entity Graph) | `GraphSource` | `candidates.py` |
| **Scoring (Ranker)** | Heavy Ranker | `MLScorer` | `scoring.py` |
| **Scoring (Safety)** | Trust & Safety Models | `SafetyScorer` | `safety.py` |
| **Filtering** | Visibility Filters | `Filter` classes | `filtering.py` |
| **Data** | Tweetypie / Manhattan | `DataLoader` | `data_loader.py` |

## 3. Architectural Flow
The system follows a strict multi-stage funnel architecture.

### Stage 1: Candidate Retrieval
**Goal**: Fetch ~500 relevant tweets from millions.
- **In-Network**: Fetches tweets from followed users (`InNetworkSource`).
- **Out-Of-Network**: Uses embedding similarity (`SimClusters`, `TwHIN`) to find relevant content from non-followed users (`OutOfNetworkSource`).
- **Graph Traversal**: Finds tweets from users similar to those followed (`GraphSource`).

### Stage 2: Scoring (Heavy Ranker)
**Goal**: Score each candidate to predict engagement.
- **Feature Extraction**: `MLScorer._extract_model_features` extracts ~20 features (user, tweet, interaction context).
- **Inference**: Uses `EngagementModel` (PyTorch) to predict `p(Engagement)`, `p(Follow)`, and `p(Embedding similarity)`.
- **Fallback**: If no model is trained, uses heuristic scoring based on dataset statistics.

### Stage 3: Filtering & Selection
**Goal**: Apply business rules and safety constraints.
- **Safety**: `SafetyScorer` runs `NSFWModel` and `ToxicityModel`. High scores result in heavy penalties or removal.
- **Diversity**: `DiversityScorer` penalizes consecutive tweets from the same author.
- **Balance**: `ContentBalanceFilter` ensures a 50/50 mix of In-Network and Out-Of-Network content.

## 4. Machine Learning Models
The system uses PyTorch to implement simplified versions of production models.

### Engagement Model (Heavy Ranker)
- **File**: `models/engagement_model.py`
- **Type**: Multi-Task Learning (MTL) with Shared Bottom.
- **Heads**:
    1.  `engagement`: Predicts probability of interaction (Like/Retweet).
    2.  `follow`: Predicts probability of a follow.
    3.  `embedding`: Predicts embedding similarity.
- **Rationale**: Optimizes for multiple conflicting objectives simultaneously, preventing clickbait.

### Embedding Model (TwHIN)
- **File**: `models/embedding_model.py`
- **Type**: Two-Tower (Dual Encoder).
- **Architecture**: User Tower and Tweet Tower map features to a shared latent space ($\mathbb{R}^{128}$).
- **Rationale**: Allows efficient retrieval via Approximate Nearest Neighbor (ANN) search (simulated here).

### Safety Models
- **File**: `models/safety_models.py`
- **Type**: Independent Binary Classifiers.
- **Models**: `NSFWModel`, `ToxicityModel`.
- **Rationale**: Safety is a hard constraint, not just a ranking feature. These run independently to flag harmful content.

## 5. Data & Simulation
- **Generation**: `prep/generate_dataset.py` creates synthetic data with power-law distributions for followers and interactions to mimic real social network topology.
- **Loading**: `data_loader.py` uses a Singleton pattern to load Parquet files into memory, simulating a distributed data store.

## 6. Embedding Consolidation Flow

The three embeddings work together in a staged pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Parallel Candidate Retrieval (~500 candidates)                 │
├─────────────────┬─────────────────────┬─────────────────────────────────┤
│ InNetwork (200) │ OutOfNetwork (200)  │ Graph (100)                     │
│ No embedding    │ SimClusters + TwHIN │ RealGraph                       │
└────────┬────────┴──────────┬──────────┴────────────────┬────────────────┘
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Feature Consolidation (20 features per candidate)             │
│ • simclusters_score (0.0 if N/A)                                        │
│ • twhin_score (0.0 if N/A)                                              │
│ • author_similarity (0.0 if N/A)                                        │
│ • engagement_prob, follow_prob, + 15 more                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Multi-Head Scoring (Weighted Fusion)                           │
│ ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────────┐   │
│ │ Engagement 50%│ │ Follow 20%    │ │ Embedding 30%                 │   │
│ │               │ │               │ │ SimClusters×0.25+TwHIN×0.20   │   │
│ │               │ │               │ │ +RealGraph×0.15               │   │
│ └───────┬───────┘ └───────┬───────┘ └───────────────┬───────────────┘   │
│         └─────────────────┼─────────────────────────┘                   │
│                           ▼                                             │
│                   Final Score (0.0-1.0)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Not all candidates use all embeddings. Missing scores default to 0.0, enabling graceful cross-source comparison.

## 7. Discrepancy Analysis: Mimic vs Original Twitter Algorithm

This section identifies key differences between this mimic and the original Twitter codebase, noting where simplifications are acceptable ("OK") and where rationales may be misleading ("Fix Needed").

### 7.1 SimClusters Implementation

| Aspect | Original Twitter | Mimic | Status |
|--------|-----------------|-------|--------|
| **Algorithm** | Community detection via Metropolis-Hastings sampling on Producer-Producer similarity graph | Hash-based deterministic mapping: `hash(interest) % 150` | ✅ OK (documented proxy) |
| **Input** | Follow graph (bipartite user-producer graph) | User `interests` field (predefined categories) | ✅ OK (simplified) |
| **Clusters** | ~145,000 communities from 20M producers | 150 fixed buckets | ✅ OK (scale) |
| **Embedding Type** | Sparse "InterestedIn" vectors computed from KnownFor matrix | Sparse dict from hashed interests | ✅ Structurally similar |

**Implementation Note**: The mimic uses a hash-based proxy (`hash(interest) % 150`) instead of actual community detection. This is a valid simplification that preserves the sparse embedding structure while being computationally tractable for educational purposes. 

### 7.2 TwHIN Implementation

| Aspect | Original Twitter | Mimic | Status |
|--------|-----------------|-------|--------|
| **Architecture** | Knowledge graph embeddings (TransE-style) on heterogeneous graph | Two-tower neural network (user/tweet encoders) | ✅ OK (alternative approach) |
| **Training Data** | Follow, Favorite, Reply, Retweet edges | Synthetic interaction pairs | ✅ OK (simplified) |
| **Embedding Dim** | 200-dim (per TwHIN paper) | 128-dim | ✅ OK (scale) |

**Implementation Note**: The mimic uses a two-tower neural network architecture instead of knowledge graph embeddings (TransE). Both approaches achieve the same goal: learning dense semantic embeddings for user-tweet similarity. The two-tower approach is widely used in production recommendation systems (YouTube, Pinterest) and is more intuitive for educational purposes. The README correctly identifies this as a "two-tower neural network" rather than claiming to implement TransE.

### 7.3 Safety Models

| Aspect | Original Twitter | Mimic | Status |
|--------|-----------------|-------|--------|
| **NSFW Model** | BERT-based text encoder (Twitter-BERT) + image models | Simple MLP on 10 numeric features | ✅ OK (metadata-based proxy) |
| **Toxicity Model** | Twitter-BERT / BERTweet with fine-tuning | Simple MLP on 10 numeric features | ✅ OK (metadata-based proxy) |
| **Input** | Raw text + images | Numeric features (text_length, has_media, etc.) | ✅ OK (no text in synthetic data) |

**Implementation Note**: The mimic uses metadata-based safety scoring (MLP on numeric features) instead of NLP-based content analysis. This is necessary because the synthetic dataset contains no actual text or images to analyze. The approach demonstrates the safety filtering architecture and thresholds while being honest about limitations. The README correctly notes these are "metadata-based" models rather than claiming to perform actual content analysis.

### 7.4 Heavy Ranker

| Aspect | Original Twitter | Mimic | Status |
|--------|-----------------|-------|--------|
| **Features** | ~6,000 features | 20 features | ✅ OK (scale) |
| **Architecture** | MaskNet (TensorFlow) | Simple MLP (PyTorch) | ✅ OK (simplified) |
| **Multi-task** | Predicts multiple engagement types | 3 heads (engagement, follow, embedding) | ✅ Accurate |
| **Serving** | Navi (Rust-based model server) | Direct PyTorch inference | ✅ OK (single-machine) |

**Status**: ✅ This is a **good mimic**. The multi-head architecture correctly captures the production pattern.

### 7.5 Pipeline Architecture

| Aspect | Original Twitter | Mimic | Status |
|--------|-----------------|-------|--------|
| **Framework** | Product Mixer (Scala) | Python classes | ✅ Structurally accurate |
| **Candidate Sources** | Earlybird, CR-Mixer, UTEG, FRS | InNetwork, OutOfNetwork, Graph | ✅ Good coverage |
| **Mixing** | Tweets + Ads + WTF + Conversations | Tweets only | ✅ OK (simplified) |

**Status**: ✅ The pipeline structure is a **faithful mimic** of the Product Mixer pattern.

## 8. Summary: What's Accurate vs Simplified

| Component | Accuracy | Notes |
|-----------|----------|-------|
| Pipeline Architecture | ✅ High | Faithful Product Mixer mimic |
| Candidate Sources | ✅ High | Correct 3-source pattern |
| Heavy Ranker | ✅ High | Multi-head MTL is accurate |
| SimClusters | ✅ Medium | Hash-based proxy (documented as such) |
| TwHIN | ✅ Medium | Two-tower alternative (documented as such) |
| Safety Models | ✅ Medium | Metadata-based proxy (documented as such) |
| Feature Engineering | ✅ High | Correct patterns, reduced scale |
| Evaluation Metrics | ✅ High | NDCG, RCE, AUC match production |

**Overall**: This is a **well-documented educational mimic** that captures the architectural essence of Twitter's recommendation system. The simplifications are reasonable for a single-machine implementation, and the documentation clearly identifies where alternative approaches are used (hash-based SimClusters proxy, two-tower TwHIN, metadata-based safety models) rather than claiming full algorithmic fidelity.
