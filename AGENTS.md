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
| **Retrieval (In-Network)** | Earlybird | `InNetworkSource` | `candidates.py` |
| **Retrieval (Embedding)** | SimClusters / TwHIN | `OutOfNetworkSource` | `candidates.py` |
| **Retrieval (Graph)** | RealGraph / GraphJet | `GraphSource` | `candidates.py` |
| **Scoring (Ranker)** | Heavy Ranker (MaskNet) | `MLScorer` | `scoring.py` |
| **Scoring (Safety)** | Trust & Safety Models | `SafetyScorer` | `safety.py` |
| **Filtering** | Visibility Filters | `Filter` classes | `filtering.py` |
| **Data** | Tweet/User Store | `DataLoader` | `data_loader.py` |

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
- **Inference**: Uses `EngagementModel` (PyTorch) to predict `p(Like)`, `p(Reply)`, and `p(Follow)`.
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
