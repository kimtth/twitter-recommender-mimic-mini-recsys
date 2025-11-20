# Twitter Recommendation Algorithm - System Flow Diagram

## High-Level Request Flow

**Source Files:**
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/for_you/ForYouProductPipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/following/FollowingProductPipelineConfig.scala`
- `pushservice/src/main/scala/com/twitter/frigate/pushservice/PushServiceMain.scala`

```mermaid
flowchart TB
    User[User Request] --> ProductPipeline[Product Pipeline]
    ProductPipeline --> |For You Timeline| ForYouFlow[For You Flow]
    ProductPipeline --> |Following Timeline| FollowingFlow[Following Flow]
    ProductPipeline --> |Lists| ListsFlow[Lists Flow]
    ProductPipeline --> |Notifications| NotifFlow[Notifications Flow]
    
    ForYouFlow --> MixerPipeline[Mixer Pipeline]
    FollowingFlow --> FollowingMixer[Following Mixer]
    ListsFlow --> ListsMixer[Lists Mixer]
    NotifFlow --> PushService[Push Service]
    
    MixerPipeline --> CandidateGen[Candidate Generation]
    MixerPipeline --> Ranking[Ranking & Scoring]
    MixerPipeline --> Filtering[Filtering & Heuristics]
    MixerPipeline --> Mixing[Content Mixing]
    
    Mixing --> Response[Response to User]
    
    style ProductPipeline fill:#e1f5ff
    style MixerPipeline fill:#ffe1f5
    style CandidateGen fill:#fff5e1
    style Ranking fill:#e1ffe1
```

## For You Timeline - Detailed Flow

**Source Files:**
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/for_you/ForYouProductPipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/for_you/ForYouScoredTweetsMixerPipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/for_you/ForYouScoredTweetsCandidatePipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/ScoredTweetsRecommendationPipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/scoring_pipeline/ScoredTweetsScoringPipelineConfig.scala`

```mermaid
flowchart TB
    subgraph Request["Request Entry"]
        UserReq[User Request] --> ForYouProduct[ForYouProductPipelineConfig]
    end
    
    subgraph Orchestration["Orchestration Layer"]
        ForYouProduct --> ForYouMixer[ForYouScoredTweetsMixerPipelineConfig]
        ForYouMixer --> ScoredTweetsCandidate[ForYouScoredTweetsCandidatePipelineConfig]
        ForYouMixer --> AdsCandidate[ForYouAdsCandidatePipelineConfig]
        ForYouMixer --> WTFCandidate[ForYouWhoToFollowCandidatePipelineConfig]
        ForYouMixer --> ConversationCandidate[ForYouConversationServiceCandidatePipelineConfig]
    end
    
    subgraph TweetRec["Tweet Recommendation Pipeline"]
        ScoredTweetsCandidate --> RecommendationPipeline[ScoredTweetsRecommendationPipelineConfig]
        
        RecommendationPipeline --> CandidateSources[Candidate Sources]
        
        CandidateSources --> InNetwork[ScoredTweetsInNetworkCandidatePipelineConfig<br/>Earlybird Search]
        CandidateSources --> TweetMixer[ScoredTweetsTweetMixerCandidatePipelineConfig<br/>CR-Mixer]
        CandidateSources --> UTEG[ScoredTweetsUtegCandidatePipelineConfig<br/>User Tweet Entity Graph]
        CandidateSources --> FRS[ScoredTweetsFrsCandidatePipelineConfig<br/>Follow Recommendations]
        
        InNetwork --> FeatureHydration[Feature Hydration]
        TweetMixer --> FeatureHydration
        UTEG --> FeatureHydration
        FRS --> FeatureHydration
        
        FeatureHydration --> Scoring[ScoredTweetsScoringPipelineConfig]
        Scoring --> MLScorer[Heavy Ranker<br/>Neural Network]
        MLScorer --> ScoredCandidates[Scored Candidates]
    end
    
    subgraph Mixing["Content Mixing & Assembly"]
        ScoredCandidates --> Selectors[Selectors & Heuristics]
        AdsCandidate --> Selectors
        WTFCandidate --> Selectors
        ConversationCandidate --> Selectors
        
        Selectors --> Diversity[Author Diversity]
        Selectors --> ContentBalance[Content Balance]
        Selectors --> Dedup[Deduplication]
        
        Diversity --> Filters[Filters]
        ContentBalance --> Filters
        Dedup --> Filters
        
        Filters --> VisibilityFilter[Visibility Filters]
        Filters --> FeedbackFilter[Feedback Fatigue]
        Filters --> HealthFilter[Health & Safety]
    end
    
    subgraph Marshalling["Response Marshalling"]
        VisibilityFilter --> Transform[Transformers]
        FeedbackFilter --> Transform
        HealthFilter --> Transform
        
        Transform --> SocialContext[Social Context]
        Transform --> ConversationModules[Conversation Modules]
        Transform --> ProductFeatures[Product Features]
        
        SocialContext --> DomainMarshaller[Domain Marshaller]
        ConversationModules --> DomainMarshaller
        ProductFeatures --> DomainMarshaller
        
        DomainMarshaller --> Timeline[Timeline Response]
    end
    
    style Request fill:#e1f5ff
    style Orchestration fill:#ffe1f5
    style TweetRec fill:#fff5e1
    style Mixing fill:#e1ffe1
    style Marshalling fill:#f5e1ff
```

## Candidate Generation Sources

**Source Files:**
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/ScoredTweetsInNetworkCandidatePipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/ScoredTweetsTweetMixerCandidatePipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/ScoredTweetsUtegCandidatePipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/candidate_pipeline/ScoredTweetsFrsCandidatePipelineConfig.scala`
- `cr-mixer/server/src/main/scala/com/twitter/cr_mixer/` (CR-Mixer service)
- `follow-recommendations-service/server/src/main/scala/com/twitter/follow_recommendations/` (FRS service)
- `src/scala/com/twitter/recos/user_tweet_entity_graph/` (UTEG service)

```mermaid
flowchart TB
    subgraph Sources["Candidate Sources"]
        Query[User Query] --> SearchIndex[Earlybird Search Index<br/>In-Network Tweets]
        Query --> CRMixer[CR-Mixer<br/>Out-of-Network Tweets]
        Query --> UTEG[User Tweet Entity Graph<br/>GraphJet-based]
        Query --> FRS[Follow Recommendation Service<br/>FutureGraph Tweets]
        Query --> Communities[Communities]
        Query --> Lists[Lists]
    end
    
    subgraph CRMixerFlow["CR-Mixer Pipeline"]
        CRMixer --> SignalExtract[Signal Extraction]
        SignalExtract --> UserProfile[User Profile Service]
        SignalExtract --> RealGraph[Real Graph]
        SignalExtract --> SimClusters[SimClusters]
        
        UserProfile --> CandGenServices[Candidate Generation Services]
        RealGraph --> CandGenServices
        SimClusters --> CandGenServices
        
        CandGenServices --> Deduping[Deduping & Filtering]
        Deduping --> LightRanking[Light Ranking]
    end
    
    subgraph FRSFlow["FRS Pipeline"]
        FRS --> FRSCandGen[Candidate Generation]
        FRSCandGen --> FRSSignals[User Signals & Algorithms]
        FRSSignals --> FRSFiltering[Filtering]
        FRSFiltering --> FRSRanking[ML Ranking]
        FRSRanking --> FRSTransform[Transform & Social Proof]
    end
    
    SearchIndex --> Candidates[Candidate Pool]
    LightRanking --> Candidates
    FRSTransform --> Candidates
    UTEG --> Candidates
    Communities --> Candidates
    Lists --> Candidates
    
    Candidates --> FeatureHydration[Feature Hydration<br/>~6000 Features]
    
    style Sources fill:#e1f5ff
    style CRMixerFlow fill:#ffe1f5
    style FRSFlow fill:#fff5e1
```

## Scoring and Ranking Pipeline

**Source Files:**
- `product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/scoring/ScoringPipeline.scala`
- `product-mixer/core/src/main/scala/com/twitter/product_mixer/core/pipeline/scoring/ScoringPipelineBuilder.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/scoring_pipeline/ScoredTweetsModelScoringPipelineConfig.scala`
- `home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/scorer/`
- `product-mixer/component-library/src/main/scala/com/twitter/product_mixer/component_library/module/HomeScorerClientModule.scala`

```mermaid
flowchart TB
    subgraph Input["Input"]
        Candidates[Candidate Tweets<br/>with Features] --> Gate[Gate Evaluation]
    end
    
    subgraph Scoring["Scoring Pipeline"]
        Gate --> |Pass| Selector[Selector]
        Gate --> |Fail| Skip[Skip Scoring]
        
        Selector --> SelectedCands[Selected Candidates]
        SelectedCands --> FeatureHydrator[Candidate Feature Hydrator]
        
        FeatureHydrator --> DataRecord[DataRecord Construction<br/>User-Candidate Pairs]
        
        DataRecord --> MLService[ML Prediction Service<br/>Home Scorer]
        MLService --> |TensorFlow Model| Predictions[Predictions]
        
        Predictions --> ProbFollow["P(follow|recommendation)"]
        Predictions --> ProbEngage["P(engagement|follow)"]
        
        ProbFollow --> WeightedScore[Weighted Score Calculation]
        ProbEngage --> WeightedScore
    end
    
    subgraph Reranking["Reranking"]
        WeightedScore --> ModelReranker[Model Reranking Scorer]
        ModelReranker --> PhoenixReranker[Phoenix Model Reranker]
        PhoenixReranker --> FinalScores[Final Ranked Scores]
    end
    
    subgraph Selection["Final Selection"]
        FinalScores --> PostSelector[Post-Scoring Selectors]
        Skip --> PostSelector
        
        PostSelector --> RankedResults[Ranked Results<br/>with Updated Features]
    end
    
    style Input fill:#e1f5ff
    style Scoring fill:#ffe1f5
    style Reranking fill:#fff5e1
    style Selection fill:#e1ffe1
```

## Data Flow Through Services

**Source Files:**
- `tweetypie/server/src/main/scala/com/twitter/tweetypie/` (Tweetypie service)
- `unified_user_actions/service/src/main/scala/com/twitter/unified_user_actions/service/` (UUA service)
- `user-signal-service/server/src/main/scala/com/twitter/usersignalservice/` (User Signal Service)
- `src/scala/com/twitter/simclusters_v2/` (SimClusters)
- `src/scala/com/twitter/interaction_graph/` (Real Graph)
- `graph-feature-service/src/main/scala/com/twitter/graph_feature_service/` (Graph Feature Service)
- `product-mixer/core/src/main/scala/com/twitter/product_mixer/core/` (Product Mixer framework)
- `navi/navi/src/` (Navi model serving)
- `representation-manager/server/src/main/scala/com/twitter/representation_manager/` (Representation Manager)

---

**Note for Mini-RecSys Implementation:**

Key differences from production Twitter architecture:
- **Data files**: Only 4 Parquet files (`users`, `tweets`, `interactions`, `follows`) - no pre-computed embeddings
- **Model serving**: PyTorch models loaded from `models/*.pt` files, predictions on-demand
- **Simplified pipeline**: Single-process execution vs distributed microservices
- **Feature computation**: 20 features vs 6000+ in production
- **Graceful fallback**: Falls back to feature-based heuristics if models not trained

```mermaid
flowchart LR
    subgraph DataSources["Data Sources"]
        Tweetypie[Tweetypie<br/>Tweet Data]
        UUA[Unified User Actions<br/>Real-time Stream]
        USS[User Signal Service<br/>Explicit & Implicit Signals]
    end
    
    subgraph Models["ML Models & Features"]
        SimClusters[SimClusters<br/>Community Detection]
        TwHIN[TwHIN<br/>Dense Embeddings]
        RealGraph[Real Graph<br/>User Interaction Prediction]
        TrustSafety[Trust & Safety Models<br/>NSFW/Abuse Detection]
        GraphFeatures[Graph Feature Service<br/>User-User Features]
    end
    
    subgraph Frameworks["Service Frameworks"]
        ProductMixer[Product Mixer<br/>Framework]
        Navi[Navi<br/>ML Model Serving]
        RepManager[Representation Manager<br/>Embedding Retrieval]
        TAF[Timelines Aggregation<br/>Framework]
    end
    
    subgraph Services["Core Services"]
        HomeMixer[Home Mixer]
        CRMixer[CR-Mixer]
        FRS[Follow Recommendations]
        TweetMixer[Tweet Mixer]
        PushService[Push Service]
    end
    
    DataSources --> Models
    Models --> Frameworks
    Frameworks --> Services
    
    Services --> Timeline[Timeline Response]
    Services --> Notifications[Notification Response]
    
    style DataSources fill:#e1f5ff
    style Models fill:#ffe1f5
    style Frameworks fill:#fff5e1
    style Services fill:#e1ffe1
```

## Push Notification Flow

**Source Files:**
- `pushservice/src/main/scala/com/twitter/frigate/pushservice/` (Push Service)
- `pushservice/src/main/scala/com/twitter/frigate/pushservice/model/candidate/`
- `pushservice/src/main/python/models/light_ranking/` (Light Ranker model)
- `pushservice/src/main/python/models/heavy_ranking/` (Heavy Ranker model)
- `pushservice/src/main/scala/com/twitter/frigate/pushservice/take/predicates/` (Filters)

```mermaid
flowchart TB
    subgraph Trigger["Notification Trigger"]
        Event[User Event/Action] --> PushService[Push Service]
    end
    
    subgraph CandGen["Candidate Generation"]
        PushService --> CandSources[Multiple Candidate Sources]
        CandSources --> TweetCands[Tweet Candidates]
        CandSources --> UserCands[User Candidates]
        CandSources --> TrendCands[Trend Candidates]
    end
    
    subgraph Ranking["Two-Stage Ranking"]
        TweetCands --> LightRanker[Light Ranker<br/>Pre-selection]
        UserCands --> LightRanker
        TrendCands --> LightRanker
        
        LightRanker --> Filtered[Filtered Candidate Pool<br/>~100-1000 candidates]
        
        Filtered --> HeavyRanker[Heavy Ranker<br/>Multi-task Learning]
        HeavyRanker --> ProbOpen["P(open notification)"]
        HeavyRanker --> ProbEngage["P(engagement)"]
        
        ProbOpen --> FinalScore[Final Relevance Score]
        ProbEngage --> FinalScore
    end
    
    subgraph Delivery["Notification Delivery"]
        FinalScore --> TopK[Select Top-K]
        TopK --> Personalization[Personalization<br/>Timing & Frequency]
        Personalization --> VisFilter[Visibility Filters]
        VisFilter --> Send[Send Notification]
    end
    
    style Trigger fill:#e1f5ff
    style CandGen fill:#ffe1f5
    style Ranking fill:#fff5e1
    style Delivery fill:#e1ffe1
```
