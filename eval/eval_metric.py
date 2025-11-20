"""Evaluation metrics for recommendation system - mimics Twitter's evaluation"""
import numpy as np
from collections import defaultdict
from data_loader import get_data_loader


class RecommendationMetrics:
    """Compute recommendation quality metrics"""
    
    @staticmethod
    def ndcg_at_k(ranked_items, relevant_items, k=10):
        """
        Normalized Discounted Cumulative Gain at K
        Used by Twitter for ranking quality evaluation
        """
        if not relevant_items:
            return 0.0
        
        # DCG@K
        dcg = 0.0
        for i, item_id in enumerate(ranked_items[:k]):
            if item_id in relevant_items:
                # Relevance score (1 if relevant, 0 otherwise)
                rel = 1.0
                # Discount by log2(position + 2)
                dcg += rel / np.log2(i + 2)
        
        # IDCG@K (ideal DCG with perfect ranking)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(ranked_items, relevant_items, k=10):
        """
        Precision@K - fraction of top-K that are relevant
        """
        if k == 0:
            return 0.0
        
        top_k = ranked_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / k
    
    @staticmethod
    def recall_at_k(ranked_items, relevant_items, k=10):
        """
        Recall@K - fraction of relevant items in top-K
        """
        if not relevant_items:
            return 0.0
        
        top_k = ranked_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / len(relevant_items)
    
    @staticmethod
    def mean_reciprocal_rank(ranked_items, relevant_items):
        """
        MRR - reciprocal rank of first relevant item
        """
        for i, item_id in enumerate(ranked_items):
            if item_id in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def hit_rate_at_k(ranked_items, relevant_items, k=10):
        """
        Hit Rate@K - whether any relevant item appears in top-K
        """
        top_k = set(ranked_items[:k])
        return 1.0 if top_k & set(relevant_items) else 0.0


class EngagementMetrics:
    """Compute engagement prediction metrics - mimics Twitter's ML metrics"""
    
    @staticmethod
    def rce(predictions, labels, weights=None):
        """
        Relative Cross Entropy (RCE)
        Twitter's primary metric for binary prediction models
        Measures improvement over baseline (predicting average CTR)
        """
        if weights is None:
            weights = np.ones(len(labels))
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Model cross entropy
        ce_model = -np.sum(
            weights * (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        ) / np.sum(weights)
        
        # Baseline cross entropy (always predict average)
        avg_label = np.sum(weights * labels) / np.sum(weights)
        avg_label = np.clip(avg_label, 1e-7, 1 - 1e-7)
        ce_baseline = -(avg_label * np.log(avg_label) + (1 - avg_label) * np.log(1 - avg_label))
        
        return 100.0 * (1 - ce_model / ce_baseline)
    
    @staticmethod
    def auc_roc(predictions, labels):
        """
        Area Under ROC Curve
        Standard metric for binary classification
        """
        # Sort by predictions
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_labels = labels[sorted_indices]
        
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Count inversions
        auc = 0.0
        pos_seen = 0
        for label in sorted_labels:
            if label == 0:
                auc += pos_seen
            else:
                pos_seen += 1
        
        return auc / (n_pos * n_neg)
    
    @staticmethod
    def precision_recall_auc(predictions, labels):
        """
        Area Under Precision-Recall Curve
        Used by Twitter for imbalanced datasets
        """
        # Sort by predictions descending
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Compute precision and recall at each threshold
        precisions = []
        recalls = []
        
        tp = 0
        fp = 0
        total_pos = np.sum(labels)
        
        if total_pos == 0:
            return 0.0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / total_pos
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(recalls)):
            auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
        
        return auc
    
    @staticmethod
    def precision_at_recall(predictions, labels, target_recall=0.9):
        """
        Precision at fixed recall threshold
        Twitter uses this for safety models (e.g., precision @ 0.9 recall)
        """
        # Sort by predictions descending
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_labels = labels[sorted_indices]
        
        total_pos = np.sum(labels)
        if total_pos == 0:
            return 0.0
        
        tp = 0
        fp = 0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            recall = tp / total_pos
            if recall >= target_recall:
                precision = tp / (tp + fp)
                return precision
        
        return 0.0


class OfflineEvaluator:
    """
    Offline evaluation framework
    Simulates Twitter's offline evaluation pipeline
    """
    
    def __init__(self, test_interactions):
        self.data_loader = get_data_loader()
        self.test_interactions = test_interactions
        
        # Build ground truth: user -> set of engaged tweets
        self.ground_truth = defaultdict(set)
        for inter in test_interactions:
            self.ground_truth[inter['user_id']].add(inter['tweet_id'])
    
    def evaluate_pipeline(self, pipeline, test_users, k_values=[5, 10, 20]):
        """
        Evaluate recommendation pipeline on test users
        
        Args:
            pipeline: The recommendation pipeline to evaluate
            test_users: List of user IDs to test
            k_values: List of K values for @K metrics
        
        Returns:
            Dictionary of metric results
        """
        from pipeline import Query
        
        results = {
            'ndcg': {k: [] for k in k_values},
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'hit_rate': {k: [] for k in k_values},
            'mrr': [],
        }
        
        print(f"\nEvaluating on {len(test_users)} test users...")
        
        for user_id in test_users:
            # Get ground truth
            relevant_items = self.ground_truth.get(user_id, set())
            if not relevant_items:
                continue
            
            # Get recommendations
            query = Query(user_id=user_id, max_results=max(k_values))
            try:
                recommendations = pipeline.process(query)
                ranked_items = [c.id for c in recommendations]
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
            
            # Compute metrics at different K
            for k in k_values:
                results['ndcg'][k].append(
                    RecommendationMetrics.ndcg_at_k(ranked_items, relevant_items, k)
                )
                results['precision'][k].append(
                    RecommendationMetrics.precision_at_k(ranked_items, relevant_items, k)
                )
                results['recall'][k].append(
                    RecommendationMetrics.recall_at_k(ranked_items, relevant_items, k)
                )
                results['hit_rate'][k].append(
                    RecommendationMetrics.hit_rate_at_k(ranked_items, relevant_items, k)
                )
            
            # MRR (single value, not @K)
            results['mrr'].append(
                RecommendationMetrics.mean_reciprocal_rank(ranked_items, relevant_items)
            )
        
        # Aggregate results
        aggregated = {}
        for metric_name, k_dict in results.items():
            if metric_name == 'mrr':
                aggregated['mrr'] = np.mean(results['mrr']) if results['mrr'] else 0.0
            else:
                for k, values in k_dict.items():
                    key = f'{metric_name}@{k}'
                    aggregated[key] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def evaluate_engagement_prediction(self, pipeline, test_users):
        """
        Evaluate engagement prediction quality
        Tests the ML scoring accuracy
        """
        from pipeline import Query
        
        all_predictions = []
        all_labels = []
        
        print(f"\nEvaluating engagement prediction on {len(test_users)} users...")
        
        for user_id in test_users:
            relevant_items = self.ground_truth.get(user_id, set())
            
            # Get recommendations with scores
            query = Query(user_id=user_id, max_results=100)
            try:
                recommendations = pipeline.process(query)
            except:
                continue
            
            for candidate in recommendations:
                # Prediction is the engagement probability from features
                pred = candidate.features.get('engagement_prob', candidate.score)
                # Label is 1 if user interacted, 0 otherwise
                label = 1.0 if candidate.id in relevant_items else 0.0
                
                all_predictions.append(pred)
                all_labels.append(label)
        
        if not all_predictions:
            return {}
        
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Compute engagement metrics
        metrics = {
            'rce': EngagementMetrics.rce(predictions, labels),
            'auc_roc': EngagementMetrics.auc_roc(predictions, labels),
            'pr_auc': EngagementMetrics.precision_recall_auc(predictions, labels),
            'precision@recall_0.9': EngagementMetrics.precision_at_recall(
                predictions, labels, target_recall=0.9
            ),
        }
        
        # Compute CTR
        metrics['ctr'] = np.mean(labels)
        metrics['predicted_ctr'] = np.mean(predictions)
        
        return metrics
    
    def evaluate_by_source(self, pipeline, test_users):
        """
        Break down metrics by candidate source
        (InNetwork vs OutOfNetwork vs Graph)
        """
        from pipeline import Query
        
        source_metrics = defaultdict(lambda: {
            'count': 0,
            'predictions': [],
            'labels': []
        })
        
        print(f"\nEvaluating by source on {len(test_users)} users...")
        
        for user_id in test_users:
            relevant_items = self.ground_truth.get(user_id, set())
            
            query = Query(user_id=user_id, max_results=100)
            try:
                recommendations = pipeline.process(query)
            except:
                continue
            
            for candidate in recommendations:
                source = candidate.source
                pred = candidate.features.get('engagement_prob', candidate.score)
                label = 1.0 if candidate.id in relevant_items else 0.0
                
                source_metrics[source]['count'] += 1
                source_metrics[source]['predictions'].append(pred)
                source_metrics[source]['labels'].append(label)
        
        # Compute metrics per source
        results = {}
        for source, data in source_metrics.items():
            if data['count'] == 0:
                continue
            
            preds = np.array(data['predictions'])
            labels = np.array(data['labels'])
            
            results[source] = {
                'count': data['count'],
                'ctr': np.mean(labels),
                'predicted_ctr': np.mean(preds),
                'auc_roc': EngagementMetrics.auc_roc(preds, labels),
                'pr_auc': EngagementMetrics.precision_recall_auc(preds, labels),
            }
        
        return results


def split_interactions(interactions, test_ratio=0.2, seed=42):
    """
    Split interactions into train/test for evaluation
    Test set simulates held-out future interactions
    """
    np.random.seed(seed)
    
    # Group by user
    user_interactions = defaultdict(list)
    for inter in interactions:
        user_interactions[inter['user_id']].append(inter)
    
    train_interactions = []
    test_interactions = []
    
    # For each user, split their interactions
    for user_id, inters in user_interactions.items():
        n_test = max(1, int(len(inters) * test_ratio))
        
        # Shuffle user's interactions
        shuffled = inters.copy()
        np.random.shuffle(shuffled)
        
        # Split
        test_interactions.extend(shuffled[:n_test])
        train_interactions.extend(shuffled[n_test:])
    
    return train_interactions, test_interactions


def print_evaluation_results(results, title="Evaluation Results"):
    """Pretty print evaluation metrics"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    if isinstance(results, dict):
        # Group by metric type
        ranking_metrics = {}
        engagement_metrics = {}
        
        for key, value in results.items():
            if '@' in key or key == 'mrr':
                ranking_metrics[key] = value
            else:
                engagement_metrics[key] = value
        
        if ranking_metrics:
            print("\n📊 Ranking Metrics:")
            print("-" * 60)
            for metric, value in sorted(ranking_metrics.items()):
                print(f"  {metric:30s}: {value:7.4f}")
        
        if engagement_metrics:
            print("\n💬 Engagement Metrics:")
            print("-" * 60)
            for metric, value in sorted(engagement_metrics.items()):
                if 'ctr' in metric.lower():
                    print(f"  {metric:30s}: {value:7.4f} ({value*100:.2f}%)")
                else:
                    print(f"  {metric:30s}: {value:7.4f}")
    
    print("=" * 60)
