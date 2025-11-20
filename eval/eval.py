"""Run evaluation on mini-recsys"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import create_for_you_pipeline
from eval_metric import (
    OfflineEvaluator,
    split_interactions,
    print_evaluation_results
)
from data_loader import get_data_loader
import numpy as np


def main():
    print("=" * 70)
    print("Mini-RecSys Evaluation".center(70))
    print("Mimics Twitter's Evaluation Pipeline".center(70))
    print("=" * 70)
    
    # Load data
    data_loader = get_data_loader()
    
    # Use pre-defined train/test split from dataset
    print("\n[INFO] Using temporal train/test split from dataset...")
    train_interactions = [i for i in data_loader.interactions if i.get('split') == 'train']
    test_interactions = [i for i in data_loader.interactions if i.get('split') == 'test']
    
    if not train_interactions or not test_interactions:
        # Fallback to random split if dataset doesn't have split field
        print("[WARN] No split field found, falling back to random split...")
        train_interactions, test_interactions = split_interactions(
            data_loader.interactions,
            test_ratio=0.2,
            seed=42
        )
    
    print(f"  Train interactions: {len(train_interactions)}")
    print(f"  Test interactions:  {len(test_interactions)}")
    
    # Get test users (users with test interactions)
    test_user_ids = list(set(inter['user_id'] for inter in test_interactions))
    print(f"  Test users: {len(test_user_ids)}")
    
    # Use ALL test users (or fixed sample) for deterministic evaluation
    user_test_counts = {}
    for inter in test_interactions:
        user_id = inter['user_id']
        user_test_counts[user_id] = user_test_counts.get(user_id, 0) + 1
    
    # Filter to users with at least 2 test interactions for meaningful metrics
    active_test_users = [uid for uid, count in user_test_counts.items() if count >= 2]
    
    # Use fixed subset for speed (deterministic with seed)
    np.random.seed(42)
    eval_users = sorted(active_test_users)[:100]  # First 100 users (deterministic)
    print(f"  Evaluating on: {len(eval_users)} test users (with >=2 test interactions)")
    
    # Create pipeline
    print("\n[INFO] Building recommendation pipeline...")
    pipeline = create_for_you_pipeline()
    
    # Initialize evaluator
    evaluator = OfflineEvaluator(test_interactions)
    
    # 1. Ranking Metrics
    print("\n" + "=" * 70)
    print("1. RANKING QUALITY EVALUATION")
    print("=" * 70)
    ranking_results = evaluator.evaluate_pipeline(
        pipeline,
        eval_users,
        k_values=[5, 10, 20]
    )
    print_evaluation_results(ranking_results, "Ranking Metrics")
    
    # 2. Engagement Prediction Metrics
    print("\n" + "=" * 70)
    print("2. ENGAGEMENT PREDICTION EVALUATION")
    print("=" * 70)
    engagement_results = evaluator.evaluate_engagement_prediction(
        pipeline,
        eval_users
    )
    print_evaluation_results(engagement_results, "Engagement Metrics")
    
    # 3. Source-level Breakdown
    print("\n" + "=" * 70)
    print("3. METRICS BY CANDIDATE SOURCE")
    print("=" * 70)
    source_results = evaluator.evaluate_by_source(
        pipeline,
        eval_users
    )
    
    print("\n[INFO] Candidate Source Performance:")
    print("-" * 70)
    for source, metrics in sorted(source_results.items()):
        print(f"\n  {source}:")
        print(f"    Candidates:        {metrics['count']}")
        print(f"    CTR:               {metrics['ctr']:.4f} ({metrics['ctr']*100:.2f}%)")
        print(f"    Predicted CTR:     {metrics['predicted_ctr']:.4f}")
        print(f"    AUC-ROC:           {metrics['auc_roc']:.4f}")
        print(f"    PR-AUC:            {metrics['pr_auc']:.4f}")
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print("\n[INFO] Key Takeaways:")
    
    ndcg_10 = ranking_results.get('ndcg@10', 0)
    precision_10 = ranking_results.get('precision@10', 0)
    rce = engagement_results.get('rce', 0)
    auc_roc = engagement_results.get('auc_roc', 0)
    
    print("\n  Ranking Quality:")
    print(f"    • NDCG@10 = {ndcg_10:.4f}  {'GOOD' if ndcg_10 > 0.3 else 'NEEDS IMPROVEMENT'}")
    print(f"    • Precision@10 = {precision_10:.4f}  {'GOOD' if precision_10 > 0.1 else 'NEEDS IMPROVEMENT'}")
    
    print("\n  Engagement Prediction:")
    print(f"    • RCE = {rce:.2f}%  {'GOOD' if rce > 5 else 'NEEDS IMPROVEMENT'}")
    print(f"    • AUC-ROC = {auc_roc:.4f}  {'GOOD' if auc_roc > 0.6 else 'NEEDS IMPROVEMENT'}")
    
    print("\n  Interpretation:")
    print("    • NDCG@10 > 0.3:     Good ranking quality")
    print("    • RCE > 5%:          Better than baseline model")
    print("    • AUC-ROC > 0.6:     Decent discrimination ability")
    
    print("\n" + "=" * 70)
    print("[OK] Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
