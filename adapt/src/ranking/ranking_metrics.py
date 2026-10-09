"""
Ranking and decision evaluation metrics for ADAPT.
Computes Top-1 Accuracy, Mean Reciprocal Rank (MRR), and NDCG@K.
"""

from typing import List, Sequence
import numpy as np


def compute_top1_accuracy(predicted_top: Sequence[str], ground_truth_top: Sequence[str]) -> float:
    """Compute Top-1 exact match accuracy."""
    assert len(predicted_top) == len(ground_truth_top)
    if len(predicted_top) == 0:
        return 0.0
    matches = sum(1 for p, g in zip(predicted_top, ground_truth_top) if p == g)
    return float(matches / len(predicted_top))


def compute_mrr(predicted_rankings: Sequence[List[str]], optimal_actions: Sequence[str]) -> float:
    """Compute Mean Reciprocal Rank of the optimal action within predicted rankings."""
    assert len(predicted_rankings) == len(optimal_actions)
    if len(predicted_rankings) == 0:
        return 0.0
    reciprocal_ranks = []
    for ranking, optimal in zip(predicted_rankings, optimal_actions):
        try:
            rank = ranking.index(optimal) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


def compute_ndcg_at_k(predicted_rankings: Sequence[List[str]], ideal_rankings: Sequence[List[str]], k: int = 4) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
    Relevance is assigned graded scores based on position in ideal ranking:
    rank 0 -> 4, rank 1 -> 3, rank 2 -> 2, rank 3 -> 1.
    """
    assert len(predicted_rankings) == len(ideal_rankings)
    if len(predicted_rankings) == 0:
        return 0.0

    ndcg_list = []
    for pred, ideal in zip(predicted_rankings, ideal_rankings):
        ideal_relevance = {act: float(len(ideal) - idx) for idx, act in enumerate(ideal)}

        dcg = 0.0
        for i, act in enumerate(pred[:k]):
            rel = ideal_relevance.get(act, 0.0)
            dcg += rel / np.log2(i + 2)

        idcg = 0.0
        for i, act in enumerate(ideal[:k]):
            rel = ideal_relevance.get(act, 0.0)
            idcg += rel / np.log2(i + 2)

        if idcg > 0:
            ndcg_list.append(dcg / idcg)
        else:
            ndcg_list.append(1.0)

    return float(np.mean(ndcg_list))
