"""
Evaluation script for the scratch DecisionTransformer on the held-out test split.
Reports Top-1 Accuracy, MRR, NDCG@4, and per-class decision metrics.
"""

import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adapt.src.tokenizer.bpe_tokenizer import AdaptTokenizer
from adapt.src.transformer.decision_transformer import DecisionTransformer, TransformerConfig
from adapt.src.ranking.dataset import ScenarioRankingDataset, dynamic_collate_fn, IDX_TO_ACTION
from adapt.src.ranking.ranking_metrics import compute_top1_accuracy, compute_mrr, compute_ndcg_at_k


def evaluate_test_set(
    test_path: str,
    tokenizer_path: str,
    checkpoint_path: str,
    output_report_path: str = None,
):
    torch.set_num_threads(4)
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TransformerConfig.from_dict(checkpoint["config"])
    model = DecisionTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AdaptTokenizer(tokenizer_path)
    test_dataset = ScenarioRankingDataset(test_path, tokenizer, max_seq_len=160)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    all_pred_top = []
    all_true_top = []
    all_pred_rankings = []
    all_ideal_rankings = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            action_logits = model.score_candidate_actions(input_ids, attention_mask=attention_mask)
            sorted_indices = torch.argsort(action_logits, dim=-1, descending=True).cpu().numpy()

            for i in range(len(batch["target_idx"])):
                pred_actions = [IDX_TO_ACTION[idx] for idx in sorted_indices[i]]
                pred_top = pred_actions[0]
                true_top = batch["optimal_action"][i]

                all_pred_rankings.append(pred_actions)
                all_pred_top.append(pred_top)
                all_true_top.append(true_top)
                all_ideal_rankings.append(batch["ground_truth_ranking"][i])

                per_class_total[true_top] += 1
                if pred_top == true_top:
                    per_class_correct[true_top] += 1

    top1_acc = compute_top1_accuracy(all_pred_top, all_true_top)
    mrr = compute_mrr(all_pred_rankings, all_true_top)
    ndcg4 = compute_ndcg_at_k(all_pred_rankings, all_ideal_rankings, k=4)

    class_accuracies = {}
    for act, total in per_class_total.items():
        corr = per_class_correct[act]
        class_accuracies[act] = {
            "correct": corr,
            "total": total,
            "accuracy": round(corr / total, 4) if total > 0 else 0.0,
        }

    report = {
        "dataset": "held_out_test_set",
        "total_test_examples": len(all_pred_top),
        "top1_accuracy": round(top1_acc, 4),
        "mrr": round(mrr, 4),
        "ndcg4": round(ndcg4, 4),
        "per_class_metrics": class_accuracies,
    }

    print("\n--- Held-Out Test Evaluation Results ---")
    print(f"Total Test Examples: {len(all_pred_top)}")
    print(f"Top-1 Accuracy:     {top1_acc * 100:.2f}%")
    print(f"MRR:                {mrr:.4f}")
    print(f"NDCG@4:             {ndcg4:.4f}")
    print("Per-Class Accuracy:")
    for act, metrics in class_accuracies.items():
        print(f"  {act:16s}: {metrics['accuracy'] * 100:5.2f}% ({metrics['correct']}/{metrics['total']})")

    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report to {output_report_path}")

    return report


if __name__ == "__main__":
    test_f = os.path.join("adapt", "data", "processed", "test.jsonl")
    tok_f = os.path.join("adapt", "artifacts", "tokenizer", "adapt_tokenizer.json")
    ckpt_f = os.path.join("adapt", "artifacts", "checkpoints", "transformer_best.pt")
    out_rep = os.path.join("adapt", "experiments", "transformer_test_results.json")

    evaluate_test_set(test_f, tok_f, ckpt_f, out_rep)
