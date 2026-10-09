"""
Offline Retraining & Regression Evaluation Pipeline.
Combines base training dataset with approved, verified feedback.
Trains candidate checkpoint offline and runs regression checks against frozen test suite.
Prevents regression and guarantees explicit approval gate before deployment.
"""

import os
import sys
import json
import shutil
import time
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adapt.src.feedback.feedback_store import FeedbackStore
from adapt.src.training.train_transformer import train_transformer
from adapt.src.evaluation.eval_transformer import evaluate_test_set


def run_offline_retraining(
    base_train_path: str,
    val_path: str,
    test_path: str,
    feedback_store_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_allowed_accuracy_drop: float = 0.02,
) -> Dict[str, Any]:
    print("======================================================================")
    print("          ADAPT GOVERNED OFFLINE RETRAINING & REGRESSION SUITE        ")
    print("======================================================================")

    store = FeedbackStore(feedback_store_path)
    approved_export_path = os.path.join("adapt", "data", "processed", "approved_feedback_export.jsonl")
    num_approved = store.export_approved_for_retraining(approved_export_path)
    print(f"Ingested {num_approved} reviewed & approved feedback records.")

    # 1. Baseline Performance on Test Set
    baseline_ckpt = os.path.join(output_dir, "transformer_best.pt")
    assert os.path.exists(baseline_ckpt), f"Baseline checkpoint not found at {baseline_ckpt}"
    print("\n--- Evaluating Production Baseline Model on Golden Test Set ---")
    base_eval = evaluate_test_set(
        test_path=test_path,
        tokenizer_path=tokenizer_path,
        checkpoint_path=baseline_ckpt,
    )
    base_acc = base_eval["top1_accuracy"]
    base_mrr = base_eval["mrr"]
    print(f"Production Baseline: Accuracy = {base_acc*100:.2f}%, MRR = {base_mrr:.4f}")

    # 2. Prepare Combined Training Set
    combined_train_path = os.path.join("adapt", "data", "processed", "retrain_combined_train.jsonl")
    shutil.copyfile(base_train_path, combined_train_path)

    # Convert approved feedback to decision examples if any
    if num_approved > 0:
        with open(approved_export_path, "r", encoding="utf-8") as f_in, open(combined_train_path, "a", encoding="utf-8") as f_out:
            for line in f_in:
                fb = json.loads(line)
                # Synthetic/augmented scenario for feedback item
                example = {
                    "example_id": f"fb_augmented_{fb['feedback_id']}",
                    "scenario_text": f"User reviewed scenario {fb['example_id']}. Verified optimal action: {fb['user_selected_action']}.",
                    "optimal_action": fb["user_selected_action"],
                    "ground_truth_ranking": [fb["user_selected_action"]],
                }
                f_out.write(json.dumps(example) + "\n")
        print(f"Successfully combined {num_approved} approved feedback items into training stream.")

    # 3. Train Candidate Checkpoint Offline
    candidate_dir = os.path.join(output_dir, "candidate_retrain")
    os.makedirs(candidate_dir, exist_ok=True)
    print("\n--- Initiating Offline Candidate Retraining ---")
    train_transformer(
        train_path=combined_train_path,
        val_path=val_path,
        tokenizer_path=tokenizer_path,
        output_dir=candidate_dir,
        epochs=4,
        batch_size=32,
        d_model=160,
        n_layers=4,
        n_heads=4,
        lr=5e-4,
        seed=101,  # Different seed for retrain run
    )

    candidate_ckpt = os.path.join(candidate_dir, "transformer_best.pt")

    # 4. Run Strict Regression Suite on Frozen Golden Test Set
    print("\n--- Running Golden Test Set Regression Suite on Candidate ---")
    candidate_eval = evaluate_test_set(
        test_path=test_path,
        tokenizer_path=tokenizer_path,
        checkpoint_path=candidate_ckpt,
    )
    cand_acc = candidate_eval["top1_accuracy"]
    cand_mrr = candidate_eval["mrr"]

    print("\n======================================================================")
    print("                     REGRESSION COMPARISON RESULTS                    ")
    print("======================================================================")
    print(f"Baseline Accuracy:  {base_acc * 100:.2f}% | Candidate: {cand_acc * 100:.2f}%")
    print(f"Baseline MRR:       {base_mrr:.4f}   | Candidate: {cand_mrr:.4f}")
    acc_diff = cand_acc - base_acc

    # Regression gate: accuracy must not drop significantly
    passed_regression = acc_diff >= -max_allowed_accuracy_drop

    status_tag = "APPROVED_FOR_DEPLOYMENT" if passed_regression else "REJECTED_REGRESSION_DETECTED"
    print(f"Gate Status: {status_tag} (Accuracy delta: {acc_diff * 100:+.2f}%)")

    report = {
        "status": status_tag,
        "passed_regression": passed_regression,
        "baseline_accuracy": base_acc,
        "candidate_accuracy": cand_acc,
        "accuracy_delta": round(acc_diff, 4),
        "baseline_mrr": base_mrr,
        "candidate_mrr": cand_mrr,
        "approved_feedback_count": num_approved,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    report_path = os.path.join("adapt", "experiments", "retraining_regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if passed_regression:
        approved_ckpt_path = os.path.join(output_dir, "transformer_approved.pt")
        shutil.copyfile(candidate_ckpt, approved_ckpt_path)
        print(f"Candidate passed gate! Copied approved weights to {approved_ckpt_path}")
    else:
        print("Candidate REJECTED! Production weights preserved intact.")

    return report


if __name__ == "__main__":
    b_train = os.path.join("adapt", "data", "processed", "train.jsonl")
    v_data = os.path.join("adapt", "data", "processed", "val.jsonl")
    t_data = os.path.join("adapt", "data", "processed", "test.jsonl")
    fb_store = os.path.join("adapt", "data", "feedback_staging.jsonl")
    tok_f = os.path.join("adapt", "artifacts", "tokenizer", "adapt_tokenizer.json")
    out_d = os.path.join("adapt", "artifacts", "checkpoints")

    run_offline_retraining(b_train, v_data, t_data, fb_store, out_d, tok_f)
