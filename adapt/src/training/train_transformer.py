"""
Reproducible Training Pipeline for DecisionTransformer Candidate Ranking.
Trains the compact transformer from scratch (random initialization) on CPU with dynamic padding.
Logs loss curves, metrics, and saves checkpoints with complete metadata.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adapt.src.tokenizer.bpe_tokenizer import AdaptTokenizer
from adapt.src.transformer.decision_transformer import DecisionTransformer, TransformerConfig
from adapt.src.ranking.dataset import ScenarioRankingDataset, dynamic_collate_fn, IDX_TO_ACTION
from adapt.src.ranking.ranking_metrics import compute_top1_accuracy, compute_mrr, compute_ndcg_at_k


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transformer(
    train_path: str,
    val_path: str,
    tokenizer_path: str,
    output_dir: str,
    epochs: int = 4,
    batch_size: int = 32,
    d_model: int = 160,
    n_layers: int = 4,
    n_heads: int = 4,
    lr: float = 5e-4,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    print(f"Using training device: {device}", flush=True)

    # Load tokenizer
    tokenizer = AdaptTokenizer(tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}", flush=True)

    # Create datasets
    train_dataset = ScenarioRankingDataset(train_path, tokenizer, max_seq_len=160)
    val_dataset = ScenarioRankingDataset(val_path, tokenizer, max_seq_len=160)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    # Initialize transformer from random initialization
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        max_seq_len=256,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = DecisionTransformer(config).to(device)
    param_count = model.count_parameters()
    print(f"Initialized scratch DecisionTransformer with {param_count:,} parameters (~{param_count / 1e6:.2f}M).", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_mrr = -1.0
    best_checkpoint_path = os.path.join(output_dir, "transformer_best.pt")
    history = []

    start_time = time.time()
    print(f"Beginning Training Loop ({len(train_loader)} batches/epoch)...", flush=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_idx = batch["target_idx"].to(device)

            optimizer.zero_grad()

            action_logits = model.score_candidate_actions(input_ids, attention_mask=attention_mask)
            loss = criterion(action_logits, target_idx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if (b_idx + 1) % 15 == 0:
                print(f"  [Epoch {epoch} | Step {b_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        all_pred_top = []
        all_true_top = []
        all_pred_rankings = []
        all_ideal_rankings = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_idx = batch["target_idx"].to(device)

                action_logits = model.score_candidate_actions(input_ids, attention_mask=attention_mask)
                loss_v = criterion(action_logits, target_idx)
                val_loss += loss_v.item()

                sorted_indices = torch.argsort(action_logits, dim=-1, descending=True).cpu().numpy()
                for i in range(len(batch["target_idx"])):
                    pred_actions = [IDX_TO_ACTION[idx] for idx in sorted_indices[i]]
                    all_pred_rankings.append(pred_actions)
                    all_pred_top.append(pred_actions[0])
                    all_true_top.append(batch["optimal_action"][i])
                    all_ideal_rankings.append(batch["ground_truth_ranking"][i])

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = compute_top1_accuracy(all_pred_top, all_true_top)
        val_mrr = compute_mrr(all_pred_rankings, all_true_top)
        val_ndcg = compute_ndcg_at_k(all_pred_rankings, all_ideal_rankings, k=4)
        epoch_sec = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_top1_acc": round(val_acc, 4),
            "val_mrr": round(val_mrr, 4),
            "val_ndcg4": round(val_ndcg, 4),
            "epoch_duration_sec": round(epoch_sec, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val MRR: {val_mrr:.4f} | "
            f"Val NDCG: {val_ndcg:.4f}",
            flush=True,
        )

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "val_metrics": epoch_record,
                },
                best_checkpoint_path,
            )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.1f}s. Best Val MRR: {best_val_mrr:.4f}", flush=True)

    # Save metadata log
    meta_path = os.path.join(output_dir, "transformer_train_meta.json")
    meta_info = {
        "model_name": "DecisionTransformer",
        "parameters": model.count_parameters(),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "duration_seconds": round(elapsed_time, 2),
        "best_val_mrr": best_val_mrr,
        "best_checkpoint": best_checkpoint_path,
        "history": history,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"Saved metadata log to {meta_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=160)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_file = os.path.join("adapt", "data", "processed", "train.jsonl")
    val_file = os.path.join("adapt", "data", "processed", "val.jsonl")
    tok_file = os.path.join("adapt", "artifacts", "tokenizer", "adapt_tokenizer.json")
    out_dir = os.path.join("adapt", "artifacts", "checkpoints")

    train_transformer(
        train_path=train_file,
        val_path=val_file,
        tokenizer_path=tok_file,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        seed=args.seed,
    )
