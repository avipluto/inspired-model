"""
Training script for the Structured Consequence Predictor (MLP).
Fits preprocessor, trains MLP on consequence targets, and compares against baseline.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adapt.src.structured_predictor.preprocessor import StructuredFeaturePreprocessor
from adapt.src.structured_predictor.mlp_predictor import ConsequenceMLP


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def load_dataset_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_targets(records):
    fatigues = []
    burnouts = []
    incidentals = []

    for r in records:
        cons = r.get("action_consequences", {})
        # Target based on commute option if available
        c_commute = cons.get("COMMUTE", {})
        fat = c_commute.get("fatigue_index")
        burn = c_commute.get("burnout_risk")

        if fat is None:
            fat = 3.0
        if burn is None:
            burn = 0.3

        # Incidental variance modeled from city cost tier
        tier = r.get("structured_facts", {}).get("city_cost_tier", "MEDIUM")
        tier_str = tier.value if hasattr(tier, "value") else str(tier)
        inc = 250.0 if tier_str == "HIGH" else (150.0 if tier_str == "MEDIUM" else 80.0)

        fatigues.append(float(fat))
        burnouts.append(float(burn))
        incidentals.append(float(inc))

    return (
        np.array(fatigues, dtype=np.float32),
        np.array(burnouts, dtype=np.float32),
        np.array(incidentals, dtype=np.float32),
    )


def train_mlp_predictor(
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    train_recs = load_dataset_records(train_path)
    val_recs = load_dataset_records(val_path)

    # 1. Fit Preprocessor
    preprocessor = StructuredFeaturePreprocessor()
    preprocessor.fit(train_recs)
    prep_path = os.path.join(output_dir, "preprocessor_state.json")
    preprocessor.save(prep_path)
    print(f"Saved preprocessor state to {prep_path}")

    # 2. Extract features and targets
    X_train = preprocessor.transform_batch(train_recs)
    y_fat_tr, y_burn_tr, y_inc_tr = extract_targets(train_recs)

    X_val = preprocessor.transform_batch(val_recs)
    y_fat_val, y_burn_val, y_inc_val = extract_targets(val_recs)

    # Simple Baseline: Mean predictor on validation set
    mean_fat = float(np.mean(y_fat_tr))
    mean_burn = float(np.mean(y_burn_tr))
    baseline_fat_mae = float(np.mean(np.abs(y_fat_val - mean_fat)))
    baseline_burn_mae = float(np.mean(np.abs(y_burn_val - mean_burn)))
    print(f"--- Baselines (Mean Predictor) ---")
    print(f"  Fatigue Baseline MAE: {baseline_fat_mae:.4f}")
    print(f"  Burnout Baseline MAE: {baseline_burn_mae:.4f}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_fat_tr),
        torch.from_numpy(y_burn_tr),
        torch.from_numpy(y_inc_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_fat_val),
        torch.from_numpy(y_burn_val),
        torch.from_numpy(y_inc_val),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model & Optimizer
    model = ConsequenceMLP(input_dim=16, hidden_dims=(64, 32, 16), dropout=0.15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    crit_fatigue = nn.SmoothL1Loss()
    crit_burnout = nn.BCELoss()
    crit_incidental = nn.MSELoss()

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(output_dir, "mlp_consequence_best.pt")
    history = []

    start_time = time.time()
    print("Beginning MLP Training...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, b_fat, b_burn, b_inc in train_loader:
            bx = bx.to(device)
            b_fat = b_fat.to(device)
            b_burn = b_burn.to(device)
            b_inc = b_inc.to(device)

            optimizer.zero_grad()
            preds = model(bx)

            loss_fat = crit_fatigue(preds["fatigue_index"], b_fat)
            loss_burn = crit_burnout(preds["burnout_risk"], b_burn)
            loss_inc = crit_incidental(preds["incidental_variance"], b_inc) * 1e-4

            loss = loss_fat + 2.0 * loss_burn + loss_inc
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_fat_errors = []
        val_burn_errors = []

        with torch.no_grad():
            for bx, b_fat, b_burn, b_inc in val_loader:
                bx = bx.to(device)
                b_fat = b_fat.to(device)
                b_burn = b_burn.to(device)
                b_inc = b_inc.to(device)

                preds = model(bx)
                l_fat = crit_fatigue(preds["fatigue_index"], b_fat)
                l_burn = crit_burnout(preds["burnout_risk"], b_burn)
                l_inc = crit_incidental(preds["incidental_variance"], b_inc) * 1e-4

                val_loss += (l_fat + 2.0 * l_burn + l_inc).item()

                val_fat_errors.extend((preds["fatigue_index"] - b_fat).abs().cpu().numpy())
                val_burn_errors.extend((preds["burnout_risk"] - b_burn).abs().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        fat_mae = float(np.mean(val_fat_errors))
        burn_mae = float(np.mean(val_burn_errors))

        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / len(train_loader), 4),
            "val_loss": round(avg_val_loss, 4),
            "val_fatigue_mae": round(fat_mae, 4),
            "val_burnout_mae": round(burn_mae, 4),
        })

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Val Loss: {avg_val_loss:.4f} | "
                f"Fatigue MAE: {fat_mae:.4f} (baseline: {baseline_fat_mae:.4f}) | "
                f"Burnout MAE: {burn_mae:.4f} (baseline: {baseline_burn_mae:.4f})",
                flush=True,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_fatigue_mae": fat_mae,
                    "val_burnout_mae": burn_mae,
                },
                best_ckpt_path,
            )

    elapsed = time.time() - start_time
    print(f"MLP Training Complete in {elapsed:.1f}s. Best Checkpoint saved to {best_ckpt_path}")

    # Save meta
    meta_path = os.path.join(output_dir, "mlp_train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "ConsequenceMLP",
                "epochs": epochs,
                "duration_seconds": round(elapsed, 2),
                "best_val_loss": round(best_val_loss, 4),
                "baseline_fatigue_mae": round(baseline_fat_mae, 4),
                "baseline_burnout_mae": round(baseline_burn_mae, 4),
                "history": history,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    train_f = os.path.join("adapt", "data", "processed", "train.jsonl")
    val_f = os.path.join("adapt", "data", "processed", "val.jsonl")
    out_dir = os.path.join("adapt", "artifacts", "checkpoints")

    train_mlp_predictor(train_f, val_f, out_dir, epochs=30, batch_size=64, lr=2e-3, seed=42)
