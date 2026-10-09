"""
Dataset validation script for ADAPT.
Verifies schema compliance, distribution of actions, missing-fact clarification cases,
and verifies zero duplicate leakage between train and test splits.
"""

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.data_schemas import DecisionExample, ActionType


def validate_splits(data_dir: str):
    print("--- Validating Dataset Splits ---")
    splits = ["train", "val", "test"]
    seen_texts = set()
    total_records = 0
    
    for split in splits:
        file_path = os.path.join(data_dir, f"{split}.jsonl")
        assert os.path.exists(file_path), f"Missing split file: {file_path}"
        
        records = []
        action_counts = Counter()
        clarify_count = 0
        duplicate_count = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Pydantic schema validation
                ex = DecisionExample.model_validate(data)
                records.append(ex)
                action_counts[ex.optimal_action.value] += 1
                if ex.should_clarify:
                    clarify_count += 1
                if ex.scenario_text in seen_texts:
                    duplicate_count += 1
                if split == "train":
                    seen_texts.add(ex.scenario_text)

        total_records += len(records)
        print(f"[{split.upper()}] Count: {len(records)}")
        print(f"  Optimal Actions: {dict(action_counts)}")
        print(f"  Clarification cases: {clarify_count} ({clarify_count / len(records) * 100:.1f}%)")
        if split != "train":
            print(f"  Leakage / Exact Duplicates with Train: {duplicate_count}")
            assert duplicate_count == 0, f"Leakage detected in {split}!"

    print(f"Validation Passed! Total verified records: {total_records}")


if __name__ == "__main__":
    validate_splits(os.path.join("adapt", "data", "processed"))
