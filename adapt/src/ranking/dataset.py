"""
PyTorch Dataset and dynamic-padding DataLoader for Decision Transformer candidate ranking.
Dynamically pads batches to maximum token length in the batch to maximize training efficiency on CPU.
"""

import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

from adapt.src.tokenizer.bpe_tokenizer import AdaptTokenizer
from adapt.src.data_schemas import ActionType


ACTION_TO_IDX = {
    ActionType.MOVE.value: 0,
    ActionType.COMMUTE.value: 1,
    ActionType.NEGOTIATE_HYBRID.value: 2,
    ActionType.REQUEST_INFO.value: 3,
}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


class ScenarioRankingDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AdaptTokenizer,
        max_seq_len: int = 192,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        text = rec["scenario_text"]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]

        target_action = rec["optimal_action"]
        target_idx = ACTION_TO_IDX[target_action]

        return {
            "token_ids": tokens,
            "target_idx": target_idx,
            "example_id": rec["example_id"],
            "optimal_action": target_action,
            "ground_truth_ranking": rec["ground_truth_ranking"],
        }


def dynamic_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    max_len = max(len(item["token_ids"]) for item in batch)
    
    padded_inputs = []
    attn_masks = []
    targets = []
    example_ids = []
    optimal_actions = []
    rankings = []

    for item in batch:
        toks = item["token_ids"]
        pad_len = max_len - len(toks)
        padded_inputs.append(toks + [pad_token_id] * pad_len)
        attn_masks.append([1.0] * len(toks) + [0.0] * pad_len)
        targets.append(item["target_idx"])
        example_ids.append(item["example_id"])
        optimal_actions.append(item["optimal_action"])
        rankings.append(item["ground_truth_ranking"])

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.float32),
        "target_idx": torch.tensor(targets, dtype=torch.long),
        "example_id": example_ids,
        "optimal_action": optimal_actions,
        "ground_truth_ranking": rankings,
    }
