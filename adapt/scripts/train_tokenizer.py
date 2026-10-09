"""
Script to train the BPE tokenizer on scenario texts.
Extracts text from train.jsonl and domain vocabulary corpus, trains and saves the tokenizer.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.tokenizer.bpe_tokenizer import AdaptTokenizer


def train_tokenizer(data_path: str, output_path: str, vocab_size: int = 2048):
    temp_corpus_path = os.path.join("adapt", "data", "processed", "tokenizer_corpus.txt")
    os.makedirs(os.path.dirname(temp_corpus_path), exist_ok=True)

    print(f"Extracting corpus from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as fin, open(temp_corpus_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            # Add scenario text
            fout.write(data["scenario_text"] + "\n")
            # Add action candidate texts
            for act in data.get("candidate_actions", []):
                fout.write(f"Candidate action: {act}. Recommendation decision: {act}.\n")
            # Add ranking strings
            ranking = " > ".join(data.get("ground_truth_ranking", []))
            fout.write(f"Optimal ranking: {ranking}\n")

    tokenizer = AdaptTokenizer()
    print(f"Training tokenizer with target vocab_size={vocab_size}...")
    tokenizer.train_from_files([temp_corpus_path], vocab_size=vocab_size, save_path=output_path)
    print("Tokenizer training complete!")

    # Quick sanity check
    sample_text = "Candidate received an offer for a Software Engineering Intern in Seattle. Optimal action: MOVE."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"Sanity Check:")
    print(f"  Original: {sample_text}")
    print(f"  Encoded token IDs (len={len(encoded)}): {encoded}")
    print(f"  Decoded:  {decoded}")


if __name__ == "__main__":
    train_data = os.path.join("adapt", "data", "processed", "train.jsonl")
    save_file = os.path.join("adapt", "artifacts", "tokenizer", "adapt_tokenizer.json")
    train_tokenizer(train_data, save_file, vocab_size=2048)
