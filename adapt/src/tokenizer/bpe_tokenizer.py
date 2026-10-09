"""
Custom BPE Tokenizer for ADAPT.
Trains a Byte-Pair Encoding tokenizer on the scenario dataset and domain corpus,
with explicit special tokens for decision modeling.
"""

import os
from typing import List, Union
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


class AdaptTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<action>"]

    def __init__(self, tokenizer_path: str = None):
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load(tokenizer_path)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("<pad>")

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id("<s>")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.token_to_id("</s>")

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id("<unk>")

    @property
    def action_token_id(self) -> int:
        return self.tokenizer.token_to_id("<action>")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train_from_files(
        self,
        files: List[str],
        vocab_size: int = 2048,
        save_path: str = None,
    ):
        """Train a BPE tokenizer on a list of text files."""
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files, trainer)
        self.tokenizer = tokenizer

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.tokenizer.save(save_path)
            print(f"Saved tokenizer with vocab size {self.vocab_size} to {save_path}")

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.tokenizer.save(save_path)

    def load(self, load_path: str):
        self.tokenizer = Tokenizer.from_file(load_path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        return self.tokenizer.decode(ids)
