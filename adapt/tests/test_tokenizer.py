import os
import pytest
from adapt.src.tokenizer.bpe_tokenizer import AdaptTokenizer


def test_tokenizer_load_and_encode():
    tok_path = os.path.join("adapt", "artifacts", "tokenizer", "adapt_tokenizer.json")
    if not os.path.exists(tok_path):
        pytest.skip("Tokenizer artifact not yet generated.")

    tokenizer = AdaptTokenizer(tok_path)
    assert tokenizer.vocab_size == 2048

    text = "Candidate received an offer for an internship in New York."
    ids = tokenizer.encode(text, add_special_tokens=True)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id

    decoded = tokenizer.decode(ids)
    assert "internship" in decoded.lower()
