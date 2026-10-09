import torch
import pytest
from adapt.src.transformer.decision_transformer import DecisionTransformer, TransformerConfig


def test_transformer_forward_backward():
    config = TransformerConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        pad_token_id=0,
    )
    model = DecisionTransformer(config)

    x = torch.randint(1, 500, (2, 16))
    attn_mask = torch.ones((2, 16), dtype=torch.float32)

    logits, lm_loss = model(x, attention_mask=attn_mask, labels=x)
    assert logits.shape == (2, 16, 500)
    assert lm_loss is not None
    assert lm_loss.item() > 0.0

    action_logits = model.score_candidate_actions(x, attention_mask=attn_mask)
    assert action_logits.shape == (2, 4)

    action_loss = action_logits.sum()
    action_loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_causal_masking_invariance():
    config = TransformerConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
    )
    model = DecisionTransformer(config)
    model.eval()

    x1 = torch.tensor([[5, 10, 15, 20]])
    x2 = torch.tensor([[5, 10, 99, 20]])  # Token at index 2 differs

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    # Position 0 and 1 must be strictly unaffected by future tokens
    diff_0 = (out1[0, 0] - out2[0, 0]).abs().max().item()
    diff_1 = (out1[0, 1] - out2[0, 1]).abs().max().item()
    assert diff_0 < 1e-5
    assert diff_1 < 1e-5
