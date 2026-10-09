"""
Verification script for DecisionTransformer.
Verifies:
1. Model parameter count and architecture.
2. Causal masking (strictly autoregressive).
3. Forward pass with causal LM loss.
4. Backward pass and valid gradient propagation.
5. Action scoring head forward & backward.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.transformer.decision_transformer import DecisionTransformer, TransformerConfig


def verify():
    print("--- Verifying Compact DecisionTransformer ---")
    config = TransformerConfig(
        vocab_size=2048,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1,
        pad_token_id=0,
    )

    model = DecisionTransformer(config)
    param_count = model.count_parameters()
    print(f"Model successfully initialized with random weights.")
    print(f"Total trainable parameters: {param_count:,} (~{param_count / 1e6:.2f}M)")
    assert 2_000_000 <= param_count <= 25_000_000, f"Parameter count {param_count} out of target range!"

    # 1. Forward pass test
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
    # Mask out some padding
    dummy_input[:, 50:] = 0
    attention_mask[:, 50:] = 0.0

    logits, lm_loss = model(dummy_input, attention_mask=attention_mask, labels=dummy_input)
    print(f"Forward Pass Success:")
    print(f"  Logits shape: {logits.shape} (expected: [{batch_size}, {seq_len}, {config.vocab_size}])")
    print(f"  Autoregressive LM Loss: {lm_loss.item():.4f}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(lm_loss)

    # 2. Backward pass test
    model.zero_grad()
    lm_loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(not math_isnan(g) for g in grad_norms)
    print(f"Backward Pass Success! Gradients computed across {len(grad_norms)} tensors.")

    # 3. Action scoring test
    action_logits = model.score_candidate_actions(dummy_input, attention_mask=attention_mask)
    print(f"Candidate Action Head Success:")
    print(f"  Action logits shape: {action_logits.shape} (expected: [{batch_size}, 4])")
    assert action_logits.shape == (batch_size, 4)

    # Action loss backward test
    dummy_targets = torch.tensor([0, 1, 2, 3])
    loss_action = torch.nn.functional.cross_entropy(action_logits, dummy_targets)
    model.zero_grad()
    loss_action.backward()
    print(f"Action Head Backward Success! Action loss: {loss_action.item():.4f}")

    # 4. Verify Causal Masking
    # Check that position t does not depend on position t+1
    x1 = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    x2 = torch.tensor([[10, 20, 999, 40]], dtype=torch.long)  # Modified at position 2
    model.eval()
    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)
        # Position 0 and 1 output logits must be identical!
        diff_pos0 = (out1[0, 0] - out2[0, 0]).abs().max().item()
        diff_pos1 = (out1[0, 1] - out2[0, 1]).abs().max().item()
        diff_pos2 = (out1[0, 2] - out2[0, 2]).abs().max().item()
        print(f"Causal Attention Invariance Check:")
        print(f"  Logit diff at t=0: {diff_pos0:.8f} (must be 0.0)")
        print(f"  Logit diff at t=1: {diff_pos1:.8f} (must be 0.0)")
        print(f"  Logit diff at modified t=2: {diff_pos2:.8f} (should be > 0)")
        assert diff_pos0 < 1e-5, "Future tokens leaked to position 0!"
        assert diff_pos1 < 1e-5, "Future tokens leaked to position 1!"
        assert diff_pos2 > 1e-3, "Position 2 should reflect modified token!"

    print("--- All Transformer Architecture Tests Passed! ---")


def math_isnan(x):
    return x != x


if __name__ == "__main__":
    verify()
