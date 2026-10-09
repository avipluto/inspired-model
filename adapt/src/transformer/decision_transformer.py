"""
Compact Decoder-Only Decision Language Model (PyTorch) for ADAPT.
Trained from random initialization.
Features:
- Causal Multi-Head Self-Attention
- Learned Positional Embeddings
- Pre-LayerNorm architecture
- GELU feed-forward networks
- Sequence-level candidate scoring & cross-entropy training
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerConfig":
        return cls(**data)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        context = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Compact causal decoder-only transformer initialized with random weights.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embeddings.weight

        # Action classification head (for fast multi-candidate discriminative scoring)
        self.action_head = nn.Linear(config.d_model, 4)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        input_ids: (B, T)
        attention_mask: (B, T) with 1 for real tokens and 0 for pad
        labels: (B, T) target next-token IDs for causal LM loss, or None
        """
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        hidden_states = self.ln_final(x)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for autoregressive causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    def get_pooled_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract hidden representation from last non-pad token."""
        B, T = input_ids.size()
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_embeddings(input_ids) + self.position_embeddings(positions))
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_final(x)

        if attention_mask is not None:
            # Gather last non-pad token index
            lengths = attention_mask.sum(dim=1).long() - 1
            lengths = torch.clamp(lengths, min=0)
            batch_idx = torch.arange(B, device=input_ids.device)
            return x[batch_idx, lengths]
        else:
            return x[:, -1, :]

    def score_candidate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute candidate action logits [MOVE, COMMUTE, NEGOTIATE_HYBRID, REQUEST_INFO]
        using the action projection head over the pooled context.
        """
        pooled = self.get_pooled_representation(input_ids, attention_mask=attention_mask)
        action_logits = self.action_head(pooled)
        return action_logits
