"""
PyTorch Multi-Layer Perceptron (MLP) for Structured Consequence Prediction.
Predicts measurable empirical consequences:
- Commute fatigue index [0, 10]
- Burnout risk probability [0, 1]
- Incidental dollar expense variance [0, 500]
Includes Monte-Carlo Dropout for uncertainty estimation.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsequenceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Output heads:
        # Head 1: Fatigue index [0, 10] (ReLU bounded by min-max)
        self.fatigue_head = nn.Linear(prev_dim, 1)

        # Head 2: Burnout risk probability [0, 1] (Sigmoid)
        self.burnout_head = nn.Linear(prev_dim, 1)

        # Head 3: Incidental variance [0, 500]
        self.incidental_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        fatigue = torch.clamp(self.fatigue_head(features), 0.0, 10.0)
        burnout = torch.sigmoid(self.burnout_head(features))
        incidental = F.softplus(self.incidental_head(features))

        return {
            "fatigue_index": fatigue.squeeze(-1),
            "burnout_risk": burnout.squeeze(-1),
            "incidental_variance": incidental.squeeze(-1),
        }

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 25,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Monte Carlo Dropout inference to calculate mean prediction and epistemic standard deviation.
        x: 1D or 2D tensor of shape (1, 16)
        Returns: {metric: (mean, std)}
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Enable dropout during inference for MC sampling
        self.train()
        fatigue_samples = []
        burnout_samples = []
        incidental_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(x)
                fatigue_samples.append(out["fatigue_index"].item())
                burnout_samples.append(out["burnout_risk"].item())
                incidental_samples.append(out["incidental_variance"].item())

        self.eval()
        return {
            "fatigue_index": (float(torch.tensor(fatigue_samples).mean().item()), float(torch.tensor(fatigue_samples).std().item())),
            "burnout_risk": (float(torch.tensor(burnout_samples).mean().item()), float(torch.tensor(burnout_samples).std().item())),
            "incidental_variance": (float(torch.tensor(incidental_samples).mean().item()), float(torch.tensor(incidental_samples).std().item())),
        }
