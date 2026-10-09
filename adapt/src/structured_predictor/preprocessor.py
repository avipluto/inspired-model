"""
Feature Preprocessor for Structured Consequence Predictor.
Handles:
- Numerical feature normalization (mean/std scaling).
- Categorical one-hot encoding.
- Missing-data imputation with indicator flags.
- Complete serialization of preprocessor state for consistent inference.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


NUMERICAL_KEYS = [
    "monthly_stipend",
    "base_living_expenses",
    "relocation_rent",
    "commute_daily_minutes",
    "commute_monthly_cost",
    "employer_flexibility_score",
    "hybrid_days_in_office",
]

TIER_MAP = {"LOW": [1, 0, 0], "MEDIUM": [0, 1, 0], "HIGH": [0, 0, 1]}
COMMUTE_MAP = {"TRANSIT": [1, 0, 0], "DRIVING": [0, 1, 0], "COMBINED": [0, 0, 1]}


class StructuredFeaturePreprocessor:
    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, records: List[Dict[str, Any]]):
        """Compute medians, means, and stds for numerical features across records."""
        for key in NUMERICAL_KEYS:
            vals = []
            for r in records:
                facts = r.get("structured_facts", {})
                v = facts.get(key)
                if v is not None:
                    vals.append(float(v))

            if len(vals) == 0:
                vals = [0.0]

            self.medians[key] = float(np.median(vals))
            self.means[key] = float(np.mean(vals))
            s = float(np.std(vals))
            self.stds[key] = s if s > 1e-6 else 1.0

        self.is_fitted = True

    def transform_single(self, facts: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single structured facts dictionary into a standardized 1D feature vector.
        Features layout (16 total):
        [0..6]  Standardized numerical values
        [7..9]  Missing indicator flags (stipend, rent, commute)
        [10..12] One-hot city cost tier (LOW, MEDIUM, HIGH)
        [13..15] One-hot commute mode (TRANSIT, DRIVING, COMBINED)
        """
        assert self.is_fitted, "Preprocessor must be fitted before transform."
        feature_vec = []

        # 1. Numerical features with imputation
        for key in NUMERICAL_KEYS:
            v = facts.get(key)
            if v is None:
                # Impute with median
                imputed_val = self.medians[key]
            else:
                imputed_val = float(v)
            # Standardize
            normed = (imputed_val - self.means[key]) / self.stds[key]
            feature_vec.append(normed)

        # 2. Missing indicator flags
        feature_vec.append(1.0 if facts.get("monthly_stipend") is None else 0.0)
        feature_vec.append(1.0 if facts.get("relocation_rent") is None else 0.0)
        feature_vec.append(1.0 if facts.get("commute_daily_minutes") is None else 0.0)

        # 3. Categorical one-hot
        tier = str(facts.get("city_cost_tier", "MEDIUM")).upper()
        if hasattr(tier, "value"):
            tier = tier.value
        feature_vec.extend(TIER_MAP.get(tier, [0, 1, 0]))

        cmode = str(facts.get("commute_mode", "TRANSIT")).upper()
        if hasattr(cmode, "value"):
            cmode = cmode.value
        feature_vec.extend(COMMUTE_MAP.get(cmode, [1, 0, 0]))

        return np.array(feature_vec, dtype=np.float32)

    def transform_batch(self, records: List[Dict[str, Any]]) -> np.ndarray:
        vectors = [self.transform_single(r.get("structured_facts", {})) for r in records]
        return np.vstack(vectors)

    def save(self, filepath: str):
        state = {
            "means": self.means,
            "stds": self.stds,
            "medians": self.medians,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.means = state["means"]
        self.stds = state["stds"]
        self.medians = state["medians"]
        self.is_fitted = state["is_fitted"]
