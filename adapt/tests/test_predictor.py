import torch
import numpy as np
import pytest
from adapt.src.structured_predictor.preprocessor import StructuredFeaturePreprocessor
from adapt.src.structured_predictor.mlp_predictor import ConsequenceMLP


def test_preprocessor_and_mlp():
    records = [
        {
            "structured_facts": {
                "monthly_stipend": 5000.0,
                "base_living_expenses": 600.0,
                "relocation_rent": 1400.0,
                "commute_daily_minutes": 80.0,
                "commute_monthly_cost": 200.0,
                "employer_flexibility_score": 0.5,
                "hybrid_days_in_office": 2,
                "city_cost_tier": "HIGH",
                "commute_mode": "TRANSIT",
            }
        },
        {
            "structured_facts": {
                "monthly_stipend": None,  # Missing
                "base_living_expenses": 500.0,
                "relocation_rent": 1000.0,
                "commute_daily_minutes": 40.0,
                "commute_monthly_cost": 150.0,
                "employer_flexibility_score": 0.8,
                "hybrid_days_in_office": 3,
                "city_cost_tier": "LOW",
                "commute_mode": "DRIVING",
            }
        },
    ]

    prep = StructuredFeaturePreprocessor()
    prep.fit(records)
    assert prep.is_fitted

    vec1 = prep.transform_single(records[0]["structured_facts"])
    vec2 = prep.transform_single(records[1]["structured_facts"])
    assert len(vec1) == 16
    assert len(vec2) == 16
    assert vec1[7] == 0.0  # stipend not missing
    assert vec2[7] == 1.0  # stipend missing flag

    model = ConsequenceMLP(input_dim=16, hidden_dims=(32, 16))
    x = torch.from_numpy(vec1)
    out = model(x.unsqueeze(0))
    assert 0.0 <= out["fatigue_index"].item() <= 10.0
    assert 0.0 <= out["burnout_risk"].item() <= 1.0

    # MC Dropout uncertainty test
    unc = model.predict_with_uncertainty(x, num_samples=10)
    assert "fatigue_index" in unc
    assert unc["fatigue_index"][1] >= 0.0
