"""
Deterministic Consequence Calculator for ADAPT.
Performs exact arithmetic for disposable income, remaining funds, and commute time.
Integrates with the trained ConsequenceMLP for empirical non-linear risks.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np

from adapt.src.data_schemas import (
    ActionType,
    StructuredFacts,
    ActionConsequences,
)
from adapt.src.structured_predictor.preprocessor import StructuredFeaturePreprocessor
from adapt.src.structured_predictor.mlp_predictor import ConsequenceMLP


class ConsequenceCalculator:
    def __init__(
        self,
        mlp_model: Optional[ConsequenceMLP] = None,
        preprocessor: Optional[StructuredFeaturePreprocessor] = None,
    ):
        self.mlp_model = mlp_model
        self.preprocessor = preprocessor

    def compute_action_consequences(
        self,
        facts: StructuredFacts,
    ) -> Dict[str, ActionConsequences]:
        """Compute consequences for all 4 candidate actions."""
        consequences: Dict[str, ActionConsequences] = {}

        stipend = facts.monthly_stipend
        base_living = facts.base_living_expenses or 0.0
        rent = facts.relocation_rent
        commute_mins = facts.commute_daily_minutes
        commute_cost = facts.commute_monthly_cost or 0.0
        hybrid_days = facts.hybrid_days_in_office or 2

        # Predict ML consequence estimates if model is provided
        mlp_fatigue, mlp_fatigue_std = 3.0, 0.2
        mlp_burnout, mlp_burnout_std = 0.25, 0.05
        if self.mlp_model is not None and self.preprocessor is not None:
            feat_vec = self.preprocessor.transform_single(facts.model_dump())
            unc_dict = self.mlp_model.predict_with_uncertainty(torch.from_numpy(feat_vec), num_samples=15)
            mlp_fatigue, mlp_fatigue_std = unc_dict["fatigue_index"]
            mlp_burnout, mlp_burnout_std = unc_dict["burnout_risk"]

        # 1. MOVE
        if rent is not None and stipend is not None:
            move_rem = stipend - (rent + base_living)
            consequences[ActionType.MOVE.value] = ActionConsequences(
                net_remaining_funds=round(move_rem, 2),
                daily_commute_minutes=15.0,
                fatigue_index=1.2,
                burnout_risk=0.15,
                constraint_violations=[],
                is_feasible=True,
            )
        else:
            consequences[ActionType.MOVE.value] = ActionConsequences(
                constraint_violations=["MISSING_DATA: Missing rent or stipend information"],
                is_feasible=False,
            )

        # 2. COMMUTE
        if commute_mins is not None and stipend is not None:
            comm_rem = stipend - (commute_cost + base_living)
            # Combine empirical ML prediction with deterministic minutes
            fatigue = min(10.0, max(1.0, mlp_fatigue))
            burnout = min(1.0, max(0.05, mlp_burnout))
            consequences[ActionType.COMMUTE.value] = ActionConsequences(
                net_remaining_funds=round(comm_rem, 2),
                daily_commute_minutes=round(commute_mins, 1),
                fatigue_index=round(fatigue, 2),
                burnout_risk=round(burnout, 2),
                constraint_violations=[],
                is_feasible=True,
            )
        else:
            consequences[ActionType.COMMUTE.value] = ActionConsequences(
                constraint_violations=["MISSING_DATA: Missing commute or stipend information"],
                is_feasible=False,
            )

        # 3. NEGOTIATE_HYBRID
        if commute_mins is not None and stipend is not None:
            hybrid_fraction = hybrid_days / 5.0
            hyb_rem = stipend - (commute_cost * hybrid_fraction + base_living)
            hyb_commute = commute_mins * hybrid_fraction
            hyb_fatigue = min(10.0, max(1.0, mlp_fatigue * hybrid_fraction))
            hyb_burnout = min(1.0, max(0.05, mlp_burnout * hybrid_fraction))
            consequences[ActionType.NEGOTIATE_HYBRID.value] = ActionConsequences(
                net_remaining_funds=round(hyb_rem, 2),
                daily_commute_minutes=round(hyb_commute, 1),
                fatigue_index=round(hyb_fatigue, 2),
                burnout_risk=round(hyb_burnout, 2),
                constraint_violations=[],
                is_feasible=True,
            )
        else:
            consequences[ActionType.NEGOTIATE_HYBRID.value] = ActionConsequences(
                constraint_violations=["MISSING_DATA: Missing commute or stipend information"],
                is_feasible=False,
            )

        # 4. REQUEST_INFO
        consequences[ActionType.REQUEST_INFO.value] = ActionConsequences(
            net_remaining_funds=0.0,
            daily_commute_minutes=0.0,
            fatigue_index=0.0,
            burnout_risk=0.05,
            constraint_violations=[],
            is_feasible=True,
        )

        return consequences
