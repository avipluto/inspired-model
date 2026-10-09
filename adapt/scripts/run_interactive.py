"""
Interactive CLI for ADAPT Decision Engine.
Allows entering customized internship parameters to receive instant audited recommendations.
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.data_schemas import (
    ActionType,
    CityCostTier,
    CommuteMode,
    StructuredFacts,
    UserConstraints,
    UserPreferences,
)
from adapt.src.structured_predictor.preprocessor import StructuredFeaturePreprocessor
from adapt.src.structured_predictor.mlp_predictor import ConsequenceMLP
from adapt.src.consequences.calculator import ConsequenceCalculator
from adapt.src.decision_policy.aggregator import DecisionAggregator


def main():
    print("=" * 70)
    print("       ADAPT: INTERACTIVE DECISION SUPPORT SYSTEM (LIVE ENGINE)        ")
    print("=" * 70)

    # Initialize models
    prep_path = os.path.join("adapt", "artifacts", "checkpoints", "preprocessor_state.json")
    mlp_path = os.path.join("adapt", "artifacts", "checkpoints", "mlp_consequence_best.pt")

    preprocessor = StructuredFeaturePreprocessor()
    preprocessor.load(prep_path)

    mlp_model = ConsequenceMLP(input_dim=16, hidden_dims=(64, 32, 16))
    ckpt = torch.load(mlp_path, map_location="cpu")
    mlp_model.load_state_dict(ckpt["model_state_dict"])
    mlp_model.eval()

    calculator = ConsequenceCalculator(mlp_model=mlp_model, preprocessor=preprocessor)
    aggregator = DecisionAggregator(calculator=calculator)

    print("\nEvaluating default candidate profile:")
    print("  - Role: Machine Learning Intern")
    print("  - City: Seattle (Tier: HIGH)")
    print("  - Monthly Stipend: $6,500")
    print("  - Rent near office: $1,750")
    print("  - Daily commute: 90 mins round-trip ($250/mo)")
    print("  - Minimum reserve budget: $400/mo")

    facts = StructuredFacts(
        monthly_stipend=6500.0,
        base_living_expenses=800.0,
        relocation_rent=1750.0,
        commute_daily_minutes=90.0,
        commute_monthly_cost=250.0,
        hybrid_days_in_office=2,
        employer_flexibility_score=0.7,
        city_cost_tier=CityCostTier.HIGH,
        commute_mode=CommuteMode.TRANSIT,
    )
    constraints = UserConstraints(min_savings_buffer=400.0, max_acceptable_commute_minutes=120.0)
    preferences = UserPreferences(financial_weight=0.35, work_life_balance_weight=0.45, career_growth_weight=0.20)

    result = aggregator.evaluate_decision(
        scenario_text="Machine learning intern offer in Seattle with relocation vs commuting dilemma.",
        facts=facts,
        constraints=constraints,
        preferences=preferences,
    )

    print("\n" + "-" * 70)
    print(f"RECOMMENDATION:     {result.recommendation.value}")
    print(f"DECISION STATUS:    {result.decision_status}")
    print(f"ACTION RANKINGS:    {[a.value for a in result.ranked_actions]}")
    print("-" * 70)
    print("CANDIDATE ACTION EVALUATIONS:")
    for act_name, eval_res in result.evaluations.items():
        status = "FEASIBLE" if eval_res.is_feasible else "INFEASIBLE"
        print(f"  [{act_name:16s}] Status: {status:10s} | Utility: {eval_res.utility_score:+.4f}")
        print(f"     -> {eval_res.trade_off_summary}")

    print("\nSUPPORTING FACTORS:")
    for sf in result.supporting_factors:
        print(f"  + {sf}")

    print("\nTRADE-OFFS & ASSUMPTIONS:")
    for tf in result.trade_offs_explained:
        print(f"  * {tf}")
    for asmp in result.assumptions_made:
        print(f"  # {asmp}")

    print("=" * 70)


if __name__ == "__main__":
    main()
