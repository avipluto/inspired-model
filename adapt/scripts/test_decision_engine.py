"""
Interactive validation and demonstration of ADAPT Decision Engine.
Demonstrates:
1. Feasible relocation scenario.
2. Preference sensitivity: changing weights changes the ranking.
3. Hard constraint rejection: budget violation and commute limit.
4. Clarification trigger on missing facts.
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


def run_demonstration():
    print("======================================================================")
    print("           ADAPT DECISION AGGREGATOR DEMONSTRATION & TEST              ")
    print("======================================================================")

    # Load trained ConsequenceMLP
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

    # -------------------------------------------------------------
    # Case 1: Standard balanced scenario favoring MOVE
    # -------------------------------------------------------------
    facts_1 = StructuredFacts(
        monthly_stipend=6500.0,
        base_living_expenses=800.0,
        relocation_rent=1700.0,
        commute_daily_minutes=95.0,
        commute_monthly_cost=300.0,
        hybrid_days_in_office=2,
        employer_flexibility_score=0.6,
        city_cost_tier=CityCostTier.HIGH,
        commute_mode=CommuteMode.TRANSIT,
    )
    constraints_1 = UserConstraints(min_savings_buffer=400.0, max_acceptable_commute_minutes=120.0)
    pref_wlb = UserPreferences(financial_weight=0.2, work_life_balance_weight=0.5, career_growth_weight=0.3)

    out_1 = aggregator.evaluate_decision(
        scenario_text="High stipend Seattle internship.",
        facts=facts_1,
        constraints=constraints_1,
        preferences=pref_wlb,
    )

    print("\n[CASE 1: High Work-Life Balance Priority]")
    print(f"Status:         {out_1.decision_status}")
    print(f"Recommendation: {out_1.recommendation.value}")
    print(f"Rankings:       {[a.value for a in out_1.ranked_actions]}")
    print(f"Supporting:     {out_1.supporting_factors[0]}")
    assert out_1.recommendation == ActionType.MOVE

    # -------------------------------------------------------------
    # Case 2: Preference Shift - Heavy Financial Priority
    # -------------------------------------------------------------
    pref_fin = UserPreferences(financial_weight=0.75, work_life_balance_weight=0.15, career_growth_weight=0.10)
    out_2 = aggregator.evaluate_decision(
        scenario_text="High stipend Seattle internship with extreme frugality.",
        facts=facts_1,
        constraints=constraints_1,
        preferences=pref_fin,
    )
    print("\n[CASE 2: Same scenario, shifted to 75% Financial Priority]")
    print(f"Recommendation: {out_2.recommendation.value}")
    print(f"Rankings:       {[a.value for a in out_2.ranked_actions]}")
    # Under financial priority, avoiding rent via hybrid/commute rises to top
    assert out_2.recommendation in [ActionType.NEGOTIATE_HYBRID, ActionType.COMMUTE]
    print(f"Preference shift verified: Recommendation changed to {out_2.recommendation.value}!")

    # -------------------------------------------------------------
    # Case 3: Hard Constraint Enforcement - Commute limit exceeded
    # -------------------------------------------------------------
    constraints_commute = UserConstraints(min_savings_buffer=400.0, max_acceptable_commute_minutes=60.0)
    out_3 = aggregator.evaluate_decision(
        scenario_text="Strict commute limit of 60 mins.",
        facts=facts_1,
        constraints=constraints_commute,
        preferences=pref_fin,
    )
    print("\n[CASE 3: Strict Commute Constraint (max 60m, actual commute 95m)]")
    print(f"COMMUTE Feasible: {out_3.evaluations['COMMUTE'].is_feasible}")
    print(f"COMMUTE Violations: {out_3.evaluations['COMMUTE'].constraint_violations}")
    assert not out_3.evaluations["COMMUTE"].is_feasible
    print("Hard constraint verified: COMMUTE was rejected due to commute limit violation!")

    # -------------------------------------------------------------
    # Case 4: Hard Constraint Enforcement - Budget violation on MOVE
    # -------------------------------------------------------------
    facts_tight = StructuredFacts(
        monthly_stipend=2500.0,
        base_living_expenses=700.0,
        relocation_rent=1700.0,
        commute_daily_minutes=50.0,
        commute_monthly_cost=150.0,
        hybrid_days_in_office=2,
        employer_flexibility_score=0.7,
    )
    # Remaining for MOVE = 2500 - (1700 + 700) = $100. Min reserve = $300.
    constraints_tight = UserConstraints(min_savings_buffer=300.0, max_acceptable_commute_minutes=90.0)
    out_4 = aggregator.evaluate_decision(
        scenario_text="Tight budget scenario.",
        facts=facts_tight,
        constraints=constraints_tight,
        preferences=pref_wlb,
    )
    print("\n[CASE 4: Budget Constraint Violation on MOVE]")
    print(f"MOVE Feasible: {out_4.evaluations['MOVE'].is_feasible}")
    print(f"MOVE Violations: {out_4.evaluations['MOVE'].constraint_violations}")
    assert not out_4.evaluations["MOVE"].is_feasible
    print("Hard constraint verified: MOVE was rejected because net funds ($100) < reserve ($300)!")

    # -------------------------------------------------------------
    # Case 5: Missing Information triggers REQUEST_CLARIFICATION
    # -------------------------------------------------------------
    facts_missing = StructuredFacts(
        monthly_stipend=None,  # Unknown stipend!
        base_living_expenses=700.0,
        relocation_rent=1500.0,
        commute_daily_minutes=60.0,
        commute_monthly_cost=150.0,
    )
    out_5 = aggregator.evaluate_decision(
        scenario_text="Offer without known stipend.",
        facts=facts_missing,
        constraints=constraints_1,
        preferences=pref_wlb,
    )
    print("\n[CASE 5: Missing Critical Information]")
    print(f"Decision Status: {out_5.decision_status}")
    print(f"Recommendation:  {out_5.recommendation.value}")
    print(f"Missing Facts:   {out_5.missing_critical_facts}")
    print(f"Prompt:          {out_5.clarification_prompt}")
    assert out_5.decision_status == "REQUEST_CLARIFICATION"
    assert out_5.recommendation == ActionType.REQUEST_INFO
    assert "monthly_stipend" in out_5.missing_critical_facts
    print("Clarification policy verified: Successfully requested clarification without guessing!")

    print("\n======================================================================")
    print("         ALL DECISION ENGINE TEST CASES PASSED SUCCESSFULLY!          ")
    print("======================================================================")


if __name__ == "__main__":
    run_demonstration()
