import pytest
from adapt.src.data_schemas import (
    ActionType,
    CityCostTier,
    CommuteMode,
    StructuredFacts,
    UserConstraints,
    UserPreferences,
)
from adapt.src.consequences.calculator import ConsequenceCalculator
from adapt.src.decision_policy.aggregator import DecisionAggregator


def test_decision_aggregator_logic():
    calc = ConsequenceCalculator()
    aggregator = DecisionAggregator(calculator=calc)

    # 1. Normal feasible case
    facts = StructuredFacts(
        monthly_stipend=6000.0,
        base_living_expenses=700.0,
        relocation_rent=1600.0,
        commute_daily_minutes=90.0,
        commute_monthly_cost=250.0,
        hybrid_days_in_office=2,
        employer_flexibility_score=0.6,
        city_cost_tier=CityCostTier.HIGH,
        commute_mode=CommuteMode.TRANSIT,
    )
    constraints = UserConstraints(min_savings_buffer=400.0, max_acceptable_commute_minutes=120.0)
    prefs = UserPreferences(financial_weight=0.3, work_life_balance_weight=0.5, career_growth_weight=0.2)

    res = aggregator.evaluate_decision("Test scenario", facts, constraints, prefs)
    assert res.decision_status == "RECOMMENDED"
    assert res.recommendation in [ActionType.MOVE, ActionType.NEGOTIATE_HYBRID]
    assert len(res.supporting_factors) > 0


def test_decision_aggregator_missing_facts():
    calc = ConsequenceCalculator()
    aggregator = DecisionAggregator(calculator=calc)

    facts = StructuredFacts(
        monthly_stipend=None,  # Missing!
        base_living_expenses=700.0,
        relocation_rent=1600.0,
    )
    constraints = UserConstraints()
    prefs = UserPreferences()

    res = aggregator.evaluate_decision("Missing stipend", facts, constraints, prefs)
    assert res.decision_status == "REQUEST_CLARIFICATION"
    assert res.recommendation == ActionType.REQUEST_INFO
    assert "monthly_stipend" in res.missing_critical_facts
    assert res.clarification_prompt is not None


def test_decision_aggregator_budget_constraint():
    calc = ConsequenceCalculator()
    aggregator = DecisionAggregator(calculator=calc)

    facts = StructuredFacts(
        monthly_stipend=2000.0,
        base_living_expenses=700.0,
        relocation_rent=1600.0,  # 2000 - 2300 = -$300 deficit!
        commute_daily_minutes=40.0,
        commute_monthly_cost=100.0,
    )
    constraints = UserConstraints(min_savings_buffer=200.0)
    prefs = UserPreferences()

    res = aggregator.evaluate_decision("Deficit scenario", facts, constraints, prefs)
    assert not res.evaluations["MOVE"].is_feasible
    assert any("VIOLATES_BUDGET" in v for v in res.evaluations["MOVE"].constraint_violations)
