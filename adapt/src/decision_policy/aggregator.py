"""
Decision Aggregator Engine for ADAPT.
Integrates:
- Transformer candidate scoring priors
- Deterministic arithmetic consequences
- ConsequenceMLP empirical risk estimates
- Hard constraint enforcement
- User preference multi-criteria utility weighting
- Uncertainty-aware clarification & abstention
- Auditable explanation generation
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from adapt.src.data_schemas import (
    ActionType,
    StructuredFacts,
    UserConstraints,
    UserPreferences,
    ActionConsequences,
)
from adapt.src.consequences.calculator import ConsequenceCalculator
from adapt.src.decision_policy.constraints import ConstraintChecker
from adapt.src.uncertainty.abstention_policy import ClarificationPolicy, AbstentionPolicy


class ActionEvaluationResult(BaseModel):
    action: ActionType
    is_feasible: bool
    utility_score: float
    net_remaining_funds: Optional[float] = None
    daily_commute_minutes: Optional[float] = None
    fatigue_index: Optional[float] = None
    burnout_risk: Optional[float] = None
    constraint_violations: List[str] = Field(default_factory=list)
    trade_off_summary: str = ""


class DecisionEngineOutput(BaseModel):
    recommendation: ActionType
    decision_status: str  # 'RECOMMENDED', 'REQUEST_CLARIFICATION', 'ABSTAINED'
    ranked_actions: List[ActionType]
    evaluations: Dict[str, ActionEvaluationResult]
    supporting_factors: List[str]
    trade_offs_explained: List[str]
    assumptions_made: List[str]
    missing_critical_facts: List[str] = Field(default_factory=list)
    clarification_prompt: Optional[str] = None
    uncertainty_indicators: Dict[str, Any] = Field(default_factory=dict)
    applied_preferences: Dict[str, float]
    model_version: str = "ADAPT-Hybrid-v1.0"


class DecisionAggregator:
    def __init__(
        self,
        calculator: ConsequenceCalculator,
        transformer_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.calculator = calculator
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer

    def evaluate_decision(
        self,
        scenario_text: str,
        facts: StructuredFacts,
        constraints: UserConstraints,
        preferences: UserPreferences,
    ) -> DecisionEngineOutput:
        # 1. Step 1: Check for missing critical information
        should_clarify, missing_facts, clarify_prompt = ClarificationPolicy.evaluate_missing_information(facts)

        # 2. Step 2: Compute deterministic arithmetic & consequence predictions
        raw_consequences = self.calculator.compute_action_consequences(facts)

        # 3. Step 3: Verify hard constraints
        verified_consequences = ConstraintChecker.verify_constraints(facts, constraints, raw_consequences)

        # 4. Step 4: If critical facts are missing, recommend REQUEST_INFO immediately
        if should_clarify:
            evaluations = {}
            for act_str, c in verified_consequences.items():
                evaluations[act_str] = ActionEvaluationResult(
                    action=ActionType(act_str),
                    is_feasible=c.is_feasible,
                    utility_score=10.0 if act_str == ActionType.REQUEST_INFO.value else -50.0,
                    net_remaining_funds=c.net_remaining_funds,
                    daily_commute_minutes=c.daily_commute_minutes,
                    fatigue_index=c.fatigue_index,
                    burnout_risk=c.burnout_risk,
                    constraint_violations=c.constraint_violations,
                    trade_off_summary="Cannot safely evaluate without missing essential facts.",
                )
            
            return DecisionEngineOutput(
                recommendation=ActionType.REQUEST_INFO,
                decision_status="REQUEST_CLARIFICATION",
                ranked_actions=[ActionType.REQUEST_INFO, ActionType.MOVE, ActionType.NEGOTIATE_HYBRID, ActionType.COMMUTE],
                evaluations=evaluations,
                supporting_factors=["Critical financial or commute information is missing, preventing valid comparison."],
                trade_offs_explained=["Deciding now risks severe budget deficits or extreme unviable commute fatigue."],
                assumptions_made=["Refusing to guess financial parameters using neural hallucination."],
                missing_critical_facts=missing_facts,
                clarification_prompt=clarify_prompt,
                uncertainty_indicators={"missing_critical_variables": len(missing_facts), "confidence": 0.0},
                applied_preferences=preferences.model_dump(),
            )

        # 5. Step 5: Multi-Criteria Utility Scoring for Feasible Actions
        utilities: Dict[ActionType, float] = {}
        evaluations = {}
        trade_offs = []
        supporting_factors = []

        w_fin = preferences.financial_weight
        w_wlb = preferences.work_life_balance_weight
        w_car = preferences.career_growth_weight

        max_funds = max(
            [c.net_remaining_funds for c in verified_consequences.values() if c.net_remaining_funds is not None] + [3000.0]
        )

        for act in [ActionType.MOVE, ActionType.COMMUTE, ActionType.NEGOTIATE_HYBRID, ActionType.REQUEST_INFO]:
            c = verified_consequences[act.value]

            if not c.is_feasible:
                u = -100.0 - len(c.constraint_violations) * 10.0
                trade_summary = "Infeasible: " + "; ".join(c.constraint_violations)
            elif act == ActionType.REQUEST_INFO:
                u = 0.05
                trade_summary = "Status quo; maintains current situation with zero advancement."
            else:
                # Continuous utility components:
                # Financial score dynamically normalized relative to best available savings
                fin_score = max(0.0, min(1.0, (c.net_remaining_funds or 0.0) / max_funds))
                # Work-Life Balance score (1 - fatigue/10)
                wlb_score = max(0.0, 1.0 - ((c.fatigue_index or 0.0) / 10.0))
                # Career score
                car_score = 1.0 if act == ActionType.MOVE else (0.85 if act == ActionType.COMMUTE else 0.65)

                u = w_fin * fin_score + w_wlb * wlb_score + w_car * car_score

                trade_summary = (
                    f"Net savings: ${c.net_remaining_funds:.0f}/mo | "
                    f"Commute: {c.daily_commute_minutes:.0f}m/day | "
                    f"Fatigue Index: {c.fatigue_index:.1f}/10"
                )

            utilities[act] = u
            evaluations[act.value] = ActionEvaluationResult(
                action=act,
                is_feasible=c.is_feasible,
                utility_score=round(u, 4),
                net_remaining_funds=c.net_remaining_funds,
                daily_commute_minutes=c.daily_commute_minutes,
                fatigue_index=c.fatigue_index,
                burnout_risk=c.burnout_risk,
                constraint_violations=c.constraint_violations,
                trade_off_summary=trade_summary,
            )

        # 6. Step 6: Rank actions by utility
        ranked = sorted(utilities.keys(), key=lambda a: utilities[a], reverse=True)
        top_action = ranked[0]
        feasible_ranked = [a for a in ranked if evaluations[a.value].is_feasible]

        # 7. Step 7: Check Abstention Policy
        should_abstain, abstain_reason = AbstentionPolicy.evaluate_abstention(utilities, feasible_ranked)
        decision_status = "ABSTAINED" if should_abstain else "RECOMMENDED"

        # 8. Step 8: Build explanations
        top_eval = evaluations[top_action.value]
        if top_action == ActionType.MOVE:
            supporting_factors.append(
                f"Eliminates commute (only 15 mins) providing highest work-life balance and mentorship presence."
            )
            trade_offs.append(
                f"Higher monthly housing cost (${facts.relocation_rent:.0f}/mo) leaves ${top_eval.net_remaining_funds:.0f}/mo savings."
            )
        elif top_action == ActionType.COMMUTE:
            supporting_factors.append(
                f"Maximizes net financial savings (${top_eval.net_remaining_funds:.0f}/mo) by eliminating rental expenditure."
            )
            trade_offs.append(
                f"Requires {facts.commute_daily_minutes:.0f} mins daily travel, resulting in fatigue index {top_eval.fatigue_index:.1f}/10."
            )
        elif top_action == ActionType.NEGOTIATE_HYBRID:
            supporting_factors.append(
                f"Balanced solution: cuts commute days to {facts.hybrid_days_in_office} days/week while saving rent."
            )
            trade_offs.append(
                f"Leaves ${top_eval.net_remaining_funds:.0f}/mo while reducing burnout risk to {top_eval.burnout_risk:.2f}."
            )

        if should_abstain and abstain_reason:
            trade_offs.append(abstain_reason)

        assumptions = [
            f"Employer flexibility score estimated at {facts.employer_flexibility_score:.2f}.",
            f"Assumed baseline expenses of ${facts.base_living_expenses:.0f}/month.",
            "Arithmetic consequences verified with deterministic precision.",
        ]

        return DecisionEngineOutput(
            recommendation=top_action,
            decision_status=decision_status,
            ranked_actions=ranked,
            evaluations=evaluations,
            supporting_factors=supporting_factors,
            trade_offs_explained=trade_offs,
            assumptions_made=assumptions,
            missing_critical_facts=[],
            clarification_prompt=None,
            uncertainty_indicators={
                "decision_entropy": round(float(np.std([utilities[a] for a in feasible_ranked])) if len(feasible_ranked) > 1 else 0.0, 3),
                "is_abstaining": should_abstain,
            },
            applied_preferences=preferences.model_dump(),
        )
