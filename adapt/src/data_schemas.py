"""
Data schemas for ADAPT (Adaptive Decision and Planning Transformer).
Defines strongly-typed Pydantic schemas for scenarios, candidate actions,
consequences, constraints, preferences, and feedback records.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    MOVE = "MOVE"
    COMMUTE = "COMMUTE"
    NEGOTIATE_HYBRID = "NEGOTIATE_HYBRID"
    REQUEST_INFO = "REQUEST_INFO"


class CityCostTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CommuteMode(str, Enum):
    TRANSIT = "TRANSIT"
    DRIVING = "DRIVING"
    COMBINED = "COMBINED"


class StructuredFacts(BaseModel):
    monthly_stipend: Optional[float] = Field(None, description="Monthly internship stipend in USD")
    base_living_expenses: Optional[float] = Field(None, description="Base living expenses (groceries, utilities) in USD")
    relocation_rent: Optional[float] = Field(None, description="Monthly rent near workplace in USD")
    commute_daily_minutes: Optional[float] = Field(None, description="Round-trip commute minutes if commuting")
    commute_monthly_cost: Optional[float] = Field(None, description="Monthly transit or fuel/parking cost in USD")
    hybrid_days_in_office: Optional[int] = Field(2, description="Days required in office if hybrid negotiated")
    employer_flexibility_score: Optional[float] = Field(0.5, description="Employer flexibility for remote work [0.0 - 1.0]")
    city_cost_tier: CityCostTier = Field(CityCostTier.MEDIUM, description="Cost of living tier")
    commute_mode: CommuteMode = Field(CommuteMode.TRANSIT, description="Mode of commuting")


class UserConstraints(BaseModel):
    min_savings_buffer: float = Field(200.0, description="Minimum net remaining dollars required monthly")
    max_acceptable_commute_minutes: float = Field(90.0, description="Maximum acceptable round-trip commute minutes")
    requires_in_person_mentorship: bool = Field(False, description="Whether daily in-person mentorship is required")


class UserPreferences(BaseModel):
    financial_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight placed on financial savings")
    work_life_balance_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight placed on low commute/free time")
    career_growth_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight placed on mentorship & in-person impact")


class ActionConsequences(BaseModel):
    net_remaining_funds: Optional[float] = None
    daily_commute_minutes: Optional[float] = None
    fatigue_index: Optional[float] = None  # Scale 0 to 10
    burnout_risk: Optional[float] = None    # Scale 0.0 to 1.0
    constraint_violations: List[str] = Field(default_factory=list)
    is_feasible: bool = True


class DecisionExample(BaseModel):
    example_id: str
    scenario_text: str
    structured_facts: StructuredFacts
    user_constraints: UserConstraints
    user_preferences: UserPreferences
    candidate_actions: List[ActionType] = Field(
        default_factory=lambda: [ActionType.MOVE, ActionType.COMMUTE, ActionType.NEGOTIATE_HYBRID, ActionType.REQUEST_INFO]
    )
    action_consequences: Dict[str, ActionConsequences] = Field(default_factory=dict)
    ground_truth_ranking: List[ActionType]
    optimal_action: ActionType
    should_clarify: bool = False
    missing_critical_facts: List[str] = Field(default_factory=list)
    dataset_split: str = "train"  # 'train', 'val', 'test'
    provenance: str = "synthetic_rule_grounded_v1"


class FeedbackRecord(BaseModel):
    feedback_id: str
    example_id: str
    timestamp: str
    model_version: str
    recommended_action: ActionType
    user_selected_action: ActionType
    user_satisfaction_score: float = Field(..., ge=1.0, le=5.0)
    user_comments: Optional[str] = None
    verified_outcome: Optional[str] = None
    provenance: str = "user_feedback"
    review_status: str = "staged"  # 'staged', 'reviewed', 'rejected', 'approved'
