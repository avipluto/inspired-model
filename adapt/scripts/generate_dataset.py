"""
Dataset Generator for ADAPT: Internship Decision Support.
Generates train, validation, and held-out test splits with deterministic arithmetic,
rule-grounded constraints, preference weighting, and missing-information clarification cases.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.data_schemas import (
    ActionType,
    CityCostTier,
    CommuteMode,
    StructuredFacts,
    UserConstraints,
    UserPreferences,
    ActionConsequences,
    DecisionExample,
)


CITIES = [
    ("San Francisco", CityCostTier.HIGH, 1800, 2600, 700, 1100),
    ("New York", CityCostTier.HIGH, 1700, 2500, 650, 1050),
    ("Seattle", CityCostTier.HIGH, 1500, 2200, 600, 950),
    ("Austin", CityCostTier.MEDIUM, 1100, 1600, 500, 800),
    ("Chicago", CityCostTier.MEDIUM, 1000, 1500, 500, 750),
    ("Atlanta", CityCostTier.MEDIUM, 950, 1400, 450, 700),
    ("Raleigh", CityCostTier.LOW, 800, 1200, 400, 650),
    ("Pittsburgh", CityCostTier.LOW, 750, 1100, 380, 600),
    ("Columbus", CityCostTier.LOW, 700, 1050, 360, 580),
]

ROLES = [
    "Software Engineering Intern",
    "Data Science Intern",
    "Product Management Intern",
    "UX Design Intern",
    "Machine Learning Research Intern",
    "Business Analyst Intern",
]


def generate_single_scenario(
    example_id: str,
    split: str,
    rng: random.Random,
    force_missing: bool = False,
) -> DecisionExample:
    city_name, cost_tier, rent_min, rent_max, exp_min, exp_max = rng.choice(CITIES)
    role = rng.choice(ROLES)
    commute_mode = rng.choice([CommuteMode.TRANSIT, CommuteMode.DRIVING, CommuteMode.COMBINED])
    
    stipend = float(rng.randint(2800, 7500))
    base_living = float(rng.randint(exp_min, exp_max))
    rent = float(rng.randint(rent_min, rent_max))
    commute_mins = float(rng.randint(30, 150))
    commute_cost = float(rng.randint(120, 480))
    employer_flex = float(round(rng.uniform(0.1, 0.95), 2))
    hybrid_days = rng.choice([1, 2, 3])
    
    # Missing information probability (to test clarification vs abstention)
    missing_facts: List[str] = []
    should_clarify = False
    
    if force_missing:
        missing_choice = rng.choice(["stipend", "rent", "commute"])
        if missing_choice == "stipend":
            stipend = None
            missing_facts.append("monthly_stipend")
        elif missing_choice == "rent":
            rent = None
            missing_facts.append("relocation_rent")
        elif missing_choice == "commute":
            commute_mins = None
            commute_cost = None
            missing_facts.append("commute_daily_minutes")
        should_clarify = True

    # User Constraints
    min_savings = float(rng.choice([150.0, 300.0, 500.0, 800.0]))
    max_commute = float(rng.choice([60.0, 75.0, 90.0, 120.0]))
    req_mentorship = rng.choice([True, False, False])

    # User Preferences (sum to 1.0)
    w_fin = rng.uniform(0.2, 0.6)
    w_wlb = rng.uniform(0.2, 0.5)
    w_car = rng.uniform(0.1, 0.4)
    total_w = w_fin + w_wlb + w_car
    pref_fin = round(w_fin / total_w, 2)
    pref_wlb = round(w_wlb / total_w, 2)
    pref_car = round(1.0 - (pref_fin + pref_wlb), 2)

    structured_facts = StructuredFacts(
        monthly_stipend=stipend,
        base_living_expenses=base_living,
        relocation_rent=rent,
        commute_daily_minutes=commute_mins,
        commute_monthly_cost=commute_cost,
        hybrid_days_in_office=hybrid_days,
        employer_flexibility_score=employer_flex,
        city_cost_tier=cost_tier,
        commute_mode=commute_mode,
    )

    user_constraints = UserConstraints(
        min_savings_buffer=min_savings,
        max_acceptable_commute_minutes=max_commute,
        requires_in_person_mentorship=req_mentorship,
    )

    user_preferences = UserPreferences(
        financial_weight=pref_fin,
        work_life_balance_weight=pref_wlb,
        career_growth_weight=pref_car,
    )

    # Deterministic consequences and feasibility
    consequences: Dict[str, ActionConsequences] = {}
    action_utilities: Dict[ActionType, float] = {}

    # 1. MOVE
    if rent is not None and stipend is not None:
        move_rem = stipend - (rent + base_living)
        move_commute = 15.0
        move_fatigue = 1.2
        move_burnout = 0.15
        move_viols = []
        if move_rem < min_savings:
            move_viols.append(f"VIOLATES_BUDGET: Remaining ${move_rem:.0f} < ${min_savings:.0f}")
        consequences[ActionType.MOVE.value] = ActionConsequences(
            net_remaining_funds=move_rem,
            daily_commute_minutes=move_commute,
            fatigue_index=move_fatigue,
            burnout_risk=move_burnout,
            constraint_violations=move_viols,
            is_feasible=len(move_viols) == 0,
        )
    else:
        consequences[ActionType.MOVE.value] = ActionConsequences(
            constraint_violations=["MISSING_DATA: Missing rent or stipend"],
            is_feasible=False,
        )

    # 2. COMMUTE
    if commute_mins is not None and commute_cost is not None and stipend is not None:
        comm_rem = stipend - (commute_cost + base_living)
        comm_fatigue = min(10.0, commute_mins / 15.0)
        comm_burnout = min(0.95, (commute_mins / 120.0) * 0.75 + (0.2 if commute_mins > 100 else 0.0))
        comm_viols = []
        if comm_rem < min_savings:
            comm_viols.append(f"VIOLATES_BUDGET: Remaining ${comm_rem:.0f} < ${min_savings:.0f}")
        if commute_mins > max_commute:
            comm_viols.append(f"EXCEEDS_MAX_COMMUTE: {commute_mins:.0f}m > {max_commute:.0f}m limit")
        consequences[ActionType.COMMUTE.value] = ActionConsequences(
            net_remaining_funds=comm_rem,
            daily_commute_minutes=commute_mins,
            fatigue_index=round(comm_fatigue, 2),
            burnout_risk=round(comm_burnout, 2),
            constraint_violations=comm_viols,
            is_feasible=len(comm_viols) == 0,
        )
    else:
        consequences[ActionType.COMMUTE.value] = ActionConsequences(
            constraint_violations=["MISSING_DATA: Missing commute data"],
            is_feasible=False,
        )

    # 3. NEGOTIATE_HYBRID
    if commute_mins is not None and commute_cost is not None and stipend is not None:
        hybrid_fraction = hybrid_days / 5.0
        hyb_cost = commute_cost * hybrid_fraction
        hyb_rem = stipend - (hyb_cost + base_living)
        hyb_commute = commute_mins * hybrid_fraction
        hyb_fatigue = min(10.0, hyb_commute / 16.0)
        hyb_burnout = min(0.90, (hyb_commute / 130.0) * 0.6)
        hyb_viols = []
        if hyb_rem < min_savings:
            hyb_viols.append(f"VIOLATES_BUDGET: Remaining ${hyb_rem:.0f} < ${min_savings:.0f}")
        if employer_flex < 0.30:
            hyb_viols.append(f"POLICY_INCOMPATIBLE: Employer remote flexibility only {employer_flex:.2f}")
        if req_mentorship and hybrid_days < 3:
            hyb_viols.append("MENTORSHIP_MISMATCH: In-person mentorship requirement unsatisfied")
        consequences[ActionType.NEGOTIATE_HYBRID.value] = ActionConsequences(
            net_remaining_funds=hyb_rem,
            daily_commute_minutes=round(hyb_commute, 1),
            fatigue_index=round(hyb_fatigue, 2),
            burnout_risk=round(hyb_burnout, 2),
            constraint_violations=hyb_viols,
            is_feasible=len(hyb_viols) == 0,
        )
    else:
        consequences[ActionType.NEGOTIATE_HYBRID.value] = ActionConsequences(
            constraint_violations=["MISSING_DATA: Missing commute or stipend"],
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

    # Utility Scoring for Feasible Actions
    # Normalize savings between 0 and 1 using benchmark $3000 max savings
    def calc_utility(action: ActionType) -> float:
        c = consequences[action.value]
        if not c.is_feasible:
            return -100.0 - len(c.constraint_violations) * 10.0
        
        if action == ActionType.REQUEST_INFO:
            return 10.0 if should_clarify else 0.1
        
        # Financial utility
        fin_score = min(1.0, max(0.0, (c.net_remaining_funds or 0.0) / 3000.0))
        # Work-life balance utility
        wlb_score = 1.0 - ((c.fatigue_index or 0.0) / 10.0)
        # Career growth utility
        if action == ActionType.MOVE:
            car_score = 1.0
        elif action == ActionType.COMMUTE:
            car_score = 0.85
        elif action == ActionType.NEGOTIATE_HYBRID:
            car_score = 0.65
        else:
            car_score = 0.5
            
        return pref_fin * fin_score + pref_wlb * wlb_score + pref_car * car_score

    candidate_actions = [ActionType.MOVE, ActionType.COMMUTE, ActionType.NEGOTIATE_HYBRID, ActionType.REQUEST_INFO]
    for act in candidate_actions:
        action_utilities[act] = calc_utility(act)

    ranked_actions = sorted(candidate_actions, key=lambda a: action_utilities[a], reverse=True)
    optimal_action = ranked_actions[0]

    # Generate realistic natural language scenario narrative
    parts = [
        f"Candidate received an offer for a {role} in {city_name} (Cost tier: {cost_tier.value}).",
    ]
    if stipend is not None:
        parts.append(f"The monthly stipend is ${stipend:.0f}.")
    else:
        parts.append("The monthly stipend has not been specified yet.")

    parts.append(f"Estimated baseline living costs are ${base_living:.0f}/month.")

    if rent is not None:
        parts.append(f"Apartment rent near the office is quoted at ${rent:.0f}/month.")
    else:
        parts.append("Local rental estimates have not been determined.")

    if commute_mins is not None:
        parts.append(
            f"Commuting from current home via {commute_mode.value.lower()} takes {commute_mins:.0f} mins round-trip and costs ${commute_cost:.0f}/month."
        )
    else:
        parts.append("Commute transit duration and travel expense remain unconfirmed.")

    parts.append(f"Employer remote policy flexibility rating is {employer_flex:.2f}.")
    parts.append(
        f"User constraints: reserve at least ${min_savings:.0f}/mo savings, max commute {max_commute:.0f} mins, in-person mentorship required: {req_mentorship}."
    )
    parts.append(
        f"User priorities: financial {pref_fin:.2f}, work-life balance {pref_wlb:.2f}, career impact {pref_car:.2f}."
    )

    scenario_text = " ".join(parts)

    return DecisionExample(
        example_id=example_id,
        scenario_text=scenario_text,
        structured_facts=structured_facts,
        user_constraints=user_constraints,
        user_preferences=user_preferences,
        candidate_actions=candidate_actions,
        action_consequences=consequences,
        ground_truth_ranking=ranked_actions,
        optimal_action=optimal_action,
        should_clarify=should_clarify,
        missing_critical_facts=missing_facts,
        dataset_split=split,
        provenance="synthetic_rule_grounded_v1",
    )


def generate_dataset(
    output_dir: str,
    num_train: int = 1600,
    num_val: int = 400,
    num_test: int = 500,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    splits = [
        ("train", num_train, False),
        ("val", num_val, False),
        ("test", num_test, True),  # Test split uses separate seed and evaluation flag
    ]

    for split_name, count, is_test in splits:
        if is_test:
            split_rng = random.Random(seed + 100)
        else:
            split_rng = rng

        records: List[Dict[str, Any]] = []
        for i in range(count):
            ex_id = f"{split_name}_{i:05d}"
            # Inject missing facts into ~15% of scenarios
            force_missing = split_rng.random() < 0.15
            example = generate_single_scenario(
                example_id=ex_id,
                split=split_name,
                rng=split_rng,
                force_missing=force_missing,
            )
            records.append(example.model_dump())

        file_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Generated {len(records)} examples in {file_path}")


if __name__ == "__main__":
    out_dir = os.path.join("adapt", "data", "processed")
    generate_dataset(output_dir=out_dir, num_train=1600, num_val=400, num_test=500, seed=42)
