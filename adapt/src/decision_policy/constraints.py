"""
Hard Constraints Verification Engine for ADAPT.
Evaluates candidate actions against user-specified hard constraints:
- Minimum monthly budget reserve
- Maximum acceptable commute duration
- Employer policy compatibility
- Mentorship requirements
"""

from typing import Dict, List, Tuple
from adapt.src.data_schemas import (
    ActionType,
    StructuredFacts,
    UserConstraints,
    ActionConsequences,
)


class ConstraintChecker:
    @staticmethod
    def verify_constraints(
        facts: StructuredFacts,
        constraints: UserConstraints,
        consequences: Dict[str, ActionConsequences],
    ) -> Dict[str, ActionConsequences]:
        """
        Verify hard constraints for each action. Modifies constraint_violations
        and is_feasible in-place and returns the updated consequences dict.
        """
        for act_str, cons in consequences.items():
            violations: List[str] = list(cons.constraint_violations)

            # Skip already failed actions due to missing data
            if not cons.is_feasible and any("MISSING_DATA" in v for v in violations):
                continue

            # 1. Budget Constraint Check
            if cons.net_remaining_funds is not None and act_str != ActionType.REQUEST_INFO.value:
                if cons.net_remaining_funds < constraints.min_savings_buffer:
                    violations.append(
                        f"VIOLATES_BUDGET: Net disposable ${cons.net_remaining_funds:.0f} "
                        f"is below required reserve ${constraints.min_savings_buffer:.0f}"
                    )

            # 2. Commute Limit Check
            if cons.daily_commute_minutes is not None and act_str != ActionType.REQUEST_INFO.value:
                if cons.daily_commute_minutes > constraints.max_acceptable_commute_minutes:
                    violations.append(
                        f"EXCEEDS_MAX_COMMUTE: Daily commute {cons.daily_commute_minutes:.0f}m "
                        f"exceeds limit {constraints.max_acceptable_commute_minutes:.0f}m"
                    )

            # 3. Policy Compatibility Check
            if act_str == ActionType.NEGOTIATE_HYBRID.value:
                flex = facts.employer_flexibility_score or 0.5
                if flex < 0.30:
                    violations.append(
                        f"POLICY_INCOMPATIBLE: Employer remote flexibility score {flex:.2f} "
                        f"is too rigid for hybrid agreement"
                    )
                if constraints.requires_in_person_mentorship and (facts.hybrid_days_in_office or 2) < 3:
                    violations.append(
                        "MENTORSHIP_MISMATCH: In-person mentorship requirement unsatisfied by < 3 days in office"
                    )

            cons.constraint_violations = violations
            cons.is_feasible = len(violations) == 0

        return consequences
