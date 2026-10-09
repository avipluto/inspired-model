"""
Uncertainty-Aware Clarification and Abstention Policy for ADAPT.
Evaluates missing facts, candidate entropy, and consequence prediction variance
to determine when the system must request clarification or abstain from deciding.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from adapt.src.data_schemas import (
    ActionType,
    StructuredFacts,
    ActionConsequences,
)


class ClarificationPolicy:
    @staticmethod
    def evaluate_missing_information(
        facts: StructuredFacts,
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Check if critical variables required to make an informed decision are absent.
        Returns: (should_clarify, missing_fields, clarification_prompt)
        """
        missing = []
        if facts.monthly_stipend is None:
            missing.append("monthly_stipend")
        if facts.relocation_rent is None:
            missing.append("relocation_rent")
        if facts.commute_daily_minutes is None:
            missing.append("commute_daily_minutes")

        if len(missing) > 0:
            questions = []
            if "monthly_stipend" in missing:
                questions.append("What is the exact monthly internship stipend offered?")
            if "relocation_rent" in missing:
                questions.append("What is the expected monthly rent for housing near the office?")
            if "commute_daily_minutes" in missing:
                questions.append("What is the estimated round-trip commute duration from your current home?")

            prompt = "Critical information is missing before a safe decision can be recommended: " + " ".join(questions)
            return True, missing, prompt

        return False, [], None


class AbstentionPolicy:
    @staticmethod
    def evaluate_abstention(
        action_scores: Dict[ActionType, float],
        feasible_actions: List[ActionType],
        consequence_uncertainty: Dict[str, float] = None,
        entropy_threshold: float = 1.35,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if the system should abstain due to severe ambiguity or high uncertainty.
        - If no action is feasible: Abstain (No feasible options exist).
        - If top feasible options have nearly identical utility with high entropy: Warn or abstain.
        """
        if len(feasible_actions) == 0:
            return True, "ABSTAIN: All candidate actions violate hard user constraints or lack required data."

        if len(feasible_actions) == 1 and feasible_actions[0] == ActionType.REQUEST_INFO:
            return False, None

        # Calculate softmax entropy over feasible actions
        scores = np.array([action_scores[a] for a in feasible_actions])
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # Check for tie between top 2 feasible actions
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.02 and entropy > entropy_threshold:
            return True, (
                f"ABSTAIN_HIGH_UNCERTAINTY: Top feasible options ({feasible_actions[0].value} and {feasible_actions[1].value}) "
                f"are within 2% margin with high decision entropy ({entropy:.2f}). Human review recommended."
            )

        return False, None
