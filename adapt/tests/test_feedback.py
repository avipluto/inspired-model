import os
import pytest
from adapt.src.feedback.feedback_store import FeedbackStore
from adapt.src.data_schemas import ActionType


def test_feedback_store_workflow(tmp_path):
    store_file = os.path.join(tmp_path, "feedback_test.jsonl")
    store = FeedbackStore(store_file)

    rec = store.record_feedback(
        example_id="ex_001",
        model_version="v1.0",
        recommended_action=ActionType.MOVE,
        user_selected_action=ActionType.MOVE,
        user_satisfaction_score=4.5,
        user_comments="Good recommendation.",
    )
    assert rec.review_status == "staged"

    # Transition to approved
    ok = store.update_review_status(rec.feedback_id, "approved", verified_outcome="OUTCOME_VERIFIED")
    assert ok

    approved_list = store.list_feedback("approved")
    assert len(approved_list) == 1
    assert approved_list[0].verified_outcome == "OUTCOME_VERIFIED"
