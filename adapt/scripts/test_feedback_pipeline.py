"""
Test script for the Feedback and Learning Pipeline.
Tests:
1. Ingestion of user feedback into quarantined staging buffer.
2. Review status transitions ('staged' -> 'reviewed' -> 'approved' / 'rejected').
3. Noisy feedback rejection.
4. Export of approved feedback records.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from adapt.src.feedback.feedback_store import FeedbackStore
from adapt.src.data_schemas import ActionType


def test_feedback_store():
    print("--- Testing Feedback Store & Safety Gates ---")
    store_file = os.path.join("adapt", "data", "test_feedback_staging.jsonl")
    if os.path.exists(store_file):
        os.remove(store_file)

    store = FeedbackStore(store_file)

    # 1. Record legitimate feedback
    fb1 = store.record_feedback(
        example_id="train_00012",
        model_version="ADAPT-v1.0",
        recommended_action=ActionType.MOVE,
        user_selected_action=ActionType.MOVE,
        user_satisfaction_score=5.0,
        user_comments="Relocated and found a good flat share. Budget matched prediction.",
        verified_outcome="CONFIRMED_SATISFACTORY",
    )
    print(f"Recorded feedback fb1: id={fb1.feedback_id}, status={fb1.review_status}")
    assert fb1.review_status == "staged"

    # 2. Record noisy/conflicting feedback
    fb2 = store.record_feedback(
        example_id="train_00045",
        model_version="ADAPT-v1.0",
        recommended_action=ActionType.COMMUTE,
        user_selected_action=ActionType.MOVE,
        user_satisfaction_score=1.0,
        user_comments="Spam noise test 12345",
        verified_outcome=None,
    )
    print(f"Recorded noisy feedback fb2: id={fb2.feedback_id}, status={fb2.review_status}")

    # 3. Review workflow
    store.update_review_status(fb1.feedback_id, "approved", verified_outcome="EMPIRICALLY_VERIFIED")
    store.update_review_status(fb2.feedback_id, "rejected", verified_outcome="NOISY_SPAM")

    # 4. Verify filtration
    staged = store.list_feedback("staged")
    approved = store.list_feedback("approved")
    rejected = store.list_feedback("rejected")

    print(f"Filtered status counts: Staged={len(staged)}, Approved={len(approved)}, Rejected={len(rejected)}")
    assert len(staged) == 0
    assert len(approved) == 1
    assert len(rejected) == 1
    assert approved[0].feedback_id == fb1.feedback_id

    # 5. Export approved items
    export_path = os.path.join("adapt", "data", "test_approved_export.jsonl")
    num_exp = store.export_approved_for_retraining(export_path)
    print(f"Exported {num_exp} approved records to {export_path}")
    assert num_exp == 1

    print("--- Feedback Pipeline Tests Passed Successfully! ---")


if __name__ == "__main__":
    test_feedback_store()
