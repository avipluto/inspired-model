"""
Governed Feedback Store for ADAPT.
Collects user ratings, preference labels, and verified empirical outcomes into a quarantined staging store.
Enforces that feedback CANNOT update live model weights directly.
Provides review, verification, and approved dataset export for offline retraining.
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from adapt.src.data_schemas import FeedbackRecord, ActionType


class FeedbackStore:
    def __init__(self, store_path: str = "adapt/data/feedback_staging.jsonl"):
        self.store_path = store_path
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, "w", encoding="utf-8") as f:
                pass

    def record_feedback(
        self,
        example_id: str,
        model_version: str,
        recommended_action: ActionType,
        user_selected_action: ActionType,
        user_satisfaction_score: float,
        user_comments: Optional[str] = None,
        verified_outcome: Optional[str] = None,
        provenance: str = "user_feedback",
    ) -> FeedbackRecord:
        """Record new feedback into quarantined staging buffer."""
        record_id = f"fb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        record = FeedbackRecord(
            feedback_id=record_id,
            example_id=example_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=model_version,
            recommended_action=recommended_action,
            user_selected_action=user_selected_action,
            user_satisfaction_score=user_satisfaction_score,
            user_comments=user_comments,
            verified_outcome=verified_outcome,
            provenance=provenance,
            review_status="staged",  # Quarantined until human / expert review
        )

        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

        return record

    def list_feedback(self, status: Optional[str] = None) -> List[FeedbackRecord]:
        """List all feedback records, optionally filtered by review status."""
        records = []
        if not os.path.exists(self.store_path):
            return records

        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    rec = FeedbackRecord.model_validate(data)
                    if status is None or rec.review_status == status:
                        records.append(rec)
        return records

    def update_review_status(
        self,
        feedback_id: str,
        new_status: str,
        verified_outcome: Optional[str] = None,
    ) -> bool:
        """Update review status ('staged', 'reviewed', 'approved', 'rejected')."""
        assert new_status in ["staged", "reviewed", "approved", "rejected"]
        all_recs = []
        found = False

        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data["feedback_id"] == feedback_id:
                            data["review_status"] = new_status
                            if verified_outcome:
                                data["verified_outcome"] = verified_outcome
                            found = True
                        all_recs.append(data)

        if found:
            with open(self.store_path, "w", encoding="utf-8") as f:
                for d in all_recs:
                    f.write(json.dumps(d) + "\n")

        return found

    def export_approved_for_retraining(self, output_path: str) -> int:
        """Export only reviewed and approved records for offline retraining."""
        approved = self.list_feedback(status="approved")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in approved:
                f.write(json.dumps(item.model_dump()) + "\n")
        return len(approved)
