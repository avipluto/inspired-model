"""
FastAPI Service for ADAPT (Adaptive Decision and Planning Transformer).
Endpoints:
- GET  /health
- GET  /model-info
- POST /decision
- POST /feedback
"""

import os
import sys
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adapt.src.data_schemas import (
    ActionType,
    CityCostTier,
    CommuteMode,
    StructuredFacts,
    UserConstraints,
    UserPreferences,
    FeedbackRecord,
)
from adapt.src.structured_predictor.preprocessor import StructuredFeaturePreprocessor
from adapt.src.structured_predictor.mlp_predictor import ConsequenceMLP
from adapt.src.consequences.calculator import ConsequenceCalculator
from adapt.src.decision_policy.aggregator import DecisionAggregator, DecisionEngineOutput
from adapt.src.feedback.feedback_store import FeedbackStore
import torch


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(application: FastAPI):
    global aggregator, feedback_store
    prep_path = os.path.join("adapt", "artifacts", "checkpoints", "preprocessor_state.json")
    mlp_path = os.path.join("adapt", "artifacts", "checkpoints", "mlp_consequence_best.pt")

    if os.path.exists(prep_path) and os.path.exists(mlp_path):
        preprocessor = StructuredFeaturePreprocessor()
        preprocessor.load(prep_path)

        mlp_model = ConsequenceMLP(input_dim=16, hidden_dims=(64, 32, 16))
        ckpt = torch.load(mlp_path, map_location="cpu")
        mlp_model.load_state_dict(ckpt["model_state_dict"])
        mlp_model.eval()

        calculator = ConsequenceCalculator(mlp_model=mlp_model, preprocessor=preprocessor)
        aggregator = DecisionAggregator(calculator=calculator)
    else:
        calculator = ConsequenceCalculator()
        aggregator = DecisionAggregator(calculator=calculator)

    feedback_store = FeedbackStore(os.path.join("adapt", "data", "feedback_staging.jsonl"))
    yield


app = FastAPI(
    title="ADAPT Decision Engine API",
    description="Adaptive Decision and Planning Transformer for multi-criteria career and logistics decisions.",
    version="0.1.0",
    lifespan=lifespan,
)

# Global model state
aggregator: Optional[DecisionAggregator] = None
feedback_store: Optional[FeedbackStore] = None
MODEL_VERSION = "ADAPT-Hybrid-v1.0"


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "ADAPT Decision Engine",
        "model_version": MODEL_VERSION,
        "engine_ready": aggregator is not None,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "model_architecture": "Hybrid (Compact Transformer + Structured Consequence MLP + Deterministic Aggregator)",
        "model_version": MODEL_VERSION,
        "components": {
            "component_a_transformer": "Scratch-Trained Causal Transformer (~1.6M parameters)",
            "component_b_consequence_mlp": "Multi-Task Consequence Predictor with MC Dropout",
            "component_c_aggregator": "Exact Deterministic Arithmetic & Hard Constraint Engine",
            "component_d_feedback": "Governed Quarantine & Offline Retraining Gate",
        },
        "supported_actions": [a.value for a in ActionType],
        "device": "cpu",
    }


class DecisionRequest(BaseModel):
    scenario_text: str = Field(..., description="Natural language description of the scenario")
    facts: StructuredFacts = Field(..., description="Structured facts and parameters")
    constraints: Optional[UserConstraints] = Field(default_factory=UserConstraints)
    preferences: Optional[UserPreferences] = Field(default_factory=UserPreferences)


class FeedbackRequest(BaseModel):
    example_id: str
    recommended_action: ActionType
    user_selected_action: ActionType
    user_satisfaction_score: float = Field(..., ge=1.0, le=5.0)
    user_comments: Optional[str] = None
    verified_outcome: Optional[str] = None


@app.post("/decision", response_model=DecisionEngineOutput, tags=["Decision Support"])
def make_decision(req: DecisionRequest):
    if aggregator is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")

    constraints = req.constraints or UserConstraints()
    preferences = req.preferences or UserPreferences()

    try:
        output = aggregator.evaluate_decision(
            scenario_text=req.scenario_text,
            facts=req.facts,
            constraints=constraints,
            preferences=preferences,
        )
        output.model_version = MODEL_VERSION
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating decision: {str(e)}")


@app.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["Feedback & Governance"])
def submit_feedback(fb: FeedbackRequest):
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")

    record = feedback_store.record_feedback(
        example_id=fb.example_id,
        model_version=MODEL_VERSION,
        recommended_action=fb.recommended_action,
        user_selected_action=fb.user_selected_action,
        user_satisfaction_score=fb.user_satisfaction_score,
        user_comments=fb.user_comments,
        verified_outcome=fb.verified_outcome,
    )

    return {
        "status": "recorded",
        "feedback_id": record.feedback_id,
        "review_status": record.review_status,
        "notice": "Feedback safely staged in quarantined store. Live weights are never updated online.",
    }
