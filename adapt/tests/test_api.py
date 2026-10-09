import pytest
from fastapi.testclient import TestClient
from adapt.src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "supported_actions" in data


def test_decision_endpoint(client):
    payload = {
        "scenario_text": "Software Engineering Intern offer in Seattle.",
        "facts": {
            "monthly_stipend": 6000.0,
            "base_living_expenses": 750.0,
            "relocation_rent": 1600.0,
            "commute_daily_minutes": 80.0,
            "commute_monthly_cost": 220.0,
            "hybrid_days_in_office": 2,
            "employer_flexibility_score": 0.6,
            "city_cost_tier": "HIGH",
            "commute_mode": "TRANSIT",
        },
        "constraints": {
            "min_savings_buffer": 300.0,
            "max_acceptable_commute_minutes": 100.0,
        },
        "preferences": {
            "financial_weight": 0.4,
            "work_life_balance_weight": 0.4,
            "career_growth_weight": 0.2,
        },
    }
    response = client.post("/decision", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "recommendation" in res
    assert "decision_status" in res
    assert res["decision_status"] == "RECOMMENDED"
    assert "evaluations" in res


def test_feedback_endpoint(client):
    payload = {
        "example_id": "test_ex_100",
        "recommended_action": "MOVE",
        "user_selected_action": "MOVE",
        "user_satisfaction_score": 4.0,
        "user_comments": "Worked well.",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 201
    res = response.json()
    assert res["status"] == "recorded"
    assert "feedback_id" in res
    assert res["review_status"] == "staged"
