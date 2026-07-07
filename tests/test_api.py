"""
API tests. Require the model artifacts to exist (run `python src/train.py`
once beforehand) since /predict needs a loaded model - mirrors how you'd
gate integration tests in CI after a training step.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.main import app  # noqa: E402
from api.schemas import CustomerFeatures  # noqa: E402

MODEL_EXISTS = (Path(__file__).resolve().parent.parent / "models" / "churn_model.pkl").exists()

SAMPLE_CUSTOMER = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the app's lifespan
    # (startup/shutdown) handlers - without `with`, the model never loads
    # and every request would 503.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


@pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet - run src/train.py")
def test_predict_returns_valid_probability(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_label"] in {"low", "medium", "high"}


@pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet - run src/train.py")
def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "best_model" in response.json()


def test_predict_rejects_invalid_payload(client):
    bad_payload = dict(SAMPLE_CUSTOMER)
    bad_payload["Contract"] = "Not a real contract"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
