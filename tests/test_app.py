import sys
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# torch and transformers are 3GB — replace with fakes so CI runs in seconds
mock_transformers = MagicMock()
mock_transformers.pipeline.return_value = MagicMock(
    return_value=[{"label": "POSITIVE", "score": 0.9998}]
)
sys.modules["transformers"] = mock_transformers
sys.modules["torch"] = MagicMock()

from app.main import app

client = TestClient(app)


# Gate 1: is the service reachable?
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


# Gate 2: does predict return the expected fields?
def test_predict_returns_sentiment():
    response = client.post("/predict", json={"text": "I love this course!"})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()


# Gate 3: is bad input rejected before reaching the model?
def test_empty_text_is_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
