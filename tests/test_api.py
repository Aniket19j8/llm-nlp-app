from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_sentiment():
    r = client.post("/v1/sentiment", json={"text": "I love this!"})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data and "score" in data
