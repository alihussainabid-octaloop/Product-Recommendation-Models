from fastapi.testclient import TestClient
from final_main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommend_unauthenticated():
    response = client.post("/model/recommend", data={"review": "great product"})
    assert response.status_code == 401
