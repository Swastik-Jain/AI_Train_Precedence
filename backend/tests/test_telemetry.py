from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_telemetry_endpoint():
    res = client.get("/api/v1/telemetry")
    assert res.status_code == 200, "Telemetry endpoint should return 200 OK"
