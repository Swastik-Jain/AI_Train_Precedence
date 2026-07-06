from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_sandbox_block_worsens_or_matches_baseline_impact():
    baseline = client.post(
        "/api/v1/simulation/analyze", json={"latencies": {}, "forced_actions": {}}
    ).json()["impact"]

    client.post("/api/v1/sandbox/blocks", json={
        "element_id": "edge-1-2",
        "type": "TRACK_SEGMENT",
        "start_time": "2000-01-01T00:00:00Z",
        "end_time": "2099-01-01T00:00:00Z",
        "severity": "TOTAL_BLOCK",
        "reason": "test_sandbox_block_worsens_or_matches_baseline_impact",
    })

    scenario = client.post(
        "/api/v1/simulation/analyze", json={"latencies": {}, "forced_actions": {}}
    ).json()["impact"]

    client.delete("/api/v1/sandbox/blocks/edge-1-2")

    # A total block on a live segment should never produce fewer total-delay
    # minutes than a scenario with no block at all.
    assert scenario["scenario_delay"] >= baseline["baseline_delay"]
